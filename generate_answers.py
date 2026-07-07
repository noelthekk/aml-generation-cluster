"""Generate answers for the RAGAS test set across all four retrieval configs.

Run this on the cluster. Model size defaults to `8b`, the primary generator used for
the four-config comparison (fixed regardless of hardware - see plan.md Sec 5,
"Generator model identity" - a controlled comparison across retrieval configs needs
the generator held constant); `70b` is a generator-scale ablation on top of the same
200 (config, query) pairs, added 2026-07-07 to check whether the retrieval-config
ranking holds with a stronger generator (see plan.md Sec 5, "Generator-scale ablation"
and plan/report.md).

Model size determines precision and the VRAM partition to request:
    8b  -> Llama-3.1-8B-Instruct,  BF16,       ~16GB weights   -> 18GB+ partition
    70b -> Llama-3.1-70B-Instruct, NF4 4-bit,  ~35-40GB weights -> 72GB partition
(70B in BF16 is ~140GB of weights and does not fit a 72GB partition, so it runs NF4
4-bit - same quantization the Qwen2.5-72B reference-drafting run used successfully on
this cluster. Caveat for the write-up: the 8B/70B comparison is then BF16-vs-NF4 as
well as 8B-vs-70B; quantization cost on a 70B is small but nonzero, so say so honestly
rather than presenting it as a pure scale comparison.)

Self-contained: no dependency on the rest of the aml-hybrid-rag project. Includes its
own copies of scripts/retriever.py, scripts/graph_build.py, and the data files they
need (clauses.jsonl, cross_refs.jsonl, test_set.jsonl); data/chroma_db/ is rebuilt
locally with scripts/build_chroma.py.

Before running:
    1. Copy .env.example to .env and set HF_TOKEN.
    2. uv sync
    3. uv run python scripts/build_chroma.py   (once, rebuilds the vector index)

Usage:
    uv run python generate_answers.py                  # defaults to --model-size 8b
    uv run python generate_answers.py --model-size 70b

Or via run.sh (backgrounded, survives SSH logout - see README.md):
    bash run.sh start        # defaults to 8b
    bash run.sh start 70b

Output: results/answers_<model-size>.jsonl - one row per (config, query) pair, 200
rows total (4 configs x 50 queries): query, gold_ids, query_type, config, retrieved
(clause_id list), answer, citations. Rows where the model's JSON output didn't parse
get an `answer` field starting with "Generation error:" and an empty `citations` list
- check for those first. The model-size suffix keeps runs from overwriting each other.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline as hf_pipeline_fn
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from retriever import (  # noqa: E402
    load_retrievers, dense_only_retrieve, sparse_only_retrieve,
    dense_sparse_retrieve, hybrid_retrieve,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("generate_answers")

DATA_DIR = Path("data")
CHROMA_DIR = DATA_DIR / "chroma_db"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
TEST_SET_PATH = DATA_DIR / "test_set.jsonl"

MODEL_IDS = {
    "8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "70b": "NousResearch/Meta-Llama-3.1-70B-Instruct",
}
MAX_NEW_TOKENS = 1024
TOP_K = 10
RRF_K = 60
GRAPH_HOPS = 2

ALL_CONFIGS = ["dense_only", "sparse_only", "dense_sparse", "hybrid"]

SYSTEM_PROMPT = """You are an AML compliance analyst. Answer using ONLY the regulatory clauses below.

Rules:
1. Every factual claim must be followed by [clause_id] inline.
2. State explicitly if the provided context does not answer the question.
3. Output valid JSON with two keys:
   - answer: your response with inline [clause_id] citations
   - citations: list of clause_id strings you cited

Context:
{context}"""


def format_context(results: list) -> str:
    parts = []
    for r in results:
        cid = r["clause_id"]
        text = r["text"][:800]
        parts.append(f"[{cid}]\n{text}")
    return "\n\n---\n\n".join(parts)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieve_for_config(config: str, query: str, vectorstore, bm25, clauses, G) -> list:
    if config == "dense_only":
        return dense_only_retrieve(vectorstore, clauses, query, k=TOP_K)
    if config == "sparse_only":
        return sparse_only_retrieve(bm25, clauses, query, k=TOP_K)
    if config == "dense_sparse":
        return dense_sparse_retrieve(query, vectorstore, bm25, clauses, k=TOP_K, rrf_k=RRF_K)
    if config == "hybrid":
        return hybrid_retrieve(query, vectorstore, bm25, clauses, G, k=TOP_K, graph_hops=GRAPH_HOPS, rrf_k=RRF_K)
    raise ValueError(f"unknown config: {config}")


def load_model(model_id: str, model_size: str):
    """8b loads BF16 (matches the primary run's precision); 70b loads NF4 4-bit
    (BF16 at 70B is ~140GB of weights - does not fit a 72GB partition)."""
    if model_size == "70b":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs = {"quantization_config": quant_config, "device_map": "auto"}
        precision = "NF4 4-bit"
    else:
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        precision = "BF16"

    logger.info("Loading %s (%s)...", model_id, precision)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    logger.info("Model loaded (%.1fs)  GPU: %s", time.time() - t0, torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    logger.info("VRAM free/total (GB): %d/%d", free // 1024**3, total // 1024**3)
    return tokenizer, model


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    arg_parser.add_argument(
        "--model-size",
        default="8b",
        choices=sorted(MODEL_IDS),
        help="8b (default) needs an 18GB+ partition (BF16); 70b needs a 72GB partition (NF4 4-bit)",
    )
    args = arg_parser.parse_args()
    model_id = MODEL_IDS[args.model_size]
    output_path = RESULTS_DIR / f"answers_{args.model_size}.jsonl"

    if not torch.cuda.is_available():
        raise EnvironmentError("No CUDA device found - this script requires a GPU.")

    logger.info("Model size: %s (%s)", args.model_size, model_id)
    logger.info("Output: %s", output_path)

    logger.info("Loading retrievers from %s", DATA_DIR)
    t0 = time.time()
    vectorstore, bm25, clauses, G = load_retrievers(DATA_DIR, CHROMA_DIR)
    logger.info("Retrievers loaded: %d clauses  (%.1fs)", len(clauses), time.time() - t0)

    test_set = load_jsonl(TEST_SET_PATH)
    logger.info("Loaded %d test queries", len(test_set))

    tokenizer, model = load_model(model_id, args.model_size)
    gen_pipe = hf_pipeline_fn(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False, return_full_text=False,
    )
    llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen_pipe))
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{query}"),
    ])
    chain = prompt | llm | parser

    pairs = [(config, item) for config in ALL_CONFIGS for item in test_set]
    n_errors = 0
    n_rows = 0
    with output_path.open("w", encoding="utf-8") as f:
        progress = tqdm(pairs, unit="query", dynamic_ncols=True)
        for config, item in progress:
            query = item["query"]
            progress.set_description(f"{config:<12} {query[:40]!r}")
            retrieved = retrieve_for_config(config, query, vectorstore, bm25, clauses, G)
            context = format_context(retrieved)
            try:
                parsed = chain.invoke({"query": query, "context": context})
                answer = parsed.get("answer", "")
                citations = parsed.get("citations", [])
            except Exception as exc:
                answer = f"Generation error: {exc}"
                citations = []
                n_errors += 1
                logger.warning("[%d/%d] GENERATION ERROR  config=%s  %r: %s",
                               n_rows + 1, len(pairs), config, query[:50], exc)

            row = {
                "query": query,
                "gold_ids": item["gold_ids"],
                "query_type": item.get("query_type", "unknown"),
                "config": config,
                "retrieved": [r["clause_id"] for r in retrieved],
                "answer": answer,
                "citations": citations,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_rows += 1

    logger.info(
        "Done: %d rows written to %s  (%d generation errors)",
        n_rows, output_path, n_errors,
    )
    if n_errors:
        logger.warning("Check the %d 'Generation error:' rows first.", n_errors)


if __name__ == "__main__":
    main()
