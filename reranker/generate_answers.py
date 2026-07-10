"""Generate answers for the cross-encoder reranking config, over the RAGAS test set.

Run this on the cluster. This is a companion to ../generate_answers.py (the parent
generation_cluster/ folder), not a replacement: that script generates the primary
four-config comparison (dense_only, sparse_only, dense_sparse, hybrid); this one
generates a fifth, experimental config - hybrid_rerank - which retrieves a wider
candidate set (k_wide=30) via the existing, unchanged hybrid_retrieve(), then
re-scores those candidates' actual clause text against the query with a cross-encoder
and takes the reranked top-10 (see scripts/retriever.py's hybrid_rerank_retrieve and
plan/improvement_plan.md P3 for the full design and motivation).

A local IR-metrics-only gate (no LLM cost) found this clears the bar decisively,
unlike the two prior retrieval experiments (dynamic RRF, query-type-conditioned
budget), both of which were correctly killed at this same gate stage:
  - exact_anchor:     14/25 -> 15/25 top-1, MRR 0.606 -> 0.693
  - cross_reference:   5/25 -> 13/25 top-1, MRR 0.372 -> 0.627  (the hard category)
  - all 50 queries:   19/50 -> 28/50 top-1, MRR 0.489 -> 0.660
See README.md's Status section for the full k_wide sweep (15/20/30 all improve over
baseline; 30 is best) and test_ir_gate.py for the gate itself.

Generator is fixed at Llama-3.1-8B-Instruct, BF16 - the same identity as the primary
comparison (never the 70B ablation) - so this new config's answers are directly
comparable to the existing dense_only/sparse_only/dense_sparse/hybrid answers already
in plan/implementation/results/answers.jsonl.

Self-contained: no dependency on the rest of the aml-hybrid-rag project. Includes its
own copies of scripts/retriever.py, scripts/graph_build.py, and the data files they
need (clauses.jsonl, cross_refs.jsonl, test_set.jsonl); data/chroma_db/ is rebuilt
locally with scripts/build_chroma.py.

Before running:
    1. Copy .env.example to .env and set HF_TOKEN.
    2. uv sync
    3. uv run python scripts/build_chroma.py   (once, rebuilds the vector index)

Usage:
    uv run python generate_answers.py

Or via run.sh (backgrounded, survives SSH logout - see README.md):
    bash run.sh start
    bash run.sh status
    bash run.sh tail
    bash run.sh stop

Output: results/answers_hybrid_rerank.jsonl - one row per query, 50 rows total:
query, gold_ids, query_type, config, retrieved (clause_id list, post-rerank order),
answer, citations. Rows where the model's JSON output didn't parse get an `answer`
field starting with "Generation error:" and an empty `citations` list - check for
those first (matches the parent generation_cluster/'s convention; recoverable via
../../implementation/scripts/recover_answers.py since decoding is deterministic).
"""
import json
import logging
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline as hf_pipeline_fn
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from retriever import load_retrievers, load_reranker, hybrid_rerank_retrieve  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("generate_answers_rerank")

DATA_DIR = Path("data")
CHROMA_DIR = DATA_DIR / "chroma_db"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
TEST_SET_PATH = DATA_DIR / "test_set.jsonl"
OUTPUT_PATH = RESULTS_DIR / "answers_hybrid_rerank.jsonl"

MODEL_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"  # fixed - same identity as the
                                                       # primary comparison, so this
                                                       # config's answers are directly
                                                       # comparable to the existing ones
MAX_NEW_TOKENS = 1024
RRF_K = 60
K_WIDE = 30  # best variant from the local IR gate - see README.md's Status section

CONFIG_NAME = "hybrid_rerank"

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


def main() -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("No CUDA device found - this script requires a GPU.")

    logger.info("Config: %s", CONFIG_NAME)
    logger.info("Output: %s", OUTPUT_PATH)

    logger.info("Loading retrievers from %s", DATA_DIR)
    t0 = time.time()
    vectorstore, bm25, clauses, G = load_retrievers(DATA_DIR, CHROMA_DIR)
    reranker = load_reranker()
    logger.info("Retrievers + reranker loaded: %d clauses  (%.1fs)", len(clauses), time.time() - t0)

    test_set = load_jsonl(TEST_SET_PATH)
    logger.info("Loaded %d test queries", len(test_set))

    logger.info("Loading %s (BF16)...", MODEL_ID)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
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
    logger.info("Model loaded (%.1fs)  GPU: %s", time.time() - t0, torch.cuda.get_device_name(0))

    n_errors = 0
    n_rows = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        progress = tqdm(test_set, unit="query", dynamic_ncols=True)
        for item in progress:
            query = item["query"]
            query_type = item.get("query_type", "unknown")
            progress.set_description(f"{query[:50]!r}")

            retrieved = hybrid_rerank_retrieve(
                query, vectorstore, bm25, clauses, G, reranker, k_wide=K_WIDE, rrf_k=RRF_K,
            )
            context = format_context(retrieved)
            try:
                parsed = chain.invoke({"query": query, "context": context})
                answer = parsed.get("answer", "")
                citations = parsed.get("citations", [])
            except Exception as exc:
                answer = f"Generation error: {exc}"
                citations = []
                n_errors += 1
                logger.warning("[%d/%d] GENERATION ERROR  %r: %s",
                               n_rows + 1, len(test_set), query[:50], exc)

            row = {
                "query": query,
                "gold_ids": item["gold_ids"],
                "query_type": query_type,
                "config": CONFIG_NAME,
                "retrieved": [r["clause_id"] for r in retrieved],
                "answer": answer,
                "citations": citations,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_rows += 1

    logger.info(
        "Done: %d rows written to %s  (%d generation errors)",
        n_rows, OUTPUT_PATH, n_errors,
    )
    if n_errors:
        logger.warning("Check the %d 'Generation error:' rows first.", n_errors)


if __name__ == "__main__":
    main()
