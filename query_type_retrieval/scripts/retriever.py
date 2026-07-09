"""Hybrid retriever combining dense (ChromaDB), sparse (BM25), and graph (NetworkX).

Public API, Phase 1 (no graph):
  load_dense_sparse(data_dir, chroma_dir)          -> (vectorstore, bm25, clauses)
  dense_sparse_retrieve(query, vectorstore, bm25, clauses, *, k, rrf_k)

Public API, Phase 2 (full hybrid):
  load_retrievers(data_dir, chroma_dir)            -> (vectorstore, bm25, clauses, G)
  hybrid_retrieve(query, vectorstore, bm25, clauses, G, *, k, graph_hops, rrf_k)

Ablation wrappers (same output schema as hybrid_retrieve):
  dense_only_retrieve(vectorstore, clauses, query, k)
  sparse_only_retrieve(bm25, clauses, query, k)

Query-type-conditioned retrieval experiment (this folder only - see
plan/vector_ranking.md Sec 6):
  hybrid_query_type_retrieve(query, query_type, vectorstore, bm25, clauses, G, *, rrf_k)
    Same as hybrid_retrieve, but k/graph_hops/graph_seeds are looked up from
    QUERY_TYPE_PARAMS by query_type instead of using one fixed value for every query.
    exact_anchor keeps the existing defaults (k=10, hops=2, seeds=5) as a control;
    cross_reference gets a larger budget (k=15, hops=3, seeds=8), testing whether the
    fixed budget was under-serving compound/synthesis queries specifically. This is an
    oracle-labeled test (uses the test set's ground-truth query_type, not a predicted
    one) - see plan/vector_ranking.md Sec 6 for why that's a legitimate first check
    before building any classifier or heuristic to predict it at inference time.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from graph_build import load_clauses, load_cross_refs, build_graph, neighbours  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokeniser: regex-based, matches notebook 01 BM25 cell.
# spaCy is not used: 2+ min on 2,568 clauses even with parser/NER disabled.
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are", "be",
    "was", "were", "will", "would", "shall", "should", "may", "might", "must",
    "not", "no", "by", "on", "at", "as", "from", "with", "this", "that", "it",
    "its", "such", "which", "who", "whom", "where", "when", "any", "all", "been",
    "has", "have", "had", "do", "does", "did", "if", "then", "than", "so", "also",
    "their", "they", "them", "these", "those", "but", "can", "into", "out", "up",
}

def tokenise(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z][a-z']*", text.lower())
        if t not in _STOPWORDS and len(t) > 2
    ]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

EMBED_MODEL = "all-MiniLM-L6-v2"


def _build_vectorstore(data_dir: Path, chroma_dir: Optional[Path], embed_model: str) -> Chroma:
    chroma_dir = Path(chroma_dir) if chroma_dir else data_dir / "chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name=embed_model, model_kwargs={"device": "cpu"})
    return Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
        collection_name="aml_clauses",
    )


def load_dense_sparse(
    data_dir: Path,
    chroma_dir: Optional[Path] = None,
    embed_model: str = EMBED_MODEL,
) -> tuple:
    """Load and return (vectorstore, bm25, clauses), no graph. Use for Phase 1 notebooks."""
    data_dir = Path(data_dir)
    logger.info("Loading dense+sparse retrievers from %s", data_dir)
    vectorstore = _build_vectorstore(data_dir, chroma_dir, embed_model)
    clauses = load_clauses(data_dir / "clauses.jsonl")
    logger.info("Loaded %d clauses", len(clauses))
    corpus_tokens = [tokenise(c["text"]) for c in clauses]
    bm25 = BM25Okapi(corpus_tokens)
    logger.info("BM25 index built")
    return vectorstore, bm25, clauses


def load_retrievers(
    data_dir: Path,
    chroma_dir: Optional[Path] = None,
    embed_model: str = EMBED_MODEL,
) -> tuple:
    """Load and return (vectorstore, bm25, clauses, G), full hybrid. Use for Phase 2 notebooks.

    Startup times on CPU (2,568 clauses):
      ChromaDB  : ~1-2s (load from disk)
      BM25      : ~0.4s (rebuild from JSONL)
      NetworkX  : ~100ms (rebuild from cross_refs.jsonl)
    """
    data_dir = Path(data_dir)
    logger.info("Loading all retrievers (dense+sparse+graph) from %s", data_dir)
    vectorstore, bm25, clauses = load_dense_sparse(data_dir, chroma_dir, embed_model)
    cross_refs = load_cross_refs(data_dir / "cross_refs.jsonl")
    G = build_graph(clauses, cross_refs=cross_refs)
    logger.info("NetworkX graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return vectorstore, bm25, clauses, G


# ---------------------------------------------------------------------------
# Individual retrievers
# ---------------------------------------------------------------------------

def dense_retrieve(vectorstore: Chroma, query: str, k: int) -> list[dict]:
    """Return top-k clauses by dense cosine similarity, sorted by score descending."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    hits = [
        {
            "clause_id": doc.metadata["clause_id"],
            "source": doc.metadata["source"],
            "text": doc.page_content,
            "score": float(score),
        }
        for doc, score in results
    ]
    logger.debug("dense_retrieve: top-1=%s (%.4f)", hits[0]["clause_id"] if hits else "none", hits[0]["score"] if hits else 0)
    return hits


def sparse_retrieve(bm25: BM25Okapi, clauses: list[dict], query: str, k: int) -> list[dict]:
    """Return top-k clauses by BM25 score, sorted by score descending."""
    tokens = tokenise(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    hits = [
        {
            "clause_id": clauses[i]["clause_id"],
            "source": clauses[i]["source"],
            "text": clauses[i]["text"],
            "score": float(scores[i]),
        }
        for i in top_indices
    ]
    logger.debug("sparse_retrieve: top-1=%s (%.4f)", hits[0]["clause_id"] if hits else "none", hits[0]["score"] if hits else 0)
    return hits


def graph_expand(G, seed_ids: list[str], hops: int) -> list[str]:
    """Return clause_ids reachable from seeds via CROSS_REFERENCES within `hops` steps.

    Ordered: hop-1 neighbours before hop-2. Seeds excluded from output.
    """
    seen = set(seed_ids)
    ordered: list[str] = []
    frontier = set(seed_ids)

    for _ in range(hops):
        next_frontier: set[str] = set()
        for node in frontier:
            for nid in neighbours(G, node, rel="CROSS_REFERENCES", hops=1):
                if nid not in seen:
                    next_frontier.add(nid)
                    seen.add(nid)
                    ordered.append(nid)
        frontier = next_frontier

    return ordered


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def rrf_fuse(ranked_lists: list[list[dict]], weights: Optional[list[float]] = None, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over multiple ranked result lists.

    Each list must contain dicts with a "clause_id" key, ordered best-first.
    k is the RRF smoothing constant (60 = Cormack et al. 2009 default).
    weights, if given, scales each list's contribution before summing (must be same
    length as ranked_lists). Default (None) is all-1.0 - the original unweighted RRF,
    so every existing call site (dense_sparse_retrieve, hybrid_retrieve) is unaffected.

    Returns [{clause_id, rrf_score}] sorted by rrf_score descending.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for rank, item in enumerate(ranked):
            cid = item["clause_id"]
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)

    return [
        {"clause_id": cid, "rrf_score": score}
        for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


# ---------------------------------------------------------------------------
# Dense + sparse retriever (Phase 1, no graph)
# ---------------------------------------------------------------------------

def dense_sparse_retrieve(
    query: str,
    vectorstore: Chroma,
    bm25: BM25Okapi,
    clauses: list[dict],
    *,
    k: int = 10,
    rrf_k: int = 60,
) -> list[dict]:
    """Dense + sparse fusion via RRF, no graph expansion. Use in Phase 1 notebooks."""
    clause_lookup = {c["clause_id"]: c for c in clauses}
    dense_results = dense_retrieve(vectorstore, query, k=k)
    sparse_results = sparse_retrieve(bm25, clauses, query, k=k)
    fused = rrf_fuse([dense_results, sparse_results], k=rrf_k)
    out = []
    for rank, item in enumerate(fused[:k]):
        cid = item["clause_id"]
        clause = clause_lookup.get(cid, {})
        out.append({
            "clause_id": cid,
            "source": clause.get("source", ""),
            "text": clause.get("text", ""),
            "rrf_score": round(item["rrf_score"], 6),
            "rank": rank + 1,
        })
    logger.info("dense_sparse_retrieve: query=%r  top-1=%s", query[:60], out[0]["clause_id"] if out else "none")
    return out


# ---------------------------------------------------------------------------
# Hybrid retriever (Phase 2, dense + sparse + graph)
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    query: str,
    vectorstore: Chroma,
    bm25: BM25Okapi,
    clauses: list[dict],
    G,
    *,
    k: int = 10,
    graph_hops: int = 2,
    rrf_k: int = 60,
    graph_seeds: int = 5,
) -> list[dict]:
    """Hybrid retrieval: dense + sparse + graph expansion -> RRF -> top-k.

    Returns list[dict] with: clause_id, source, text, rrf_score, rank
    """
    clause_lookup = {c["clause_id"]: c for c in clauses}

    dense_results = dense_retrieve(vectorstore, query, k=k)
    sparse_results = sparse_retrieve(bm25, clauses, query, k=k)

    seeds = list(
        {r["clause_id"] for r in dense_results[:graph_seeds]} |
        {r["clause_id"] for r in sparse_results[:graph_seeds]}
    )
    graph_ids = graph_expand(G, seeds, hops=graph_hops)
    graph_results = [
        {
            "clause_id": cid,
            "source": clause_lookup[cid]["source"],
            "text": clause_lookup[cid]["text"],
        }
        for cid in graph_ids
        if cid in clause_lookup
    ]

    fused = rrf_fuse([dense_results, sparse_results, graph_results], k=rrf_k)

    out = []
    for rank, item in enumerate(fused[:k]):
        cid = item["clause_id"]
        clause = clause_lookup.get(cid, {})
        out.append({
            "clause_id": cid,
            "source": clause.get("source", ""),
            "text": clause.get("text", ""),
            "rrf_score": round(item["rrf_score"], 6),
            "rank": rank + 1,
        })

    logger.info(
        "hybrid_retrieve: query=%r  graph_expanded=%d  top-1=%s",
        query[:60], len(graph_results), out[0]["clause_id"] if out else "none",
    )
    return out


# ---------------------------------------------------------------------------
# Query-type-conditioned retrieval (this folder's experiment - see
# plan/vector_ranking.md Sec 6)
# ---------------------------------------------------------------------------

# exact_anchor keeps hybrid_retrieve()'s existing defaults as a control - only
# cross_reference changes, so any effect observed is attributable to that one
# deliberate change, not confounded by touching both simultaneously.
QUERY_TYPE_PARAMS = {
    "exact_anchor":    {"k": 10, "graph_hops": 2, "graph_seeds": 5},
    "cross_reference": {"k": 15, "graph_hops": 3, "graph_seeds": 8},
}


def hybrid_query_type_retrieve(
    query: str,
    query_type: str,
    vectorstore: Chroma,
    bm25: BM25Okapi,
    clauses: list[dict],
    G,
    *,
    rrf_k: int = 60,
) -> list[dict]:
    """Same as hybrid_retrieve, but k/graph_hops/graph_seeds are looked up from
    QUERY_TYPE_PARAMS by query_type instead of one fixed value for every query.
    Motivation: the RAGAS review found hybrid genuinely improves cross-reference
    retrieval (context_recall 0.53->0.67, context_precision 0.38->0.50 vs dense_only)
    but that gain doesn't survive to answer correctness (+0.02-0.04 only) - this tests
    whether the fixed TOP_K=10/graph_hops=2 budget was under-serving cross-reference
    queries specifically, before concluding the bottleneck is purely generation-side
    (see plan/vector_ranking.md Sec 6 for the three-outcome interpretation).

    query_type is the test set's ground-truth label (oracle test, not a predicted
    value) - a legitimate first check of whether the idea has merit at all, before
    building any classifier/heuristic to predict it on unlabeled queries at inference
    time (plan/vector_ranking.md Sec 6 notes the citation-regex heuristic from the
    dynamic-RRF experiment, ../dynamic_rrf/scripts/retriever.py's query_specificity(),
    would be the natural deployable proxy if this oracle test shows a real effect).

    Returns list[dict] with: clause_id, source, text, rrf_score, rank
    """
    params = QUERY_TYPE_PARAMS[query_type]
    k, graph_hops, graph_seeds = params["k"], params["graph_hops"], params["graph_seeds"]
    clause_lookup = {c["clause_id"]: c for c in clauses}

    dense_results = dense_retrieve(vectorstore, query, k=k)
    sparse_results = sparse_retrieve(bm25, clauses, query, k=k)

    seeds = list(
        {r["clause_id"] for r in dense_results[:graph_seeds]} |
        {r["clause_id"] for r in sparse_results[:graph_seeds]}
    )
    graph_ids = graph_expand(G, seeds, hops=graph_hops)
    graph_results = [
        {
            "clause_id": cid,
            "source": clause_lookup[cid]["source"],
            "text": clause_lookup[cid]["text"],
        }
        for cid in graph_ids
        if cid in clause_lookup
    ]

    fused = rrf_fuse([dense_results, sparse_results, graph_results], k=rrf_k)

    out = []
    for rank, item in enumerate(fused[:k]):
        cid = item["clause_id"]
        clause = clause_lookup.get(cid, {})
        out.append({
            "clause_id": cid,
            "source": clause.get("source", ""),
            "text": clause.get("text", ""),
            "rrf_score": round(item["rrf_score"], 6),
            "rank": rank + 1,
        })

    logger.info(
        "hybrid_query_type_retrieve: query=%r  query_type=%s  k=%d graph_hops=%d  graph_expanded=%d  top-1=%s",
        query[:60], query_type, k, graph_hops, len(graph_results), out[0]["clause_id"] if out else "none",
    )
    return out


# ---------------------------------------------------------------------------
# Ablation wrappers: same output schema as hybrid_retrieve
# ---------------------------------------------------------------------------

def dense_only_retrieve(
    vectorstore: Chroma, clauses: list[dict], query: str, k: int = 10
) -> list[dict]:
    results = dense_retrieve(vectorstore, query, k=k)
    return [
        {
            "clause_id": r["clause_id"],
            "source": r["source"],
            "text": r["text"],
            "rrf_score": round(1.0 / (60 + i + 1), 6),
            "rank": i + 1,
        }
        for i, r in enumerate(results[:k])
    ]


def sparse_only_retrieve(
    bm25: BM25Okapi, clauses: list[dict], query: str, k: int = 10
) -> list[dict]:
    results = sparse_retrieve(bm25, clauses, query, k=k)
    return [
        {
            "clause_id": r["clause_id"],
            "source": r["source"],
            "text": r["text"],
            "rrf_score": round(1.0 / (60 + i + 1), 6),
            "rank": i + 1,
        }
        for i, r in enumerate(results[:k])
    ]
