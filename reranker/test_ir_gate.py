"""Cheap local IR-metrics-only gate for the cross-encoder reranking experiment
(plan/improvement_plan.md P3) - run BEFORE spending any cluster time, same pattern as
../dynamic_rrf and ../query_type_retrieval's own gates.

Compares hybrid_rerank_retrieve() (existing hybrid_retrieve at a wider k_wide budget,
then cross-encoder reranked back to k) against the existing hybrid_retrieve(k=10),
over all 50 test queries, split by query_type.

Usage:
    uv run python test_ir_gate.py
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from retriever import load_retrievers, load_reranker, hybrid_retrieve, hybrid_rerank_retrieve  # noqa: E402

logging.basicConfig(level=logging.WARNING)  # quiet - only want the summary

DATA_DIR = Path("data")
CHROMA_DIR = DATA_DIR / "chroma_db"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def mrr_and_top1(rows_scored: list[tuple[list[str], list[str]]]) -> tuple[int, float]:
    """rows_scored: list of (retrieved_clause_ids, gold_ids). Returns (top1_correct, mrr)."""
    top1 = 0
    rr_sum = 0.0
    for retrieved, gold in rows_scored:
        gold_set = set(gold)
        if retrieved and retrieved[0] in gold_set:
            top1 += 1
        rr = 0.0
        for rank, cid in enumerate(retrieved, start=1):
            if cid in gold_set:
                rr = 1.0 / rank
                break
        rr_sum += rr
    return top1, rr_sum / len(rows_scored)


def score_baseline(rows: list[dict], vectorstore, bm25, clauses, G) -> tuple[int, float]:
    scored = []
    for row in rows:
        results = hybrid_retrieve(row["query"], vectorstore, bm25, clauses, G)
        scored.append(([r["clause_id"] for r in results], row["gold_ids"]))
    return mrr_and_top1(scored)


def score_reranked(rows: list[dict], vectorstore, bm25, clauses, G, reranker, k_wide: int) -> tuple[int, float]:
    scored = []
    for row in rows:
        results = hybrid_rerank_retrieve(
            row["query"], vectorstore, bm25, clauses, G, reranker, k_wide=k_wide,
        )
        scored.append(([r["clause_id"] for r in results], row["gold_ids"]))
    return mrr_and_top1(scored)


def main() -> None:
    vectorstore, bm25, clauses, G = load_retrievers(DATA_DIR, CHROMA_DIR)
    reranker = load_reranker()
    test_set = load_jsonl(DATA_DIR / "test_set.jsonl")
    print(f"Loaded {len(clauses)} clauses, {len(test_set)} test queries")

    for query_type in ["exact_anchor", "cross_reference", None]:
        rows = test_set if query_type is None else [r for r in test_set if r["query_type"] == query_type]
        label = query_type or "all"
        n = len(rows)
        print(f"\n=== {label} (n={n}) ===")
        print(f"{'Variant':<45} {'Top-1 correct':<15} {'MRR':<8}")

        base_top1, base_mrr = score_baseline(rows, vectorstore, bm25, clauses, G)
        print(f"{'Existing hybrid_retrieve (k=10)':<45} {f'{base_top1}/{n}':<15} {base_mrr:.3f}")

        for k_wide in [15, 20, 30]:
            top1, mrr = score_reranked(rows, vectorstore, bm25, clauses, G, reranker, k_wide)
            print(f"{f'Reranked (k_wide={k_wide} -> k=10)':<45} {f'{top1}/{n}':<15} {mrr:.3f}")


if __name__ == "__main__":
    main()
