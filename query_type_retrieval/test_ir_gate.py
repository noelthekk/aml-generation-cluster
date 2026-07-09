"""Cheap local IR-metrics-only gate for the query-type-conditioned retrieval experiment
(plan/vector_ranking.md Sec 6) - run BEFORE spending any cluster time, same pattern as
../dynamic_rrf's own gate.

Compares hybrid_query_type_retrieve() (k/graph_hops/graph_seeds conditioned on
query_type) against the existing hybrid_retrieve() (fixed k=10/hops=2/seeds=5 for
every query), restricted to the 25 cross_reference queries - the only rows where the
two retrievers actually differ, since exact_anchor is kept at the same fixed defaults
in both.

Usage:
    uv run python test_ir_gate.py
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from retriever import load_retrievers, hybrid_retrieve, hybrid_query_type_retrieve  # noqa: E402

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


def score_variant(rows: list[dict], vectorstore, bm25, clauses, G, k: int, graph_hops: int, graph_seeds: int):
    scored = []
    for row in rows:
        results = hybrid_retrieve(
            row["query"], vectorstore, bm25, clauses, G,
            k=k, graph_hops=graph_hops, graph_seeds=graph_seeds,
        )
        scored.append(([r["clause_id"] for r in results], row["gold_ids"]))
    return mrr_and_top1(scored)


def main() -> None:
    vectorstore, bm25, clauses, G = load_retrievers(DATA_DIR, CHROMA_DIR)
    test_set = load_jsonl(DATA_DIR / "test_set.jsonl")
    cross_ref_rows = [r for r in test_set if r["query_type"] == "cross_reference"]
    print(f"Loaded {len(clauses)} clauses, {len(test_set)} test queries "
          f"({len(cross_ref_rows)} cross_reference)")

    n = len(cross_ref_rows)
    print(f"\n=== cross_reference queries only (n={n}) - isolating which parameter causes it ===")
    print(f"{'Variant':<50} {'Top-1 correct':<15} {'MRR':<8}")

    variants = [
        ("Existing hybrid (k=10, hops=2, seeds=5)", 10, 2, 5),
        ("k widened only (k=15, hops=2, seeds=5)", 15, 2, 5),
        ("hops widened only (k=10, hops=3, seeds=8)", 10, 3, 8),
        ("both widened (k=15, hops=3, seeds=8) - as designed", 15, 3, 8),
        ("mild nudge (k=12, hops=2, seeds=6)", 12, 2, 6),
    ]
    for label, k, hops, seeds in variants:
        top1, mrr = score_variant(cross_ref_rows, vectorstore, bm25, clauses, G, k, hops, seeds)
        print(f"{label:<50} {f'{top1}/{n}':<15} {mrr:.3f}")

    # Sanity check: exact_anchor rows must be byte-identical between the two retrievers
    # (same fixed k=10/hops=2/seeds=5 in both) - confirms the "control" half of the
    # design is actually unchanged, not just claimed to be.
    exact_anchor_rows = [r for r in test_set if r["query_type"] == "exact_anchor"]
    mismatches = 0
    for row in exact_anchor_rows:
        query = row["query"]
        existing = [r["clause_id"] for r in hybrid_retrieve(query, vectorstore, bm25, clauses, G)]
        adaptive = [r["clause_id"] for r in hybrid_query_type_retrieve(
            query, "exact_anchor", vectorstore, bm25, clauses, G,
        )]
        if existing != adaptive:
            mismatches += 1
    print(f"\nexact_anchor control check: {mismatches}/{len(exact_anchor_rows)} rows differ "
          f"(expect 0 - same fixed params in both)")


if __name__ == "__main__":
    main()
