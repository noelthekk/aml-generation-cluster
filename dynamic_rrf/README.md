# Dynamic RRF (Cluster)

Self-contained project to generate answers for a fifth, experimental retrieval config -
`hybrid_dynamic_rrf` - across all 50 test-set queries, using Llama-3.1-8B-Instruct (the
same fixed generator identity as the primary four-config comparison in `..`, the parent
`generation_cluster/` folder). Not part of the main `aml-hybrid-rag` project, and not a
replacement for `generation_cluster/` - that folder's four configs
(`dense_only`/`sparse_only`/`dense_sparse`/`hybrid`) already have their 8B answers; this
folder only generates the one new config, so the result is directly comparable without
regenerating anything that already exists. Nested inside `generation_cluster/` (rather
than sitting alongside it) so a single `scp`/clone of the parent folder brings this along
too.

## Status: local validation says don't run this on the cluster as-is

Before spending cluster time, the recommended cheap gate (IR metrics only, free, local,
no LLM generation) was run against all 50 real test
queries, comparing `hybrid_dynamic_rrf_retrieve()` against the existing
`hybrid_retrieve()`:

| Retriever | Top-1 correct | MRR |
|---|---|---|
| Regular hybrid (existing, unweighted RRF) | 18/50 | 0.478 |
| Dynamic RRF, as designed (0.7/0.3, sparse-favors-specific) | 10/50 | 0.364 |
| Dynamic RRF, inverted (0.7/0.3, dense-favors-specific) | 10/50 | 0.375 |
| Dynamic RRF, milder split (0.55/0.45 either direction) | 18/50 | 0.449 |

Dynamic weighting hurts retrieval on this corpus in **either direction**, not just the
one implemented - this isn't a sign bug, both directions were tested. Likely cause: this
corpus is densely templated (many MLR/POCA provisions share near-identical phrasing,
differing only by a number), which is exactly the condition where BM25 gets confused
between neighbouring clauses - the opposite of general QA benchmarks (DROP, PubMedQA,
FinanceBench) the source paper validated on, where lexical specificity reliably signals
"trust BM25 more." The existing plain, unweighted RRF is already well-tuned for this
corpus's dense/sparse balance.

**This is a genuine negative finding, not wasted code** - worth keeping in the write-up:
a literature-motivated dynamic RRF scheme, validated on general QA benchmarks, does not
transfer to a densely-templated regulatory corpus, and degrades retrieval regardless of
weighting direction or magnitude. The code below is fully working and ready to run if
there's a reason to want the generation-stage data anyway (e.g. completeness, or to
check whether a generation-stage effect diverges from the retrieval-stage one despite
the retrieval regression) - just not recommended as the next default step.

## What "dynamic RRF" means here

`../scripts/retriever.py`'s `hybrid_retrieve()` fuses dense, sparse, and graph-expanded
candidates via plain, unweighted Reciprocal Rank Fusion - every list contributes equally
regardless of the query. This folder's `retriever.py` adds
`hybrid_dynamic_rrf_retrieve()`, which instead weights dense vs sparse per query:

- **Specific** queries (naming an explicit regulation/section/paragraph/recommendation
  number - detected via `query_specificity()`, a source-agnostic regex covering all
  five corpus documents' identifier styles) weight toward sparse/BM25 (0.3 dense / 0.7
  sparse), since BM25 already wins on exact identifier matching per this project's own
  baseline ablation.
- **General** queries weight toward dense (0.7 dense / 0.3 sparse).
- Graph's contribution stays fixed at weight 1.0, not varied dynamically - hybrid
  retrieval already improves cross-reference recall/precision without needing a
  weighting change; what this experiment tested is whether reweighting dense/sparse
  specifically improves things further (per the Status section above: it doesn't, on
  this corpus).

Weighting split (0.7/0.3) and the RRF-with-weights mechanism both follow Mala, Gezici &
Giannotti (2025), "Hybrid Retrieval for Hallucination Mitigation in Large Language
Models."

The specificity heuristic was verified live against the real 50-query test set before
use here: flags 24/25 `exact_anchor` and 25/25 `cross_reference` rows as "specific" -
expected, since cross-reference queries in this test set also name an identifier (just
of the clause being cross-referenced from), not evidence the heuristic fails to
discriminate usefully within retrieval behavior itself.

## Setup

Travels automatically with the parent `generation_cluster/` folder - no separate `scp`
needed if that folder is already on the cluster.

1. Copy `.env.example` to `.env` and set `HF_TOKEN` (needed to download the gated Llama
   weights) - or reuse the parent folder's `.env` by symlinking/copying it here.
2. Install `uv` if not already present: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. `uv sync`
4. Build the local vector index (not included in git - rebuildable binary):
   ```bash
   uv run python scripts/build_chroma.py
   ```
   Takes a couple of minutes (CPU embeddings, `all-MiniLM-L6-v2`, 2,568 clauses). Safe to
   re-run - it loads and verifies the existing index instead of rebuilding if
   `data/chroma_db/` already has content.

If the default `torch` install doesn't pick up GPU support for the cluster's CUDA
version, check `nvidia-smi` and install the matching `torch` build from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) before
`uv sync`.

## Run

Only one MIG slice needed (a CUDA process uses a single MIG instance at a time - an
NVIDIA restriction, not a choice); the same partition/GRES as the parent
`generation_cluster/` works.

**Foreground, directly:**
```bash
uv run python generate_answers.py
```

**Backgrounded via `run.sh`** (survives SSH logout):
```bash
bash run.sh start
bash run.sh status         # running? how many of the 50 rows done? last log line
bash run.sh tail           # follow the live log (Ctrl+C stops following, not the run)
bash run.sh stop           # kill the run; completed rows are already saved
```

**Slurm batch job via `submit.sbatch`** (survives SSH logout *and* terminal/VPN loss):
```bash
mkdir -p logs
sbatch submit.sbatch              # prints a job id
squeue -u $USER                   # check status later, no active session needed
tail -f logs/slurm_<jobid>.log    # follow output live
```

Only 50 rows (one config), not 200 - expect roughly a quarter of the parent
`generation_cluster/`'s ~57 min/200-row rate, so ~15 min. `submit.sbatch`'s
`--time=01:00:00` is generous headroom, not a real estimate of need.

## Output

`results/answers_hybrid_dynamic_rrf.jsonl` - 50 rows: `query`, `gold_ids`, `query_type`,
`config` (always `"hybrid_dynamic_rrf"`), `retrieved` (clause IDs actually retrieved),
`answer`, `citations`, and `dynamic_weights` (the `[w_dense, w_sparse]` pair actually
used for that specific query - kept for inspection, since it varies per row and is the
whole point of the experiment). Written incrementally (flushed per row), so a crash
partway through doesn't lose completed rows.

Rows where the model's output wasn't valid JSON get `answer` starting with
`"Generation error: ..."` and empty `citations` - check for those first. Recoverable via
`../../implementation/scripts/recover_answers.py` rather than re-run, since decoding is
deterministic (matches the 8B/70B ablation's own recovery approach).

Copy `results/answers_hybrid_dynamic_rrf.jsonl` back to the main project's
`plan/implementation/results/` when done, alongside the existing `answers.jsonl` /
`answers_recovered.jsonl` - notebook 07's RAGAS cell groups (or the `experiment/` sandbox
notebook) can then score this fifth config against the same reference sets.
