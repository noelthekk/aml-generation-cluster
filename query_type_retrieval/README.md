# Query-Type-Conditioned Retrieval (Cluster)

Self-contained project to generate answers for a fifth, experimental retrieval config -
`hybrid_query_type` - across all 50 test-set queries, using Llama-3.1-8B-Instruct (the
same fixed generator identity as the primary four-config comparison in `..`, the parent
`generation_cluster/` folder). Not part of the main `aml-hybrid-rag` project, and not a
replacement for `generation_cluster/` - that folder's four configs already have their 8B
answers; this folder only generates the one new config. Nested inside
`generation_cluster/` (rather than sitting alongside it) so a single `scp`/clone of the
parent folder brings this along too.

## Status: local validation says don't run this on the cluster - negative result

Before spending cluster time, the recommended cheap gate (`plan/vector_ranking.md` Sec 6
- IR metrics only, free, local, no LLM generation) was run against the 25
`cross_reference` test queries (the only rows where the two retrievers differ -
`exact_anchor` uses identical fixed parameters in both, confirmed by a 0/25-mismatch
control check). `test_ir_gate.py` isolates which parameter actually causes the effect:

| Variant | Top-1 correct | MRR |
|---|---|---|
| Existing hybrid (k=10, hops=2, seeds=5) | 5/25 | 0.368 |
| k widened only (k=15, hops=2, seeds=5) | 5/25 | 0.363 |
| hops widened only (k=10, hops=3, seeds=8) | 4/25 | 0.338 |
| **Both widened (k=15, hops=3, seeds=8) - as designed** | **3/25** | **0.314** |
| Mild nudge (k=12, hops=2, seeds=6) | 5/25 | 0.377 |

(Numbers wobble by about ±1/25 between separate process runs - ChromaDB's HNSW index
does *approximate* nearest-neighbor search, which has run-to-run variance on near-tied
scores even against a fixed persisted index; confirmed deterministic *within* a single
process via a repeated-call diff. The pattern above replicated across two independent full
runs: "both widened" was the worst variant both times, "mild nudge" the best or
tied-best both times.)

**Widening `graph_hops` is the harmful lever, not `k`.** k-only widening is roughly
neutral to mildly positive; hops-only widening consistently hurts; combining both (the
originally designed `cross_reference` parameter set) is consistently the worst option
tested. Likely cause, and it connects directly to `dynamic_rrf/`'s own negative finding:
this corpus's cross-reference graph is unusually dense (2826 nodes, 5299 edges from just
2,568 clauses) and the source documents are heavily templated - a third graph hop reaches
a much larger, noisier candidate neighborhood without a proportional gain in genuinely
relevant matches, diluting RRF's per-candidate score contributions across more
near-duplicate/tangentially-related clauses.

**This is a genuine negative finding, not wasted code** - worth keeping in the write-up
alongside `dynamic_rrf/`'s: two different literature-motivated attempts to adapt this
project's retrieval fusion further (reweighting sources, widening retrieval budget by
query type) both fail on this specific corpus, for a related underlying reason (dense
templating/cross-referencing diluting ranking signal as the candidate pool widens). That
is itself evidence the existing fixed, unweighted, fixed-budget RRF design is already
close to well-tuned for this corpus, not an oversight. The code below is fully working
and ready to run if there's a reason to want the generation-stage data anyway - just not
recommended as the next default step.

## What "query-type-conditioned retrieval" means here

`../scripts/retriever.py`'s `hybrid_retrieve()` uses the same fixed `k`/`graph_hops`/
`graph_seeds` for every query regardless of type. This folder's `retriever.py` adds
`hybrid_query_type_retrieve()`, which looks up those three parameters from
`QUERY_TYPE_PARAMS` by the query's `query_type` label instead:

- `exact_anchor` keeps the existing defaults (`k=10`, `graph_hops=2`, `graph_seeds=5`) as
  a control.
- `cross_reference` was tested at `k=15`, `graph_hops=3`, `graph_seeds=8` - a larger
  budget, motivated by the RAGAS review finding that hybrid genuinely improves
  cross-reference retrieval (`context_recall` 0.53->0.67, `context_precision` 0.38->0.50
  vs `dense_only`) but that gain barely survives to answer correctness (+0.02-0.04 only)
  - the open question was whether the fixed budget was under-serving these harder
    queries. Per the Status section above: it isn't - more budget makes retrieval worse,
    not better.

This uses the test set's ground-truth `query_type` label directly (an oracle test, not a
predicted value) - a legitimate first check of whether the idea has merit at all before
building any classifier or heuristic to predict query type on unlabeled queries at
inference time. Full design, the three-outcome interpretation this was meant to
distinguish between, and how it compares to `dynamic_rrf/`'s experiment and three
literature papers' hybrid-ranking approaches: see `plan/vector_ranking.md` Sec 6.

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

To reproduce the local IR-metrics gate (no GPU/cluster needed):
```bash
uv run python test_ir_gate.py
```

If the default `torch` install doesn't pick up GPU support for the cluster's CUDA
version, check `nvidia-smi` and install the matching `torch` build from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) before
`uv sync`.

## Run (not recommended given the negative result above, but fully working)

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

`results/answers_hybrid_query_type.jsonl` - 50 rows: `query`, `gold_ids`, `query_type`,
`config` (always `"hybrid_query_type"`), `retrieved` (clause IDs actually retrieved),
`answer`, `citations`, and `retrieval_params` (the `{k, graph_hops, graph_seeds}` actually
used for that row - kept for inspection, since it varies by `query_type`). Written
incrementally (flushed per row), so a crash partway through doesn't lose completed rows.

Rows where the model's output wasn't valid JSON get `answer` starting with
`"Generation error: ..."` and empty `citations` - check for those first. Recoverable via
`../../implementation/scripts/recover_answers.py` rather than re-run, since decoding is
deterministic (matches the 8B/70B ablation's own recovery approach).

Copy `results/answers_hybrid_query_type.jsonl` back to the main project's
`plan/implementation/results/` when done, alongside the existing `answers.jsonl` /
`answers_recovered.jsonl` - notebook 07's RAGAS cell groups (or the `experiment/` sandbox
notebook) can then score this fifth config against the same reference sets.
