# Cross-Encoder Reranking (Cluster)

Self-contained project to generate answers for a fifth, experimental retrieval config -
`hybrid_rerank` - across all 50 test-set queries, using Llama-3.1-8B-Instruct (the
same fixed generator identity as the primary four-config comparison in `..`, the parent
`generation_cluster/` folder). Not part of the main `aml-hybrid-rag` project, and not a
replacement for `generation_cluster/` - that folder's four configs already have their 8B
answers; this folder only generates the one new config. Nested inside
`generation_cluster/` (rather than sitting alongside it) so a single `scp`/clone of the
parent folder brings this along too.

## Status: local validation says this is worth running on the cluster - positive result

Two prior retrieval-ranking experiments in sibling folders (`../dynamic_rrf/`,
`../query_type_retrieval/`) both found that widening or reweighting the RRF candidate
pool by *rank position alone* hurts on this corpus - RRF never reads clause content,
only where each independent retriever ranked it, and this corpus has many
near-duplicate/templated clauses RRF can't disambiguate by rank alone. A cross-encoder
reranker reads the actual clause text jointly with the query - a genuinely different
mechanism, not another variation on reweighting/widening.

Before spending cluster time, the same cheap IR-metrics-only gate used by both prior
experiments was run against all 50 test queries, split by `query_type`:

| | Existing hybrid (k=10) | Reranked (k_wide=15) | Reranked (k_wide=20) | Reranked (k_wide=30) |
|---|---|---|---|---|
| `exact_anchor` (n=25) | 14/25, MRR 0.606 | 14/25, MRR 0.636 | 15/25, MRR 0.693 | 15/25, MRR 0.693 |
| `cross_reference` (n=25) | 5/25, MRR 0.372 | 12/25, MRR 0.538 | 12/25, MRR 0.571 | **13/25, MRR 0.627** |
| all 50 | 19/50, MRR 0.489 | 26/50, MRR 0.587 | 27/50, MRR 0.632 | **28/50, MRR 0.660** |

**This clears the gate decisively, unlike either prior experiment.** The gain is
largest exactly where it matters most: `cross_reference` queries (the persistently hard
category throughout this project) more than double their top-1 accuracy and gain +0.255
MRR. The improvement scales with `k_wide` (wider candidate pool before reranking is
consistently better across the range tested) - `k_wide=30` is what `generate_answers.py`
uses. `test_ir_gate.py` reproduces this table locally, no GPU/cluster needed.

## What "cross-encoder reranking" means here

`../scripts/retriever.py`'s `hybrid_retrieve()` is unchanged and still called directly -
this folder's `retriever.py` adds `hybrid_rerank_retrieve()`, which:

1. Calls the existing `hybrid_retrieve()` at a wider candidate budget (`k_wide=30`
   instead of the usual `k=10`) to get a bigger pool of RRF-fused candidates.
2. Scores each candidate's actual clause text jointly with the query using
   `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB, CPU-feasible, already a transitive
   dependency via `sentence-transformers`).
3. Returns the top-10 by that cross-encoder score, not by the original RRF rank.

Unlike the two prior experiments (which only ever changed *how many* candidates or *how
RRF weights rank position*), this changes *what decides the final ordering* - real
text-level relevance instead of independent retrievers' rank alone.

## Setup

Travels automatically with the parent `generation_cluster/` folder - no separate `scp`
needed if that folder is already on the cluster and you re-`scp`/re-clone it wholesale.
If the parent is already on the cluster from an earlier run and you only need to add
this new subfolder, copy just this folder over:

```bash
scp -r plan/generation_cluster/reranker <cluster>:<path>/generation_cluster/
```

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
The cross-encoder reranking step runs on CPU and is genuinely compute-heavy (multiple
minutes for the full 50-query x 3-`k_wide`-variant sweep across all three `query_type`
groupings) - this is expected, not a hang.

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
`generation_cluster/`'s ~57 min/200-row rate, plus the CPU-side reranking step (fast -
MiniLM, 30 candidates per query). `submit.sbatch`'s `--time=01:00:00` is generous
headroom, not a real estimate of need.

## Output

`results/answers_hybrid_rerank.jsonl` - 50 rows: `query`, `gold_ids`, `query_type`,
`config` (always `"hybrid_rerank"`), `retrieved` (clause IDs, post-rerank order),
`answer`, `citations`. Written incrementally (flushed per row), so a crash partway
through doesn't lose completed rows.

Rows where the model's output wasn't valid JSON get `answer` starting with
`"Generation error: ..."` and empty `citations` - check for those first. Recoverable via
`../../implementation/scripts/recover_answers.py` rather than re-run, since decoding is
deterministic (matches the 8B/70B ablation's own recovery approach).

Copy `results/answers_hybrid_rerank.jsonl` back to the main project's
`plan/implementation/results/` when done, alongside the existing `answers.jsonl` /
`answers_recovered.jsonl` - notebook 07's RAGAS cell groups (or the `experiment/` sandbox
notebook) can then score this fifth config against the same reference sets:

```bash
scp <cluster>:<path>/generation_cluster/reranker/results/answers_hybrid_rerank.jsonl \
    plan/implementation/results/
```

Retrieval quality improving (this folder's Status table above) is not the same claim as
answer correctness improving - the primary hybrid config already showed a case where a
real retrieval gain on `cross_reference` queries (`context_recall` 0.53->0.67) shrank to
almost nothing by the time an answer was generated (+0.02-0.04 correctness). RAGAS
scoring after this run is what actually settles whether the reranking gain survives to
correctness, not the IR-metrics gate alone.
