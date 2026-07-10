# Answer Generation (Cluster)

Self-contained project to generate answers for all 50 test-set queries across all four
retrieval configs (`dense_only`, `sparse_only`, `dense_sparse`, `hybrid`) — 200 rows per
run — using Llama-3.1 at a model size you choose. Not part of the main `aml-hybrid-rag`
project — has its own dependencies and its own copies of `scripts/retriever.py`,
`scripts/graph_build.py`, `scripts/build_chroma.py`, and the data files they need
(`clauses.jsonl`, `cross_refs.jsonl`, `test_set.jsonl`). `data/chroma_db/` is *not*
included (gitignored — a rebuildable binary index, regenerated locally in setup step 5).

## Two things this folder does

| Run | `--model-size` | Precision | Partition | Purpose |
|---|---|---|---|---|
| Primary comparison | `8b` (default) | BF16 | 18GB+ | Generator held fixed at Llama-3.1-8B across all four retrieval configs — the actual four-config comparison notebook 07 evaluates. Fixed regardless of hardware, so retrieval-quality differences aren't confounded with generator-quality differences. |
| Generator-scale ablation | `70b` | NF4 4-bit | 72GB | Same 200 (config, query) pairs re-generated with Llama-3.1-70B, to check whether the retrieval-config ranking holds with a stronger generator. Added 2026-07-07. |

Both runs share every other variable — same retrievers, same `TOP_K`/`RRF_K`/`GRAPH_HOPS`,
same prompt, same greedy decoding — only generator size/precision changes. Precision
caveat for the write-up: 70B runs NF4 4-bit (BF16 at 70B is ~140GB of weights — does not
fit a 72GB partition), while the primary 8B run is BF16. So the 8B/70B comparison is
BF16-vs-NF4 as well as a scale difference — quantization cost on a 70B is small but
nonzero, and the report should say so rather than presenting a pure scale comparison.

## Setup

1. Copy this whole `generation_cluster/` folder to the cluster:
   ```bash
   scp -r "C:/Users/tsono/Documents/uoe/disertation/plan/implementation/generation_cluster" username@remote_host:/path/to/remote/directory/
   ```
2. Copy `.env.example` to `.env` and set `HF_TOKEN` (needed to download the gated Llama
   weights).
3. Install `uv` if not already present: `curl -LsSf https://astral.sh/uv/install.sh | sh`
4. `uv sync`
5. Build the local vector index (not included in git, see above):
   ```bash
   uv run python scripts/build_chroma.py
   ```
   Takes a couple of minutes (CPU embeddings, `all-MiniLM-L6-v2`, 2,568 clauses). Safe to
   re-run — it loads and verifies the existing index instead of rebuilding if
   `data/chroma_db/` already has content.

If the default `torch` install doesn't pick up GPU support for your cluster's CUDA
version, check `nvidia-smi` and install the matching `torch` build from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) before
`uv sync`.

## Run

Three ways to run it, in increasing order of how much unattended time you need:

**Foreground, directly:**
```bash
uv run python generate_answers.py                  # defaults to --model-size 8b
uv run python generate_answers.py --model-size 70b
```

**Backgrounded via `run.sh`** (survives SSH logout — `nohup` + its own process group):
```bash
bash run.sh start          # defaults to 8b
bash run.sh start 70b
bash run.sh status         # running? how many of the 200 rows done? last log line
bash run.sh tail           # follow the live log (Ctrl+C stops following, not the run)
bash run.sh stop           # kill the run; completed rows are already saved
```

**Slurm batch job via `submit.sbatch`** (survives SSH logout *and* terminal/VPN loss —
the job runs under the scheduler, independent of any session): edit the
`--model-size` on `submit.sbatch`'s last line for the run you want, adjust the
`#SBATCH` resource lines to match your cluster's partition/GRES names, then:
```bash
mkdir -p logs
sbatch submit.sbatch              # prints a job id, e.g. "Submitted batch job 12345"
```
Managing the job once it's submitted:
```bash
squeue -u $USER                   # check status later, no active session needed
squeue -j 12345                   # status of just this job (PD=pending, R=running, CG=completing)
scontrol show job 12345           # full job detail: assigned node, time limit, reason if pending
tail -f logs/slurm_12345.log      # follow output live (stdout+stderr, per --output above)
scancel 12345                     # cancel the job (queued or running)
sacct -j 12345 --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode  # post-run summary, incl. exit code
```
One MIG slice only — a CUDA process can use a single MIG instance at a time (an NVIDIA
restriction, not a choice), and one 71GB-class slice comfortably fits either model size.

Progress shows as a tqdm bar (current config + query, per-query rate, ETA) plus
timestamped log lines; generation errors are logged as warnings as they happen. Timing:
the 8B/BF16 run took ~57 min for all 200 rows on one `h200_3g.71gb` slice; the 70B/NF4
run took the same, ~57 min, on the same hardware.

## Output

`results/answers_<model-size>.jsonl` — 200 rows (4 configs x 50 queries): `query`,
`gold_ids`, `query_type`, `config`, `retrieved` (clause IDs actually retrieved for that
query+config), `answer`, `citations`. Written incrementally (one line per row, flushed
immediately), so a crash partway through doesn't lose completed rows. The model-size
suffix keeps an 8B and a 70B run from overwriting each other.

Rows where the model's output wasn't valid JSON get `answer` starting with
`"Generation error: ..."` and empty `citations` — check for those first (the 8B run had
3 such rows out of 200; the 70B run had 145 — the bigger model was far more likely to
wrap its JSON in prose or skip it, a real finding in itself; both were recovered via
`scripts/recover_answers.py` in the main project rather than re-run, since decoding is
deterministic).

Copy `results/answers_<model-size>.jsonl` back to the main project's
`plan/implementation/results/` when done — that's the exact location notebook 07's
RAGAS cell groups expect, keeping the size suffix so multiple runs coexist.

## On reference drafting

This folder does not draft reference answers. `data/test_set.jsonl` already carries
`reference_14b` and `reference_72b`, drafted via the separate `cluster/` folder
(Qwen2.5-14B/72B) and human-verified row by row — references are drafted from the gold
clause text only, so they don't depend on which generator produced the answers here, and
apply unchanged to both the 8B and 70B runs. See `cluster/README.md` if a fresh draft is
ever genuinely needed.
