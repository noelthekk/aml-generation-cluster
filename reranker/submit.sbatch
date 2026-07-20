#!/usr/bin/env bash
#SBATCH --job-name=aml-reranker
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --output=logs/slurm_%j.log
#SBATCH --time=01:00:00

# One MIG slice only: a CUDA process can use a single MIG instance at a time (NVIDIA
# restriction). 18GB is enough - Llama-3.1-8B in BF16 is ~16GB of weights, matching
# ../README.md's own "18GB+" documented requirement for the primary 8B comparison (the
# sibling dynamic_rrf/query_type_retrieval experiments requested the 71GB slice instead,
# but that was availability/convenience, not a real requirement difference). Only 50
# rows here (one config), not 200 - expect roughly a quarter of
# ../generation_cluster's ~57min/200-row rate, plus the CPU-side cross-encoder rerank
# step (fast - MiniLM, k_wide=30 candidates per query). 1 hour is generous headroom.
# build_chroma.py is idempotent - verifies the existing index instead of rebuilding.

uv run python scripts/build_chroma.py
uv run python generate_answers.py
