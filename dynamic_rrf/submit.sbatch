#!/usr/bin/env bash
#SBATCH --job-name=aml-dynamic-rrf
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --output=logs/slurm_%j.log
#SBATCH --time=01:00:00

# One MIG slice only: a CUDA process can use a single MIG instance at a time (NVIDIA
# restriction). Only 50 rows here (one config), not 200 - expect roughly a quarter of
# ../generation_cluster's ~57min/200-row rate, so 1 hour is generous headroom.
# build_chroma.py is idempotent - verifies the existing index instead of rebuilding.

uv run python scripts/build_chroma.py
uv run python generate_answers.py
