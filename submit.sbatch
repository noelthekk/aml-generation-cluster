#!/usr/bin/env bash
#SBATCH --job-name=aml-answers
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --output=logs/slurm_%j.log
#SBATCH --time=08:00:00

# One MIG slice only: a CUDA process can use a single MIG instance at a time
# (NVIDIA restriction). 71GB comfortably fits either the 8B (BF16, ~16GB) or the 70B
# (NF4 4-bit, ~35-40GB) - adjust --gres if targeting different cluster hardware.
# build_chroma.py is idempotent - verifies the existing index instead of rebuilding.
#
# Edit --model-size on the last line below before submitting, depending on which run
# you want (8b or 70b).

uv run python scripts/build_chroma.py
uv run python generate_answers.py --model-size 8b
