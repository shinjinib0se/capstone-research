#!/bin/bash
#SBATCH --job-name=dpo-debug
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dpo-debug-%j.out
#SBATCH --error=logs/dpo-debug-%j.err

# Load required modules
module load python/3.11.4
module load bzip2

# Activate the correct virtual environment
cd /scratch/dpj7jx/capstone-research
source .venv/bin/activate

# (Optional safeguard) — install dependencies if missing
pip install --upgrade pip
pip install click wandb

#  Run training
python scripts/train_dpo.py \
  --train data/rlhf_pairs_train.jsonl \
  --eval data/rlhf_pairs_eval.jsonl \
  --base /scratch/dpj7jx/huggingface/models--allenai--OLMo-2-0425-1B-SFT/snapshots/0d85a3d037876ce6ac7d4311d994400fc66ac27f \
  --out out/debug-run \
  --epochs 5 \
  --bsz 1 \
  --limit_train 50 \
  --limit_eval 10
