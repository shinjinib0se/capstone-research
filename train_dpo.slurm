#!/bin/bash
#SBATCH --job-name=dpo_olmo2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dpo_%j.out
#SBATCH --error=logs/dpo_%j.err

module load miniforge/24.3.0-py3.11
conda activate cast_dpo

cd ~/capstone-research

python scripts/train_dpo.py \
  --train data/rlhf_pairs_train.jsonl \
  --eval data/rlhf_pairs_eval.jsonl \
  --base allenai/OLMo-2-1B \
  --out runs/dpo_olmo2_1b_beta05 \
  --epochs 2 \
  --bsz 2 \
  --lr 1e-5 \
  --limit_train 2000 \
  --limit_eval 200
