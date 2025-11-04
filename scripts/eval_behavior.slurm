#!/bin/bash
#SBATCH --job-name=eval-behav
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/eval_behavior_%j.out
#SBATCH --error=logs/eval_behavior_%j.err

module load python/3.11.4
source /scratch/dpj7jx/capstone-research/.venv/bin/activate

python scripts/eval_behavior.py \
  --base /scratch/dpj7jx/huggingface/models--allenai--OLMo-2-0425-1B-SFT/snapshots/0d85a3d037876ce6ac7d4311d994400fc66ac27f \
  --tuned runs/dpo_olmo2_1b_full/checkpoint-2000 \
  --pairs data/rlhf_pairs_eval.jsonl \
  --outdir runs/dpo_behavior_eval
