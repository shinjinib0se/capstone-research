#!/bin/bash
#SBATCH --job-name=eval-behav-v2
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/eval_behavior_%j.out
#SBATCH --error=logs/eval_behavior_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dpj7jx@virginia.edu

# ----------------------------
# Setup
# ----------------------------
module purge
module load python/3.11.4

# Navigate to your project directory
cd /scratch/dpj7jx/capstone-research

# Activate virtual environment
source .venv/bin/activate

# ----------------------------
# Run evaluation
# ----------------------------
python scripts/eval_behavior_v2.py \
  --base /scratch/dpj7jx/huggingface/models--allenai--OLMo-2-0425-1B-SFT/snapshots/0d85a3d037876ce6ac7d4311d994400fc66ac27f \
  --tuned runs/dpo_olmo2_1b_full/checkpoint-2000 \
  --pairs data/rlhf_pairs_eval_labeled.jsonl \
  --outdir runs/dpo_behavior_eval_v2

# ----------------------------
# Done
# ----------------------------
echo "Finished eval_behavior_v2 run at $(date)"
