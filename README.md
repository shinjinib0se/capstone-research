# capstone-research

## Quickstart (WSL)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python sanity.py
python scripts/generate_olmo.py --prompt "Explain supervised fine-tuning in one sentence."


python scripts/generate.py   --model_id allenai/OLMo-2-0425-1B-SFT   --infile data/benign.jsonl   --outfile out/benign.olmo2_1b_sft.greedy.t128.jsonl   --max_new_tokens 128   --temperature 0

# Run DPO locally (relatively GPU-friendly)
# this trains OLMo2-1B with reference-free DPO using LoRA and QLoRA, shorter contect, and gradient checkpointing. this is a local run to verify the popeline end to end.
python scripts/train_dpo_min.py   --model models/olmo2-1b --ref_model models/olmo2-1b   --train data/rlhf_pairs_train.jsonl   --out runs/dpo_olmo2_1b   --epochs 1 --bsz 1 --gas 16 --lr 5e-6 --beta 0.1   --fp16 --lora --q4 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05   --max_len 1024 --max_prompt_len 384   --reference_free --grad_ckpt --no_flash