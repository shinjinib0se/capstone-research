# capstone-research

## Quickstart (WSL)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python sanity.py
python scripts/generate_olmo.py --prompt "Explain supervised fine-tuning in one sentence."


python scripts/generate.py   --model_id allenai/OLMo-2-0425-1B-SFT   --infile data/benign.jsonl   --outfile out/benign.olmo2_1b_sft.greedy.t128.jsonl   --max_new_tokens 128   --temperature 0