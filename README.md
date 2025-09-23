# capstone-research
## Quickstart (WSL)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python sanity.py
python scripts/generate_olmo.py --prompt "Explain supervised fine-tuning in one sentence."
