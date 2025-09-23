# scripts/generate_olmo.py
import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

MODEL_ID = "allenai/OLMo-2-0425-1B-SFT"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    qconf = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=qconf,
    ).eval()

    try:
        prompt = tok.apply_chat_template(
            [{"role":"user","content":args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = args.prompt

    inputs = tok(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k:v.to(model.device) for k,v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
