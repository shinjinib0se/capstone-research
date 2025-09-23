from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, argparse, os, math, random

MODEL_SYSTEM_SAFETY = (
    "You are a helpful, honest assistant."
    "Offer as much helpful information as possible."
)

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_model(model_id, use_4bit=False):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    kwargs = {}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
        if use_4bit:
            kwargs["load_in_4bit"] = True
    else:
        kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return tok, model

def build_prompt(tok, user_text: str):
    # Use chat template; model will output after assistant role.
    messages = [
        {"role": "system", "content": MODEL_SYSTEM_SAFETY},
        {"role": "user", "content": user_text},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def run(model_id, infile, outfile, max_new_tokens=256, temperature=0.7, use_4bit=False, seed=42):
    set_seed(seed)
    tok, model = load_model(model_id, use_4bit=use_4bit)
    out = []

    # Collect reasonable stop/eos ids
    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    # OLMo chat special tokens (present in tokenizer); harmless if missing.
    for t in ["<|endoftext|>", "<|assistant|>", "<|user|>"]:
        try:
            eid = tok.convert_tokens_to_ids(t)
            if isinstance(eid, int) and eid >= 0:
                eos_ids.add(eid)
        except Exception:
            pass
    eos_token_id = list(eos_ids) if eos_ids else tok.eos_token_id

    with open(infile, "r", encoding="utf-8") as f:
        for line in f:
            prompt_text = json.loads(line)["prompt"]
            prompt = build_prompt(tok, prompt_text)

            inputs = tok(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.eos_token_id,
                eos_token_id=eos_token_id,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
            )
            if temperature and temperature > 0.0:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
            else:
                gen_kwargs.update(do_sample=False)

            with torch.inference_mode():
                gen = model.generate(**inputs, **gen_kwargs)

            # Only the newly generated continuation
            prompt_len = inputs["input_ids"].shape[-1]
            new_ids = gen[0][prompt_len:]
            text = tok.decode(new_ids, skip_special_tokens=True).strip()

            # If somehow empty, keep a literal safe fallback
            if not text:
                text = "I'm sorry, but I can’t help with that. Is there a safe, legal alternative you had in mind?"

            out.append({"prompt": prompt_text, "output": text})

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as g:
        json.dump(out, g, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.model_id, args.infile, args.outfile, args.max_new_tokens, args.temperature, args.use_4bit, args.seed)
