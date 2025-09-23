# sanity.py
# Quick smoke test for OLMo-2 SFT: loads the model, runs one prompt, prints output + env info.

import sys
import platform
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "allenai/OLMo-2-0425-1B-SFT"  # small & fast; you can swap to 7B later

# --- runtime / device setup ---
cuda = torch.cuda.is_available()
dtype = torch.float16 if cuda else torch.float32

use_4bit = False
try:
    import bitsandbytes as _  # noqa: F401
    use_4bit = True
except Exception:
    use_4bit = False  # fall back to full precision load

# --- load tokenizer & model ---
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

load_kwargs = dict(device_map="auto", torch_dtype=dtype)
if cuda and use_4bit:
    load_kwargs["load_in_4bit"] = True  # comment this out if you hit bnb/CUDA issues

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
model.eval()

# --- build a minimal chat prompt ---
messages = [{"role": "user", "content": "In one sentence, what is OLMo 2?"}]
try:
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
except Exception:
    # Fallback if the tokenizer doesn't expose a chat template for some reason
    prompt = messages[0]["content"] + "\n"

inputs = tok(prompt, return_tensors="pt")
if cuda:
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

# --- generate ---
with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,   # greedy for deterministic sanity
        pad_token_id=tok.eos_token_id,
    )

text = tok.decode(out[0], skip_special_tokens=True)

# --- report ---
print("=== Environment ===")
print("python:", sys.version.split()[0])
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("cuda_available:", cuda)
print("device:", torch.cuda.get_device_name(0) if cuda else "CPU")
print("dtype:", str(dtype))
print("load_in_4bit:", use_4bit and cuda)
print("model_id:", MODEL_ID)
print("\n=== Prompt ===\n", prompt)
print("\n=== Output ===\n", text)
