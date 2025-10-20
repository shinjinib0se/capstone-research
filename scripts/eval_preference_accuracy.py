#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate preference accuracy for a DPO/SFT policy by scoring chosen vs rejected responses.

Input JSONL format (one per line):
{"prompt": "...", "chosen": "...", "rejected": "..."}

Usage examples:
  python scripts/eval_preference_accuracy.py \
    --model runs/dpo_olmo2_1b/checkpoint-1188 \
    --base  allenai/OLMo-2-1B \
    --pairs data/rlhf_pairs_eval.jsonl

  # If you have a fully merged (non-PEFT) model:
  python scripts/eval_preference_accuracy.py \
    --model your-merged-model-dir \
    --pairs data/rlhf_pairs_eval.jsonl
"""

import argparse, json, sys, os, math
from typing import List, Dict, Tuple
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import log_softmax

# Optional: PEFT (only needed if --base is provided and a PEFT adapter is used)
def _maybe_load_peft(model, adapter_dir: str):
    try:
        from peft import PeftModel
    except Exception as e:
        print("PEFT not installed; assuming non-PEFT model. If you intended to use LoRA adapters, install peft.",
              file=sys.stderr)
        return model
    return PeftModel.from_pretrained(model, adapter_dir)


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@torch.no_grad()
def score_logprob_sum(
    model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    max_length: int = 4096,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> List[float]:
    """
    Returns sum log-probabilities of `responses` conditioned on `prompts`.
    Only response tokens are scored; prompt tokens are masked out.
    """
    assert len(prompts) == len(responses)
    # Build concatenated inputs (prompt + response)
    # Left padding keeps prompt alignment simple when packing batches.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize separately to get prompt lengths for masking
    enc_prompt = tokenizer(prompts, add_special_tokens=False, return_tensors=None)
    enc_resp   = tokenizer(responses, add_special_tokens=False, return_tensors=None)

    concat_ids = []
    prompt_lens = []
    for p_ids, r_ids in zip(enc_prompt["input_ids"], enc_resp["input_ids"]):
        prompt_lens.append(len(p_ids))
        # Add an optional leading space/EOS handling is already managed by tokenizer
        ids = p_ids + r_ids
        # Truncate from the left if too long (since we're left padding anyway)
        if len(ids) > max_length:
            ids = ids[-max_length:]
            # If we truncated into the prompt, adjust prompt length accordingly
            prompt_l = max(0, len(ids) - len(r_ids))
            prompt_lens[-1] = prompt_l
        concat_ids.append(ids)

    # Pad to batch
    max_len = max(len(x) for x in concat_ids)
    input_ids = []
    attention_mask = []
    labels = []

    pad_id = tokenizer.pad_token_id

    for ids, p_len in zip(concat_ids, prompt_lens):
        pad_len = max_len - len(ids)
        padded = [pad_id] * pad_len + ids
        amask  = [0] * pad_len + [1] * len(ids)

        # Labels: -100 for prompt tokens + padding; response tokens equal to input_ids (teacher forcing)
        # Note: We shift internally when gathering token logprobs.
        lab = [-100] * (pad_len + p_len) + ids[p_len:]

        input_ids.append(padded)
        attention_mask.append(amask)
        labels.append(lab)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # Shift for causal LM: logits predict next token
    # logits: [B, T, V]; we score positions where labels != -100 at t (which correspond to token at t)
    logits = outputs.logits  # [B, T, V]
    log_probs = log_softmax(logits, dim=-1)  # [B, T, V]

    # Gather log-probs for the gold tokens at positions where labels != -100
    target_tokens = labels.clone()
    # Replace -100 to some valid index to allow gather; mask afterwards
    target_tokens_mask = (target_tokens != -100)
    safe_targets = target_tokens.masked_fill(~target_tokens_mask, 0)  # dummy index 0 where masked

    token_logprobs = log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * target_tokens_mask  # zero out masked positions

    # Sum across sequence to get total response logprob
    seq_logprob = token_logprobs.sum(dim=-1)  # [B]
    return seq_logprob.tolist()


def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Path/name to policy model. "
                        "If --base is given, this is assumed to be a PEFT adapter directory.")
    p.add_argument("--base", default=None,
                   help="Base model name (e.g., allenai/OLMo-2-1B) if --model is a PEFT adapter. "
                        "Omit if --model is a fully merged model.")
    p.add_argument("--pairs", required=True, help="JSONL file with {prompt, chosen, rejected}.")
    p.add_argument("--batch_size", type=int, default=8, help="Pair batch size (each pair makes 2 forward passes).")
    p.add_argument("--max_length", type=int, default=4096, help="Max sequence length for scoring.")
    p.add_argument("--device", default=None, help="cuda | cpu (auto if omitted)")
    p.add_argument("--trust_remote_code", action="store_true", help="Pass through to Transformers loaders.")
    p.add_argument("--dtype", default="auto",
                   help="auto | float16 | bfloat16 | float32")
    args = p.parse_args()

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Resolve dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": torch.float16 if torch.cuda.is_available() else torch.float32
    }
    torch_dtype = dtype_map.get(args.dtype, dtype_map["auto"])

    # Load tokenizer
    tok_src = args.model  # prefer adapter dir tokenizer if present
    if not (Path(tok_src) / "tokenizer.json").exists():
        tok_src = args.base or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if args.base:
        # PEFT path: base + adapter
        base = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=torch_dtype,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=args.trust_remote_code,
        )
        model = _maybe_load_peft(base, args.model)
    else:
        # Fully merged model
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=args.trust_remote_code,
        )
    model.eval()
    model.to(device)

    # Read pairs
    pairs = read_jsonl(args.pairs)
    if len(pairs) == 0:
        print(json.dumps({"error": "No examples found in pairs file.", "pairs": args.pairs}))
        sys.exit(1)

    total = 0
    correct = 0
    margins: List[float] = []

    # Batch over pairs (each pair becomes two lists: chosen and rejected)
    for batch in batched(pairs, args.batch_size):
        prompts_c = [ex["prompt"] for ex in batch]
        prompts_r = [ex["prompt"] for ex in batch]
        chosen    = [ex["chosen"] for ex in batch]
        rejected  = [ex["rejected"] for ex in batch]

        # Score logprob sums
        lp_chosen = score_logprob_sum(model, tokenizer, prompts_c, chosen,
                                      max_length=args.max_length, device=device)
        lp_reject = score_logprob_sum(model, tokenizer, prompts_r, rejected,
                                      max_length=args.max_length, device=device)

        for lc, lr in zip(lp_chosen, lp_reject):
            total += 1
            if lc > lr:
                correct += 1
            margins.append(lc - lr)

    acc = correct / max(1, total)
    mean_margin = float(sum(margins) / max(1, len(margins)))
    # Some extra diagnostics
    wins = sum(1 for m in margins if m > 0)
    ties = sum(1 for m in margins if m == 0)
    losses = total - wins - ties

    result = {
        "pairs_file": args.pairs,
        "model": args.model,
        "base": args.base,
        "total_pairs": total,
        "accuracy": round(acc, 6),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_logprob_margin": round(mean_margin, 6),
        "device": str(device),
        "dtype": str(torch_dtype).replace("torch.", "")
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
