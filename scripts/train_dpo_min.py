#!/usr/bin/env python
# DPO fine-tuning for PKU-style pairs (prompt/chosen/rejected) — TRL 0.23 compatible.

import argparse, json, os, random, torch
from typing import Dict, List, Optional, Tuple

from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig

# ---------- utils ----------
def _has_flash2() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False

FIELD_ALIASES = {
    "prompt":  ["prompt", "input", "instruction", "question", "context"],
    "chosen":  ["chosen", "response_chosen", "preferred", "pos", "answer_chosen"],
    "rejected":["rejected","response_rejected","dispreferred","neg","answer_rejected"],
}

def pick(d: Dict, keys: List[str]):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return None

def normalize(rec: Dict) -> Optional[Tuple[str, str, str]]:
    P = pick(rec, FIELD_ALIASES["prompt"])
    C = pick(rec, FIELD_ALIASES["chosen"])
    R = pick(rec, FIELD_ALIASES["rejected"])
    if (C is None or R is None) and {"output_1","output_2","label"} <= set(rec):
        C, R = (rec["output_1"], rec["output_2"]) if str(rec["label"]) in {"1","A"} else (rec["output_2"], rec["output_1"])
    return (P, C, R) if (P and C and R) else None

def load_pairs_as_hfdataset(path: str, limit: Optional[int] = None) -> HFDataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit: break
            rec = json.loads(line)
            t = normalize(rec)
            if t:
                p, c, r = t
                rows.append({"prompt": p, "chosen": c, "rejected": r})
    if not rows:
        raise ValueError(f"No usable rows in {path}. Expect PKU-style prompt/chosen/rejected.")
    return HFDataset.from_list(rows)

# ---------- args ----------
def parse_args():
    ap = argparse.ArgumentParser(description="DPO training on preference pairs (TRL 0.23).")
    ap.add_argument("--model", required=True, help="HF id or local path for policy (e.g., allenai/OLMo-2-0425-1B).")
    ap.add_argument("--ref_model", default=None, help="HF id/path for reference (defaults to --model).")
    ap.add_argument("--train", required=True, help="Train JSONL with PKU-style pairs.")
    ap.add_argument("--eval", help="Eval JSONL with PKU-style pairs.")
    ap.add_argument("--out", required=True, help="Output dir.")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=1, help="Per-device train batch size.")
    ap.add_argument("--gas", type=int, default=16, help="Gradient accumulation steps.")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta temperature.")
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--no_flash", action="store_true", help="Disable FlashAttention2 and use SDPA.")
    # precision & speed
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    # quantization + LoRA
    ap.add_argument("--q4", action="store_true", help="Load in 4-bit (QLoRA).")
    ap.add_argument("--q8", action="store_true", help="Load in 8-bit.")
    ap.add_argument("--lora", action="store_true", help="Enable LoRA adapters.")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--reference_free", action="store_true", help="Use reference-free DPO (no ref model in memory).")
    ap.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing.")
    ap.add_argument("--max_prompt_len", type=int, default=512)
    return ap.parse_args()

# ---------- main ----------
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # quant + dtype
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    bnb = None
    if args.q4 or args.q8:
        bnb = BitsAndBytesConfig(
            load_in_4bit=args.q4,
            load_in_8bit=(args.q8 and not args.q4),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype or torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    attn_impl = "sdpa" if (args.no_flash or not _has_flash2()) else "flash_attention_2"

    # policy model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        quantization_config=bnb,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # optional LoRA
    if args.lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if args.q4 or args.q8:
            model = prepare_model_for_kbit_training(model)
        target = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        lconf = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                           target_modules=target, task_type="CAUSAL_LM", bias="none")
        model = get_peft_model(model, lconf)

    # Enable grad checkpointing here (after LoRA/k-bit prep, before trainer)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        try:
            model.config.use_cache = False  # required for grad ckpt to work
        except Exception:
            pass

    # datasets (must be 🤗 Datasets with .map for TRL 0.23)
    train_ds = load_pairs_as_hfdataset(args.train)
    eval_ds  = load_pairs_as_hfdataset(args.eval) if args.eval else None

    # DPOConfig packs both training + DPO hyperparams in TRL 0.23
    dpo_cfg = DPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=max(1, args.bsz),
        gradient_accumulation_steps=args.gas,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=["tensorboard"],
        bf16=args.bf16,
        fp16=(args.fp16 and not args.bf16),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=(args.eval_steps if eval_ds is not None else None),
        beta=args.beta,
        loss_type="sigmoid",
        max_length=args.max_len,
        max_prompt_length=args.max_prompt_len,
        precompute_ref_log_probs=False,
        reference_free=args.reference_free,     
        gradient_checkpointing=args.grad_ckpt,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=(None if args.reference_free else (args.ref_model or args.model)),
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,   # tokenizer / processor in TRL 0.23
        # no custom data_collator needed; TRL builds its own
    )

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

if __name__ == "__main__":
    main()
