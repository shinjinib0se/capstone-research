import argparse, json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig

def load_pairs(path, limit=None):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i>=limit: break
            ex=json.loads(line)
            rows.append({
                "prompt": ex["prompt"],
                "chosen": ex["chosen"],
                "rejected": ex["rejected"]
            })
    return Dataset.from_list(rows)

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--eval",  required=True)
ap.add_argument("--base",  default="gpt2")  # tiny for CPU sanity
ap.add_argument("--out",   default="runs/dpo_gpt2")
ap.add_argument("--limit_train", type=int, default=2000)
ap.add_argument("--limit_eval",  type=int, default=200)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--bsz",    type=int, default=2)
ap.add_argument("--lr",     type=float, default=5e-6)
ap.add_argument("--max_len",type=int, default=512)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

train_ds = load_pairs(args.train, args.limit_train)
eval_ds  = load_pairs(args.eval,  args.limit_eval)

policy = AutoModelForCausalLM.from_pretrained(args.base)
policy.config.use_cache = False
policy.gradient_checkpointing_enable()

cfg = DPOConfig(
    beta=0.1,                      # strength of preference
    max_length_prompt=args.max_len,
    max_length=args.max_len,
)

trainer = DPOTrainer(
    model=policy,
    ref_model=None,                # use implicit frozen copy
    args=TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=50,
        report_to="none"
    ),
    beta=cfg.beta,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tok,
    max_length=args.max_len,
    max_prompt_length=args.max_len,
)

trainer.train()
trainer.save_model(args.out)
tok.save_pretrained(args.out)
print("Saved DPO model to", args.out)
