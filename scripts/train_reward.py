from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch, argparse, json

def to_rows(path, limit=None, pos="chosen", neg="rejected"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit: break
            ex = json.loads(line)
            rows += [
                {"text": ex["prompt"] + "\n\n" + ex[pos], "labels": 1.0},
                {"text": ex["prompt"] + "\n\n" + ex[neg], "labels": 0.0},
            ]
    return Dataset.from_list(rows)

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--eval", required=True)
ap.add_argument("--base", default="distilroberta-base")  # << smaller & faster
ap.add_argument("--out", default="runs/reward_model_fast")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bsz", type=int, default=16)
ap.add_argument("--limit_train", type=int, default=5000)   # << cap data
ap.add_argument("--limit_eval",  type=int, default=500)
ap.add_argument("--max_length", type=int, default=256)     # << shorter seq
ap.add_argument("--freeze_encoder", action="store_true")   # train head-only
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)

dtrain = to_rows(args.train, limit=args.limit_train)
deval  = to_rows(args.eval,  limit=args.limit_eval)

def tok_fn(b):
    return tok(b["text"], truncation=True, padding="max_length", max_length=args.max_length)
dtrain = dtrain.map(tok_fn, batched=True, num_proc=1)
deval  = deval.map(tok_fn,  batched=True, num_proc=1)

cols = ["input_ids", "attention_mask", "labels"]
dtrain.set_format(type="torch", columns=cols)
deval.set_format(type="torch", columns=cols)

model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1)

# Optional: freeze encoder -> train only classifier (much faster)
if args.freeze_encoder:
    for n, p in model.named_parameters():
        if "classifier" not in n:
            p.requires_grad = False

def loss_fn(model, inputs, return_outputs=False, **kwargs):
    out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = out.logits.squeeze(-1)
    labels = inputs["labels"].to(dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    return (loss, out) if return_outputs else loss

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=1e-4 if args.freeze_encoder else 2e-5,  # faster when head-only
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",   # fewer evals = faster
        logging_steps=50,
        save_strategy="no",            # skip mid-run checkpoints
        bf16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,      # WSL safety
    ),
    train_dataset=dtrain,
    eval_dataset=deval,
    tokenizer=tok,
)
trainer.compute_loss = loss_fn
trainer.train()

# quick sanity: preferred > rejected
model.eval()
with torch.no_grad():
    batch = deval.select(range(min(64, len(deval)))).to_dict()
    import numpy as np, torch as T
    labels = np.array(batch["labels"])
    for name, mask in [("pos", labels == 1), ("neg", labels == 0)]:
        if mask.sum() == 0: continue
        inp = {
            "input_ids": T.tensor(batch["input_ids"])[mask],
            "attention_mask": T.tensor(batch["attention_mask"])[mask],
        }
        if T.cuda.is_available():
            inp = {k: v.cuda() for k, v in inp.items()}
            model.cuda()
        s = model(**inp).logits.squeeze(-1).detach().cpu()
        print(f"[sanity] {name} mean score:", float(s.mean()))

trainer.save_model(args.out)
tok.save_pretrained(args.out)
print("Saved to", args.out)
