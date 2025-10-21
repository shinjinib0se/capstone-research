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
    beta=0.3,  # strength of preference - lets try 0.3-0.5 for now (no larger than 1 tho)
    # max_length_prompt=args.max_len,
    # max_length=args.max_len,
    # loss_type = "sigmoid", # explicit. this helps avoid softmax temperature damping
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
    #beta=cfg.beta,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    # tokenizer=tok,
    # max_length=args.max_len,
    # max_prompt_length=args.max_len,
    dpo_config=cfg,
)

# this will create a training_log.csv file with loss values per step and epoch - can be plotted later with:

# import pandas as pd, matplotlib.pyplot as plt
# df = pd.read_csv("runs/dpo_olmo2/training_log.csv")
# plt.plot(df["step"], df["loss"])
# plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("DPO Loss Curve")
# plt.show()


import csv, os
logfile = os.path.join(args.out, "training_log.csv")
os.makedirs(args.out, exist_ok=True)

with open(logfile, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "epoch", "loss"])

def log_callback(state, control, **kwargs):
    if state.log_history and "loss" in state.log_history[-1]:
        step = state.log_history[-1].get("step", 0)
        loss = state.log_history[-1]["loss"]
        epoch = state.epoch
        with open(logfile, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([step, epoch, loss])

trainer.add_callback(type("Logger", (), {"on_log": log_callback}))



trainer.train()
trainer.save_model(args.out)

# this will automatically print the preference accuracy and mean log-prob margin for the fine-tuned checkpoint to compare immediately with base model
from scripts.eval_preference_accuracy import main as eval_main
import sys
sys.argv = [
    "eval_preference_accuracy.py",
    "--model", args.out,
    "--pairs", args.eval,
]
eval_main()

tok.save_pretrained(args.out)
print("Saved DPO model to", args.out)
