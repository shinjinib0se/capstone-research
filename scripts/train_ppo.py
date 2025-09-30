# scripts/train_ppo.py
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
import torch, argparse

def reward_fn_init(rm_id):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    rtok = AutoTokenizer.from_pretrained(rm_id, use_fast=True)
    rmod = AutoModelForSequenceClassification.from_pretrained(rm_id).eval()
    if torch.cuda.is_available():
        rmod.to("cuda")

    @torch.no_grad()
    def score(prompts, responses):
        texts = [p.strip() + "\n\n" + r.strip() for p, r in zip(prompts, responses)]
        enc = rtok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}
        logits = rmod(**enc).logits.squeeze(-1).float().detach().cpu()
        # standardize per batch for PPO stability
        r = (logits - logits.mean()) / (logits.std() + 1e-6)
        return r.tolist()

    return score

ap = argparse.ArgumentParser()
ap.add_argument("--policy_id", default="gpt2")  # small sanity default
ap.add_argument("--reward_model", default="runs/reward_model_fast")
ap.add_argument("--rollouts", default="data/rollout_prompts.small.jsonl")
ap.add_argument("--out", default="runs/ppo_sanity")
ap.add_argument("--save_steps", type=int, default=10)
ap.add_argument("--batch_size", type=int, default=1)
ap.add_argument("--mini_batch_size", type=int, default=1)
ap.add_argument("--max_new_tokens", type=int, default=48)
ap.add_argument("--kl_coef", type=float, default=0.05)
ap.add_argument("--epochs", type=int, default=1)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.policy_id, use_fast=True)
# GPT-2 (and some small models) don't have a pad token
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    args.policy_id,
    low_cpu_mem_usage=True,
)
if torch.cuda.is_available():
    policy.to("cuda")

cfg = PPOConfig(
    batch_size=args.batch_size,
    mini_batch_size=args.mini_batch_size,
    learning_rate=1e-6,
    seed=42,
)

ds = load_dataset("json", data_files={"rollouts": args.rollouts})["rollouts"]
prompts = [r["prompt"] for r in ds]

ppo = PPOTrainer(config=cfg, model=policy, ref_model=None, tokenizer=tok)
rm_score = reward_fn_init(args.reward_model)

step = 0
device = ppo.accelerator.device if torch.cuda.is_available() else torch.device("cpu")

for _ in range(args.epochs):
    for i in range(0, len(prompts), cfg.batch_size):
        batch_prompts = prompts[i : i + cfg.batch_size]

        # 1) Build query_tensors: list[Tensor(seq_len)], one per prompt
        query_tensors = []
        for p in batch_prompts:
            enc = tok(p, return_tensors="pt", truncation=True, max_length=512)
            q = enc["input_ids"].squeeze(0).to(device)  # 1D tensor (seq_len)
            query_tensors.append(q)

        # 2) Generate continuations (list[Tensor(new_len)])
        response_tensors = ppo.generate(
            query_tensors,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            pad_token_id=tok.pad_token_id,
        )

        # 3) Decode ONLY the generated continuations
        response_texts = [tok.decode(t, skip_special_tokens=True) for t in response_tensors]

        # 4) Rewards from RM on (prompt, response)
        rewards = rm_score(batch_prompts, response_texts)  # list[float]

        # 5) PPO update
        ppo.step(query_tensors, response_tensors, rewards)
        step += 1

        # 6) Save checkpoints
        if step % args.save_steps == 0:
            ck = f"{args.out}/checkpoints/step_{step:05d}"
            ppo.save_pretrained(ck)
            tok.save_pretrained(ck)

# final save
ck = f"{args.out}/checkpoints/step_{step:05d}_final"
ppo.save_pretrained(ck)
tok.save_pretrained(ck)
print("Saved PPO checkpoints under", args.out)
