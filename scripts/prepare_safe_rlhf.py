# scripts/prepare_safe_rlhf.py
import json, argparse, random
from datasets import load_dataset

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_id", default="PKU-Alignment/PKU-SafeRLHF-30K")  # or ...-10K
    ap.add_argument("--train_out", default="data/rlhf_pairs_train.jsonl")
    ap.add_argument("--eval_out",  default="data/rlhf_pairs_eval.jsonl")
    ap.add_argument("--rollouts_out", default="data/rollout_prompts.jsonl")
    ap.add_argument("--train_frac", type=float, default=0.95)  # split pairs into train/eval
    ap.add_argument("--max_pairs", type=int, default=20000)    # limit for first run; raise later
    ap.add_argument("--max_rollouts", type=int, default=2000)  # rollout prompts count
    args = ap.parse_args()

    ds = load_dataset(args.hf_id)
    split = ds["train"]  # most releases store everything in 'train'
    print("Example keys:", split.column_names)
    # Inspect one row if you’re curious:
    # print(split[0])

    # NOTE: Different releases have slightly different field names.
    # The common pattern: a prompt + two responses + a preference label/index.
    # We try a few common schemas below.
    pairs = []
    for ex in split:
        prompt = ex.get("question") or ex.get("prompt") or ex.get("instruction") or ""
        if not prompt:
            continue

        # Try to find two responses and which is preferred
        r0 = ex.get("response_0") or (ex.get("responses") or [None, None])[0]
        r1 = ex.get("response_1") or (ex.get("responses") or [None, None])[1]
        if r0 is None or r1 is None:
            continue

        # Preferred index: often 0/1 or in a field like 'preference', 'label', 'chosen'
        pref = ex.get("preference")
        if pref is None:
            pref = ex.get("label")
        if pref is None and "chosen" in ex and "rejected" in ex:
            chosen = ex["chosen"]; rejected = ex["rejected"]
        else:
            if pref is None:
                # Some datasets store a score for each; take the larger
                s0 = ex.get("score_0", 0.0); s1 = ex.get("score_1", 0.0)
                pref = 0 if s0 >= s1 else 1
            chosen  = r0 if pref == 0 else r1
            rejected= r1 if pref == 0 else r0

        # Filter obviously broken rows
        if not chosen or not rejected:
            continue

        pairs.append({"prompt": prompt.strip(),
                      "chosen": str(chosen).strip(),
                      "rejected": str(rejected).strip()})

    print(f"Loaded {len(pairs)} pairs before limiting")
    random.seed(42); random.shuffle(pairs)
    pairs = pairs[:args.max_pairs]

    # Train/eval split
    cut = int(len(pairs) * args.train_frac)
    train_pairs = pairs[:cut]
    eval_pairs  = pairs[cut:]
    print(f"Writing {len(train_pairs)} train pairs, {len(eval_pairs)} eval pairs")

    # Rollout prompts (unique prompts; you can also add your own)
    prompts = []
    seen = set()
    for p in pairs:
        pr = p["prompt"]
        if pr not in seen:
            seen.add(pr); prompts.append({"prompt": pr})
            if len(prompts) >= args.max_rollouts:
                break

    write_jsonl(args.train_out, train_pairs)
    write_jsonl(args.eval_out,  eval_pairs)
    write_jsonl(args.rollouts_out, prompts)
    print("Done:", args.train_out, args.eval_out, args.rollouts_out)

if __name__ == "__main__":
    main()
