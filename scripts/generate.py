from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, argparse, os

def load_model(model_id, use_4bit=False):
    tok = AutoTokenizer.from_pretrained(model_id)
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

def run(model_id, infile, outfile, max_new_tokens=256, temperature=0.7):
    tok, model = load_model(model_id)
    out = []
    with open(infile, "r", encoding="utf-8") as f:
        for line in f:
            prompt = json.loads(line)["prompt"]
            inputs = tok(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k:v.to(model.device) for k,v in inputs.items()}
            gen = model.generate(
                **inputs, do_sample=True, temperature=temperature,
                max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id
            )
            text = tok.decode(gen[0], skip_special_tokens=True)
            out.append({"prompt": prompt, "output": text})
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as g:
        json.dump(out, g, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()
    run(args.model_id, args.infile, args.outfile, args.max_new_tokens, args.temperature)
