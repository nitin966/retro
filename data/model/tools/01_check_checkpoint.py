import argparse, os, json, math, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@torch.no_grad()
def ppl_on_texts(model, tok, texts, device="cuda", max_len=2048):
    model.eval().to(device)
    total_nll, total_tok = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
        # shift labels = input_ids
        out = model(input_ids=ids, labels=ids)
        nll = out.loss.item() * (ids.numel() - 1)
        total_nll += nll
        total_tok += (ids.numel() - 1)
    return math.exp(total_nll / max(1, total_tok))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # file-size sanity (if local)
    if os.path.isdir(args.model):
        total_bytes = 0
        for r,_,files in os.walk(args.model):
            for f in files:
                if f.endswith((".bin",".safetensors")):
                    total_bytes += os.path.getsize(os.path.join(r,f))
        print(f"[size] checkpoint bytes: {total_bytes/1e9:.2f} GB")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map=None)

    triage_texts = [
        "We propose the Transformer architecture, which relies entirely on attention mechanisms.",
        "In statistics, maximum likelihood estimation seeks parameters that maximize the likelihood function.",
        "FAISS uses product quantization and inverted lists to enable fast approximate nearest neighbor search.",
    ]
    ppl = ppl_on_texts(model, tok, triage_texts, device=args.device)
    print(f"[ppl] quick triage PPL (mixed domain): {ppl:.2f}")

    # quick generations
    model.eval()
    prompts = [
        "Who wrote “Attention Is All You Need”, and in what year?",
        "Explain gradient checkpointing and its trade-offs.",
    ]
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(args.device)
        gen = model.generate(ids, max_new_tokens=128, do_sample=False, temperature=1.0)
        print(f"\n[prompt] {p}\n[gen] {tok.decode(gen[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()

