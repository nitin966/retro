#!/usr/bin/env python3
import os, sys, argparse
import numpy as np
from tqdm import tqdm
import torch

# --- make the env behave on macOS / avoid TF/JAX surprises ---
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from sentence_transformers import SentenceTransformer
from tokenize_and_chunk import get_tokenizer as get_chunk_tokenizer

def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def main(args):
    device = pick_device()
    chunk_tokenizer = get_chunk_tokenizer()
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # memmap input for speed/low RAM
    chunks = np.load(args.input, mmap_mode="r")   # shape [N, chunk_size]
    N = chunks.shape[0]
    if args.verbose:
        print(f"Input: {args.input}  shape={chunks.shape}  dtype={chunks.dtype}")
        print(f"Device: {device}  batch_size={args.batch_size}")
        print("Decoding chunks...")

    # ---- FAST decode: batch_decode in sizeable blocks ----
    # This produces the exact same strings as per-row decode with skip_special_tokens=True.
    decode_bs = max(args.batch_size, 128)  # decoding is cheap; go bigger
    texts = []
    rng = range(0, N, decode_bs)
    if args.verbose:
        rng = tqdm(rng, total=(N + decode_bs - 1) // decode_bs, desc="Decoding", unit="batch")
    for start in rng:
        stop = min(start + decode_bs, N)
        # convert the slice to python lists once (batch_decode likes lists)
        batch_ids = chunks[start:stop].tolist()
        batch_txt = chunk_tokenizer.batch_decode(
            batch_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        texts.extend(batch_txt)

    # ---- Embedding (same as your logic: single encode call over entire list) ----
    if args.verbose:
        print("Embedding...")
    sentence_embeddings = model.encode(
        texts,
        batch_size=args.batch_size,                  # compute batch size
        device=device,                               # add MPS support
        convert_to_numpy=True,
        output_value="sentence_embedding",
        normalize_embeddings=True,
        show_progress_bar=args.verbose
    )

    # cast to fp16 just like your script
    sentence_embeddings = np.asarray(sentence_embeddings).astype(np.float16, copy=False)
    np.save(args.output, sentence_embeddings)
    if args.verbose:
        print(f"Saved: {args.output}  shape={sentence_embeddings.shape}  dtype={sentence_embeddings.dtype}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Tokenized chunks (*.npy)")
    p.add_argument("output", help="Output file name (*.npy)")
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    main(args)

