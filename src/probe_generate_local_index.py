#!/usr/bin/env python3
"""
probe_generate_local_index.py
- Loads Retro model from Lightning ckpt (no repo edits)
- Uses local FAISS index + local embedding matrix (no HTTP)
- Retrieves neighbor chunks directly from shards
- Runs a single .generate() call
- Cleans up safely on macOS to avoid FAISS destructor segfaults
"""

import os
# --- safer defaults for macOS / FAISS / tokenizers ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("FAISS_DISABLE_FORK", "1")

import argparse, json, io, math
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import faiss

# project-local imports
from modeling_retro import RetroConfig, RetroModelLMHead
from dataset_retro import ChunkedSequenceDataset, ShardedChunkedSequenceDataset
from data.tokenize_and_chunk import get_tokenizer
from sentence_transformers import SentenceTransformer


def load_gen_model_from_lightning_ckpt(ckpt_path: Path, config: RetroConfig, retriever, map_location="cpu"):
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model = RetroModelLMHead(config, retriever=retriever)
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith(("base.", "lm_head.")):
            cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)
    try:
        model.tie_lm_head_embeddings()
    except Exception:
        pass
    return model


class LocalFaissRetriever:
    """
    Minimal retriever that:
    - encodes chunk text with a SentenceTransformer
    - searches local FAISS index (already trained & added)
    - returns neighbor token ids pulled from shards (with continuation chunks)
    """
    def __init__(self, index, retrieval_dataset, retriever_model, tokenizer,
                 num_neighbours: int, chunk_size: int, num_continuation_chunks: int, verbose: bool = False):
        self.index = index
        self.retrieval_dataset = retrieval_dataset
        self.retriever_model = retriever_model
        self.tokenizer = tokenizer
        self._num_neighbours = num_neighbours
        self._neighbour_len = chunk_size * (1 + num_continuation_chunks)
        self.chunk_size = chunk_size
        self.num_continuation_chunks = num_continuation_chunks
        self.verbose = verbose

    @property
    def num_neighbours(self): return self._num_neighbours
    @property
    def neighbour_len(self): return self._neighbour_len

    def retrieve_neighbours(self, chunks: torch.LongTensor):
        # chunks: [B, chunk_size]
        B, L = chunks.shape
        assert L == self.chunk_size, f"expected chunk_size={self.chunk_size}, got {L}"
        # decode -> embed -> FAISS search
        texts = [self.tokenizer.decode(chunks[i].tolist(), skip_special_tokens=True) for i in range(B)]
        q = self.retriever_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        q = np.asarray(q, dtype=np.float32, order="C")
        D, I = self.index.search(q, self.num_neighbours)
        if self.verbose:
            print(f"[probe] FAISS search I[0][:4] =", I[0, :min(4, self.num_neighbours)])
        # fetch neighbor tokens
        out = torch.full((B, self.num_neighbours, self.neighbour_len), fill_value=0, dtype=torch.long)
        for b in range(B):
            for k in range(self.num_neighbours):
                idx = int(I[b, k])
                if idx < 0:
                    continue
                toks = self.retrieval_dataset.get_chunk_tokens(
                    idx, include_continuation_chunks=self.num_continuation_chunks
                )  # np.uint16
                toks = toks.astype(np.int64)[: self.neighbour_len]
                out[b, k, :len(toks)] = torch.from_numpy(toks)
                if self.verbose and b == 0:
                    s = self.tokenizer.decode(toks.tolist(), skip_special_tokens=True)[:120].replace("\n"," ")
                    print(f"[probe] neigh#{k} idx={idx} len={len(toks)} -> {s}")
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retro-config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--prompt", default="Explain the transformer architecture.")
    ap.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto")

    # dataset / index
    ap.add_argument("--chunks", type=Path, required=True, help=".../chunks.npy")
    ap.add_argument("--seq2chunk", type=Path, required=True, help=".../seq2chunk.npy")
    ap.add_argument("--chunk2seq", type=Path, required=True, help=".../chunk2seq.npy")
    ap.add_argument("--embeddings", type=Path, required=True, help=".../chunks.embeddings.npy (float16)")
    ap.add_argument("--faiss-index", type=Path, required=True, help=".../IVF...,PQ....index (already trained)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--num-continuation-chunks", type=int, default=1)
    ap.add_argument("--retriever-model-name", default="all-MiniLM-L6-v2")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type != "cpu":
        try: torch.set_default_device(device)
        except Exception: pass

    # load config + tokenizer
    config = RetroConfig(**json.loads(args.retro_config.read_text()))
    tok = get_tokenizer()

    # shards / dataset
    ds = ShardedChunkedSequenceDataset([
        ChunkedSequenceDataset(chunks=args.chunks, seq2chunk=args.seq2chunk, chunk2seq=args.chunk2seq)
    ])
    if args.verbose:
        print(f"[probe] shards ok; total_num_chunks={ds.total_num_chunks}")

    # load FAISS index (set single thread for stability/perf on laptop)
    faiss.omp_set_num_threads(1)
    index = faiss.read_index(str(args.faiss_index))
    d = index.d
    if args.verbose:
        print(f"[probe] faiss d={d} ntotal={getattr(index,'ntotal','?')}")

    # verify embeddings dim and optionally add if ntotal==0
    emb = np.load(args.embeddings, mmap_mode="r")
    if emb.dtype != np.float16:
        raise ValueError("embeddings must be float16 (as produced by your embed script)")
    if emb.shape[1] != d:
        raise ValueError(f"embeddings dim {emb.shape[1]} != faiss d {d}")
    if getattr(index, "ntotal", 0) == 0:
        if args.verbose: print("[probe] index empty; adding all embeddings (this may take a bit)...")
        index.add(np.asarray(emb, dtype=np.float32, order="C"))
        if args.verbose: print(f"[probe] after add ntotal={index.ntotal}")

    # build retriever
    sbert = SentenceTransformer(args.retriever_model_name)
    retriever = LocalFaissRetriever(
        index=index,
        retrieval_dataset=ds,
        retriever_model=sbert,
        tokenizer=tok,
        num_neighbours=args.k,
        chunk_size=config.chunk_size,
        num_continuation_chunks=args.num_continuation_chunks,
        verbose=args.verbose
    )

    # load gen model
    model = load_gen_model_from_lightning_ckpt(
        ckpt_path=args.checkpoint, config=config, retriever=retriever,
        map_location="cpu" if device.type == "cpu" else device
    ).to(device).eval()

    # --- Generation shims (HF >= 4.3x compat) ---
    model.can_generate = MethodType(lambda self: True, model)
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        pad_token_id=config.pad_token_idx, eos_token_id=1, use_cache=False
    )
    model._supports_cache_class = True
    model.device = device

    _orig_pifg = model.prepare_inputs_for_generation
    def _pifg(self, input_ids: torch.LongTensor, **kwargs):
        out = _orig_pifg(input_ids, **kwargs)
        out["input_ids"] = out["input_ids"].to(self.device)
        if "neighbour_ids" in out:
            out["neighbour_ids"] = out["neighbour_ids"].to(self.device)
        return out
    model.prepare_inputs_for_generation = MethodType(_pifg, model)
    # ---------------------------------------------

    # tokenize prompt and generate
    input_ids = tok([args.prompt], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    if args.verbose:
        print("[probe] generating...")
    with torch.no_grad():
        out = model.generate(
            inputs=input_ids,
            do_sample=False, num_beams=1,
            min_length=10, max_length=128,
            no_repeat_ngram_size=3,
            pad_token_id=config.pad_token_idx, eos_token_id=1
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    print("\n=== RETRO LOCAL-INDEX GENERATION ===")
    print(text)
    print("====================================\n")

    # explicit cleanup to avoid FAISS teardown crashes on macOS
    del model, sbert, index, ds, emb, tok
    os._exit(0)


if __name__ == "__main__":
    main()

