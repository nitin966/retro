#!/usr/bin/env python3
import argparse, json, io, zipfile, os
from pathlib import Path
from types import MethodType

import numpy as np
import torch

# Local project imports
from modeling_retro import RetroConfig, RetroModelLMHead
from retrieval import IndexServiceRetriever, IndexServiceClient
from dataset_retro import ChunkedSequenceDataset, ShardedChunkedSequenceDataset
from data.tokenize_and_chunk import get_tokenizer
from sentence_transformers import SentenceTransformer

# Keep HF tokenizers quiet when forking happens inside generation
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ------------------------ Lightning ckpt → generation model ------------------------
def load_gen_model_from_lightning_ckpt(ckpt_path: Path, config: RetroConfig, retriever, map_location="cpu"):
    """
    Build RetroModelLMHead and load weights from a Lightning checkpoint.
    No changes to your repo are required.
    """
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    if "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint has no 'state_dict': {ckpt_path}")
    state_dict = ckpt["state_dict"]

    model = RetroModelLMHead(config, retriever=retriever)

    # Clean possible Lightning prefixes
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("base.") or k.startswith("lm_head."):
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if unexpected:
        print(f"[probe] WARN: unexpected keys ignored: {len(unexpected)} (first 5): {unexpected[:5]}")
    if missing:
        print(f"[probe] WARN: missing keys filled by init: {len(missing)} (first 5): {missing[:5]}")

    try:
        model.tie_lm_head_embeddings()
    except Exception:
        pass

    return model


# ------------------------ Build retrieval dataset & client ------------------------
def build_retrieval(
    index_spec: Path,
    index_host: str,
    index_port: int,
    tokenizer,
    num_neighbours: int,
    chunk_size: int,
    num_continuation_chunks: int,
    retriever_model_name: str,
    verbose: bool,
):
    spec_json = json.loads(index_spec.read_text())
    shards = spec_json if isinstance(spec_json, list) else spec_json.get("shards", [])
    if not shards:
        raise ValueError(f"No shards in index spec: {index_spec}")

    base = index_spec.parent
    retrieval_dataset = ShardedChunkedSequenceDataset([
        ChunkedSequenceDataset(
            chunks=base / s["chunks"],
            seq2chunk=base / s["seq2chunk"],
            chunk2seq=base / s["chunk2seq"],
        )
        for s in shards
    ])

    client = IndexServiceClient(index_host, index_port)
    # Replace ONLY this instance's `.query` with a robust loader
    client.query = MethodType(_safe_query_zip, client)

    # Quick health check
    if not client.is_available():
        raise RuntimeError(f"Index service not reachable at http://{index_host}:{index_port}/health")

    retrieval_model = SentenceTransformer(retriever_model_name)

    retriever = IndexServiceRetriever(
        index_service=client,
        retrieval_dataset=retrieval_dataset,
        retrieval_model=retrieval_model,
        tokenizer=tokenizer,
        num_neighbours=num_neighbours,
        chunk_size=chunk_size,
        num_continuation_chunks=num_continuation_chunks,
        verbose=verbose,
    )

    # Patch retriever to guard bad indices and oversize neighbour tokens
    retriever.get_neighbours_for_chunk = MethodType(_safe_get_neighbours_for_chunk, retriever)

    return retriever


# ------------------------ Robust FAISS zip reply reader ------------------------
def _safe_query_zip(self, embeddings, k: int):
    """
    Robustly read FAISS server zip reply. Supports:
      - entries named 'distances', 'indices', 'embeddings' (with/without .npy)
      - raw binary float32/int64 blobs (no npy header)
    Returns: (distances [B,K], indices [B,K], embeddings [B,K,D]) as numpy arrays.
    """
    import requests

    # Send the query
    buf = io.BytesIO()
    np.save(buf, embeddings)  # default npy
    resp = requests.post(f"http://{self.host}:{self.port}/query?k={k}", data=buf.getvalue())
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if "distances" not in names and "distances.npy" not in names:
            # Some servers might use 'D' or 'dists'; collect and surface
            print(f"[probe][index] zip entries: {names}")

        def _read_any(candidates, allow_pickle=False):
            last_err = None
            for nm in candidates:
                for variant in (nm, nm + ".npy"):
                    if variant in names:
                        data = zf.read(variant)
                        # Try np.load first
                        try:
                            return np.load(io.BytesIO(data), allow_pickle=allow_pickle)
                        except Exception as e:
                            last_err = e
                            # Try raw buffer interpretations as fallback
                            # Heuristics: float32 for distances/embeddings, int64 for indices
                            try:
                                if nm.startswith("dist"):
                                    arr = np.frombuffer(data, dtype="<f4")
                                    return arr
                                if nm.startswith("ind"):
                                    arr = np.frombuffer(data, dtype="<i8")
                                    return arr
                                if nm.startswith("emb"):
                                    arr = np.frombuffer(data, dtype="<f4")
                                    return arr
                            except Exception as e2:
                                last_err = e2
            raise RuntimeError(f"Could not load any of {candidates}; last error: {last_err}")

        distances = _read_any(["distances", "D", "dists"], allow_pickle=True)
        indices   = _read_any(["indices", "I", "idx"],   allow_pickle=True)
        embs      = _read_any(["embeddings", "E", "vecs"], allow_pickle=True)

    # If any are 1D raw buffers, try to infer B,K,D
    def _ensure_2d(x):
        return x if x.ndim == 2 else x.reshape(-1, x.size // k) if x.ndim == 1 else x

    distances = _ensure_2d(distances)
    indices   = _ensure_2d(indices)

    if embs.ndim == 1:
        # Cannot infer D without context; leave as 1D and let retriever skip if shapes mismatch
        pass

    return distances, indices, embs


# ------------------------ Safe neighbour fetch (skips bad ids, truncates) ------------------------
def _safe_get_neighbours_for_chunk(self, chunk: torch.LongTensor):
    """
    Drop-in for IndexServiceRetriever.get_neighbours_for_chunk with guards:
      - Skip invalid neighbour indices (-1 or out-of-range)
      - Truncate token sequences to self.neighbour_len
    """
    chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)
    if self.verbose:
        print("Retrieving neighbours for chunk:")
        print(chunk_text)
        print()

    # Encode the query to embedding
    embedding = self.retrieval_model.encode(
        [chunk_text],
        output_value="sentence_embedding",
        normalize_embeddings=True
    )

    # Query the FAISS service
    distances, neighbour_chunk_indices, _ = self.index_service.query(embedding, k=self.num_neighbours)
    idxs = np.asarray(neighbour_chunk_indices).reshape(-1).tolist()

    # Prepare output (pad by default)
    pad_id = getattr(self.tokenizer, "pad_token_id", 0)
    neighbours = torch.full(
        (self.num_neighbours, self.neighbour_len),
        fill_value=pad_id,
        dtype=torch.int64,
    )

    # Total chunk count for validity checks
    total_chunks = getattr(self.retrieval_dataset, "num_chunks", None)
    if total_chunks is None:
        try:
            total_chunks = sum(getattr(ds, "num_chunks", 0) for ds in self.retrieval_dataset.datasets)
        except Exception:
            total_chunks = None

    filled = 0
    for ni, ci in enumerate(idxs):
        # Validate index
        if ci is None or (isinstance(ci, (int, np.integer)) and ci < 0):
            if self.verbose:
                print(f"Neighbour {ni}: skipped (index {ci})")
            continue
        if total_chunks is not None and ci >= total_chunks:
            if self.verbose:
                print(f"Neighbour {ni}: skipped (out of range {ci} >= {total_chunks})")
            continue

        # Fetch tokens (chunk + continuation(s))
        try:
            tokens = self.retrieval_dataset.get_chunk_tokens(
                int(ci),
                include_continuation_chunks=self.num_continuation_chunks
            )
        except Exception as e:
            if self.verbose:
                print(f"Neighbour {ni}: skipped (failed to fetch chunk {ci}: {e})")
            continue

        tokens = np.asarray(tokens, dtype=np.int64).ravel()
        if tokens.size > self.neighbour_len:
            tokens = tokens[: self.neighbour_len]

        neighbours[ni, : tokens.size] = torch.from_numpy(tokens)
        filled += 1

        if self.verbose:
            try:
                print(f"Neighbour {ni}:")
                print(self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True))
            except Exception:
                pass

    if self.verbose:
        print()

    return neighbours


# ------------------------ CLI & main ------------------------
def main():
    ap = argparse.ArgumentParser(description="Probe: Lightning ckpt → RetroModelLMHead → generate() with FAISS neighbours")
    ap.add_argument("--retro-config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--prompt", default="Explain the transformer architecture in one paragraph.")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    # Retrieval options
    ap.add_argument("--use-index", action="store_true", help="Use FAISS IndexServiceRetriever; else a dummy is not provided here.")
    ap.add_argument("--index-spec", type=Path, help="Path to index.spec.json (required with --use-index)")
    ap.add_argument("--index-host", default="localhost")
    ap.add_argument("--index-port", type=int, default=8000)
    ap.add_argument("--retriever-model-name", default="all-MiniLM-L6-v2")
    ap.add_argument("--num-neighbours", type=int, default=2)
    ap.add_argument("--num-continuation-chunks", type=int, default=1)
    ap.add_argument("--retrieval-verbose", action="store_true")

    args = ap.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Best-effort default device (helps on MPS/CUDA)
    if device.type != "cpu":
        try:
            torch.set_default_device(device)
        except Exception:
            pass

    # Config + tokenizer
    config = RetroConfig(**json.loads(args.retro_config.read_text()))
    tokenizer = get_tokenizer()

    # Retrieval
    if args.use_index:
        if not args.index_spec:
            ap.error("--use-index requires --index-spec")
        retriever = build_retrieval(
            index_spec=args.index_spec,
            index_host=args.index_host,
            index_port=args.index_port,
            tokenizer=tokenizer,
            num_neighbours=args.num_neighbours,
            chunk_size=config.chunk_size,
            num_continuation_chunks=args.num_continuation_chunks,
            retriever_model_name=args.retriever_model_name,
            verbose=args.retrieval_verbose,
        )
        print("[probe] Using IndexServiceRetriever")
    else:
        raise SystemExit("This probe expects --use-index for end-to-end testing.")

    # Load model
    print("[probe] Loading model weights from checkpoint...")
    gen_model = load_gen_model_from_lightning_ckpt(
        ckpt_path=args.checkpoint,
        config=config,
        retriever=retriever,
        map_location="cpu" if device.type == "cpu" else device,
    ).to(device).eval()

    # ---- Minimal HF generation shims (no repo edits) ----
    # a) Allow transformers.generate() to proceed
    gen_model.can_generate = MethodType(lambda self: True, gen_model)

    # b) Provide a GenerationConfig and disable KV-cache to avoid plumbing _supports_cache_class
    from transformers import GenerationConfig
    gen_model.generation_config = GenerationConfig(
        pad_token_id=config.pad_token_idx,
        eos_token_id=1,
        use_cache=False,
    )

    # c) Satisfy internal capability checks
    gen_model._supports_cache_class = True
    gen_model.device = device

    # d) Ensure prepare_inputs_for_generation returns tensors on the model device
    _orig_pifg = gen_model.prepare_inputs_for_generation
    def _pifg(self, input_ids: torch.LongTensor, **kwargs):
        out = _orig_pifg(input_ids, **kwargs)
        out["input_ids"] = out["input_ids"].to(self.device)
        if "neighbour_ids" in out:
            out["neighbour_ids"] = out["neighbour_ids"].to(self.device)
        return out
    gen_model.prepare_inputs_for_generation = MethodType(_pifg, gen_model)
    # -----------------------------------------------------

    # Tokenize prompt
    input_ids = tokenizer([args.prompt], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    # Generate once
    print("[probe] Running a single generate() call...")
    with torch.no_grad():
        out = gen_model.generate(
            inputs=input_ids,
            do_sample=False,          # beamless deterministic for test
            num_beams=1,
            temperature=1.0,
            min_length=10,
            max_length=128,
            length_penalty=1.0,
            early_stopping=False,
            num_beam_groups=1,
            num_return_sequences=1,
            repetition_penalty=1.0,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=0,
            diversity_penalty=0.0,
            remove_invalid_values=False,
            pad_token_id=config.pad_token_idx,
            eos_token_id=1,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print("\n=== PROBE GENERATION ===")
    print(decoded)
    print("========================\n")
    print("[probe] Success: generate() ran end-to-end.")


if __name__ == "__main__":
    main()

