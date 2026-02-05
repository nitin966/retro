#!/usr/bin/env python3
import os, json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import faiss
from concurrent.futures import ThreadPoolExecutor

def _load_shard(path: Path):
    # mmap read-only; returns np.ndarray
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    # ensure 2D numeric
    if arr.ndim != 2:
        raise ValueError(f"{path} must be 2D, got {arr.shape}")
    # avoid copies unless needed
    if arr.dtype == np.float32 and arr.flags['C_CONTIGUOUS']:
        return arr
    # cast minimally; faiss expects float32 training data
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))

def main(args):
    # Use all cores for FAISS
    try:
        faiss.omp_set_num_threads(os.cpu_count() or 1)
    except Exception:
        pass

    spec = json.load(args.spec.open("r"))
    base_dir = args.spec.parent

    # spec can be a list of shard dicts or {"shards": [...]}
    shards = spec if isinstance(spec, list) else spec.get("shards", [])
    if not shards:
        raise ValueError("Spec must be a list or contain key 'shards' with entries having 'embeddings'.")

    # First pass: compute total to load (respect cap)
    sizes = []
    dim = None
    total = 0
    cap = int(args.max_training_vectors) if np.isfinite(args.max_training_vectors) else np.inf
    for s in shards:
        p = base_dir / s["embeddings"]
        # Peek shape/dtype cheaply
        arr = np.load(p, mmap_mode="r", allow_pickle=False)
        if dim is None:
            dim = arr.shape[1]
        elif arr.shape[1] != dim:
            raise ValueError(f"Dim mismatch: {p} has {arr.shape[1]}, expected {dim}")
        n = arr.shape[0]
        sizes.append((p, n))
        if total + n >= cap:
            total = int(min(cap, total + n))
            break
        total += n
    if dim is None or total == 0:
        raise ValueError("No training vectors found.")

    # Allocate final buffer
    embeddings = np.empty((total, dim), dtype=np.float32)

    # Second pass: load+copy with prefetch overlapping
    idx = 0
    desc = "Loading embeddings"
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = None
        for i, (path, n) in enumerate(tqdm(sizes, desc=desc)):
            # Prefetch next shard while we copy this one
            if fut is None:
                fut = ex.submit(_load_shard, path)
            curr = fut.result()
            # Launch prefetch for next (if any)
            fut = ex.submit(_load_shard, sizes[i+1][0]) if i+1 < len(sizes) else None

            take = min(n, embeddings.shape[0] - idx)
            if take <= 0:
                break
            # If dtype already float32+C-contiguous, copy is fast; else already casted in _load_shard
            embeddings[idx:idx+take, :] = curr[:take]
            idx += take

            if idx >= embeddings.shape[0]:
                break

    # Create & (optionally) move to GPUs (same as your logic)
    index = faiss.index_factory(embeddings.shape[1], args.index_type, faiss.METRIC_L2)
    if args.use_gpus:
        co = faiss.GpuMultipleClonerOptions()
        index = faiss.index_cpu_to_all_gpus(index, co)

    # Train
    print(f"Training index on {embeddings.shape[0]} vectors of dim {embeddings.shape[1]} ...")
    index.train(embeddings)

    # Save
    print("Saving index...")
    if args.use_gpus:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, args.output)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--index-type", required=True)
    parser.add_argument("--use-gpus", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-training-vectors", type=float, default=np.inf)
    args = parser.parse_args()
    main(args)

