#!/usr/bin/env python3
import io, os, zipfile, json
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# Try FAISS, else fall back to hnswlib
_USE_FAISS = False
faiss = None
try:
    import faiss  # type: ignore
    _USE_FAISS = True
except Exception:
    try:
        import hnswlib  # type: ignore
    except Exception:
        hnswlib = None

app = FastAPI()

class Settings(BaseModel):
    # Required: database embeddings matrix (same used to build the index)
    emb_path: Path
    # One of these two:
    faiss_index_path: Optional[Path] = None
    hnsw_index_path: Optional[Path] = None
    # Metric: "ip" (inner product / cosine) or "l2"
    metric: str = "ip"
    # Normalize DB vectors for IP (cosine) search
    normalize_db: bool = True

S: Settings = None  # set in main()
DB: np.ndarray = None
INDEX = None
D: int = 0

def _normalize_L2(X: np.ndarray):
    # in-place L2 normalization
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X /= norms

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.post("/query")
async def query(request: Request, k: int):
    try:
        body = await request.body()
        q = np.load(io.BytesIO(body), allow_pickle=False)
        if q.ndim == 1:
            q = q[None, :]
        if q.shape[1] != D:
            raise HTTPException(status_code=400, detail=f"Bad dim: got {q.shape[1]}, expected {D}")

        q = q.astype(np.float32, copy=False)
        # normalize queries for cosine/IP if requested
        if S.metric.lower() == "ip" and S.normalize_db:
            _normalize_L2(q)

        if _USE_FAISS and S.faiss_index_path is not None:
            # FAISS search
            distances, indices = INDEX.search(q, k)
            # FAISS returns higher==better for IP, lower==better for L2
            # Your client doesn’t care about sign; pass as-is.
        elif S.hnsw_index_path is not None:
            # HNSW search
            labels, dists = INDEX.knn_query(q, k=k)  # cosine or l2 depending on init
            indices = labels.astype(np.int64)
            distances = dists.astype(np.float32)
        else:
            raise HTTPException(status_code=500, detail="No ANN backend available")

        # Return zip: distances, indices, embeddings (for the first query only)
        key_indices = indices
        first_unique = np.unique(indices[0])
        key_embs = DB[first_unique]  # [<=K, D], your client reads it correctly

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("distances", np.asarray(distances, dtype=np.float32).tobytes())
            z.writestr("indices", np.asarray(key_indices, dtype=np.int64).tobytes())
            z.writestr("embeddings", np.asarray(key_embs, dtype=np.float32).tobytes())
        return Response(content=buf.getvalue(), media_type="application/zip")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def load_everything(emb_path: Path, faiss_index_path: Optional[Path], hnsw_index_path: Optional[Path], metric: str, normalize_db: bool):
    global DB, INDEX, D
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    DB = np.load(emb_path, mmap_mode="r")
    if DB.dtype != np.float32:
        DB = DB.astype(np.float32, copy=False)
    D = DB.shape[1]

    if metric.lower() not in ("ip", "l2"):
        raise ValueError("--metric must be 'ip' or 'l2'")

    if metric.lower() == "ip" and normalize_db:
        # normalize a RAM copy for IP/cosine search (use memmap-backed normalization with a view)
        DB = np.array(DB, dtype=np.float32, copy=True)  # materialize
        _normalize_L2(DB)

    if faiss_index_path is not None:
        if not _USE_FAISS:
            raise RuntimeError("FAISS not available but faiss_index_path was provided.")
        if not faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")
        index = faiss.read_index(str(faiss_index_path))
        # For IP/cosine, FAISS likes normalized DB already; we searched normalized queries above
        return DB, index

    if hnsw_index_path is not None:
        if hnswlib is None:
            raise RuntimeError("hnswlib not available but hnsw_index_path was provided.")
        if not hnsw_index_path.exists():
            raise FileNotFoundError(f"HNSW index not found: {hnsw_index_path}")
        space = "cosine" if metric.lower() == "ip" else "l2"
        index = hnswlib.Index(space=space, dim=D)
        index.load_index(str(hnsw_index_path))
        index.set_ef(200)
        return DB, index

    raise ValueError("Provide either faiss_index_path or hnsw_index_path")

def _parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True, help="Path to chunks.embeddings.npy")
    p.add_argument("--faiss-index", help="Path to FAISS .index")
    p.add_argument("--hnsw-index", help="Path to HNSW index")
    p.add_argument("--metric", default="ip", choices=["ip","l2"], help="Index metric (cosine==ip)")
    p.add_argument("--no-normalize-db", action="store_true", help="Disable L2 normalization for IP")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()

def main():
    global S, INDEX
    args = _parse_args()
    S = Settings(
        emb_path=Path(args.embeddings),
        faiss_index_path=Path(args.faiss_index) if args.faiss_index else None,
        hnsw_index_path=Path(args.hnsw_index) if args.hnsw_index else None,
        metric=args.metric,
        normalize_db=(not args.no_normalize_db),
    )
    emb, idx = load_everything(S.emb_path, S.faiss_index_path, S.hnsw_index_path, S.metric, S.normalize_db)
    globals()["DB"] = emb
    globals()["INDEX"] = idx
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()

