#!/usr/bin/env python3
import argparse, io, zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

import faiss
import numpy as np

# Globals loaded at startup
INDEX = None
EMB = None
KDIM = None

def npy_from_bytes(b):
    return np.load(io.BytesIO(b), allow_pickle=False)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
        self.send_error(404, "Not found")

    def do_POST(self):
        if not self.path.startswith("/query"):
            self.send_error(404, "Not found"); return

        # Parse k from URL (default 4)
        k = 4
        try:
            if "k=" in self.path:
                k = int(self.path.split("k=")[1].split("&")[0])
        except Exception:
            pass

        # Read embeddings from request body (.npy bytes)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        Q = npy_from_bytes(body).astype("float32")  # shape [B, D]
        assert Q.ndim == 2 and Q.shape[1] == KDIM, f"Bad query shape {Q.shape}, expected [B,{KDIM}]"

        # Search
        D, I = INDEX.search(Q, k)  # D: [B,k] float32, I: [B,k] int64
        # Gather key embeddings (fill -1 with zeros)
        B = I.shape[0]
        key_embs = np.zeros((B, k, KDIM), dtype="float32")
        mask = (I >= 0)
        if mask.any():
            flat_idx = I[mask]
            key_embs[mask] = EMB[flat_idx]

        # Package results as a zip of .npy files
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            b = io.BytesIO(); np.save(b, D.astype("float32")); zf.writestr("distances", b.getvalue())
            b = io.BytesIO(); np.save(b, I.astype("int64"));   zf.writestr("indices",   b.getvalue())
            b = io.BytesIO(); np.save(b, key_embs);             zf.writestr("embeddings",b.getvalue())

        data = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="FAISS index file (trained IVF*,PQ*)")
    ap.add_argument("--embeddings", required=True, help="chunks.embeddings.npy (float16/32)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--nprobe", type=int, default=512)
    args = ap.parse_args()

    global INDEX, EMB, KDIM
    EMB = np.load(args.embeddings, mmap_mode="r").astype("float32")
    KDIM = EMB.shape[1]
    INDEX = faiss.read_index(args.index)
    # If your index isn't an IndexIDMap, add vectors now if ntotal==0
    if getattr(INDEX, "ntotal", 0) == 0:
        INDEX.add(EMB)
    if hasattr(INDEX, "nprobe"):
        INDEX.nprobe = args.nprobe
    print(f"[faiss-server] loaded: emb={EMB.shape} index.ntotal={INDEX.ntotal} nprobe={getattr(INDEX,'nprobe','n/a')}")
    HTTPServer((args.host, args.port), Handler).serve_forever()

if __name__ == "__main__":
    main()

