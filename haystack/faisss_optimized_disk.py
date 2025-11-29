"""
Exact RAG indexer with multi-process ONNX embeddings + RESUME support.
 
- Builds a manifest of input files (persisted).
- Writes a robust checkpoint with file index, in-file line offset, FAISS ntotal, DocStore next_id, etc.
- On --resume, reloads FAISS IndexFlatIP+IDMap2 and DocStore, then fast-forwards to the checkpoint position.
- Embedding: SentenceTransformers (backend=onnx by default) with multi-process pool.
- Exact indexing/search: FAISS IndexFlatIP wrapped by IndexIDMap2 (add_with_ids).
"""
import timeit
import os
import sys
import gc
import io
import json
import time
import sqlite3
import argparse
from pathlib import Path
from typing import Iterator, Optional, List, Tuple, Dict
 
# Fast gzip if available
try:
    from isal import igzip as gzip  # pip install isal
except Exception:  # pragma: no cover
    import gzip  # type: ignore
 
import numpy as np
import psutil
from tqdm import tqdm
 
import faiss  # pip install faiss-cpu
import onnxruntime as ort  # pip install onnxruntime
from sentence_transformers import SentenceTransformer
 
# ----------------------------
# Env / threading (FAISS uses OpenMP)
# ----------------------------
 
def setup_cpu_env(omp_threads: int = 64):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    try:
        faiss.omp_set_num_threads(int(omp_threads))
    except Exception:
        pass
 
# ----------------------------
# DocStore (SQLite: id -> (path, offset, length))
# ----------------------------
 
class DocStore:
    def __init__(self, root: Path, start_id_offset: int = 0):
        self.root = Path(root)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        self.db = self.root / "docs.sqlite3"
        self._conn = sqlite3.connect(self.db)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS docs(
              id INTEGER PRIMARY KEY,
              path TEXT NOT NULL,
              offset INTEGER NOT NULL,
              length INTEGER NOT NULL
            );
        """)
        self._conn.commit()
        self._fh = None
        self._shard = None
        row = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM docs").fetchone()
        self.next_id = max(int(row[0]) + 1, int(start_id_offset) + 1)
        self._open_new_shard()
 
    def _open_new_shard(self):
        idx = (self.next_id // 200_000) + 1
        name = f"batch_{idx:06d}.jsonl"
        self._shard = str(self.root / "data" / name)
        self._fh = open(self._shard, "ab", buffering=0)
 
    def add_docs(self, docs: List[dict]) -> np.ndarray:
        cur = self._conn.cursor()
        ids = np.empty(len(docs), dtype=np.int64)
        for i, d in enumerate(docs):
            rec = json.dumps({"content": d["content"], "meta": d.get("meta", {})},
                             ensure_ascii=False).encode("utf-8") + b"\n"
            off = self._fh.tell()
            self._fh.write(rec)
            ln = len(rec)
            doc_id = self.next_id
            self.next_id += 1
            ids[i] = doc_id
            cur.execute("INSERT INTO docs(id, path, offset, length) VALUES(?,?,?,?)",
                        (int(doc_id), self._shard, off, ln))
            if (doc_id % 200_000) == 0:
                self._fh.flush()
                self._fh.close()
                self._open_new_shard()
        self._conn.commit()
        return ids
 
    def fetch(self, ids: List[int]) -> List[dict]:
        if not ids:
            return []
        q = "SELECT id, path, offset, length FROM docs WHERE id IN (%s)" % ",".join("?" * len(ids))
        rows = self._conn.execute(q, [int(i) for i in ids]).fetchall()
        by_path = {}
        for _id, path, off, ln in rows:
            by_path.setdefault(path, []).append((_id, off, ln))
        out = {}
        for path, items in by_path.items():
            with open(path, "rb") as fh:
                for _id, off, ln in items:
                    fh.seek(off)
                    raw = fh.read(ln)
                    out[_id] = json.loads(raw.decode("utf-8"))
        return [out[i] for i in ids if i in out]
 
    def close(self):
        try:
            if self._fh:
                self._fh.flush(); self._fh.close()
        finally:
            self._conn.close()
 
# ----------------------------
# FAISS: Exact Flat IP + IDMap2
# ----------------------------
 
class ExactFaiss:
    def __init__(self, dim: int, workdir: Path):
        self.dim = dim
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "flat.index"
        if self.path.exists():
            self.index = faiss.read_index(str(self.path))
            if not isinstance(self.index, faiss.IndexIDMap2):
                self.index = faiss.IndexIDMap2(self.index)
            print(f"📂 Loaded FAISS Flat index: ntotal={self.index.ntotal}")
        else:
            base = faiss.IndexFlatIP(self.dim)     # exact inner product (Flat) :contentReference[oaicite:2]{index=2}
            self.index = faiss.IndexIDMap2(base)   # 64-bit external IDs + reconstruct :contentReference[oaicite:3]{index=3}
            print("🔧 Created FAISS IndexFlatIP (exact) wrapped in IDMap2")
 
    def save(self):
        faiss.write_index(self.index, str(self.path))
        print(f"💾 Saved index to {self.path} (ntotal={self.index.ntotal})")
 
    def add_with_ids(self, xb: np.ndarray, ids: np.ndarray):
        assert xb.dtype == np.float32 and ids.dtype == np.int64
        faiss.normalize_L2(xb)
        self.index.add_with_ids(xb, ids)  # exact add with user IDs :contentReference[oaicite:4]{index=4}
 
    def search(self, q: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        q = q.astype(np.float32, copy=False).reshape(1, -1)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        ids = [int(x) for x in I[0] if x != -1]
        scores = [float(x) for x in D[0][:len(ids)]]
        return ids, scores
 
# ----------------------------
# Embedder (ONNX / torch / openvino)
# ----------------------------
 
class STEmbedder:
    def __init__(self,
                 name: str,
                 backend: str = "onnx",
                 provider: str = "CPUExecutionProvider",
                 onnx_file: Optional[str] = None,
                 truncate_dim: Optional[int] = None,
                 ort_intra: int = 8,
                 ort_inter: int = 1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(ort_intra)   # ORT threading knobs :contentReference[oaicite:5]{index=5}
        so.inter_op_num_threads = int(ort_inter)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # :contentReference[oaicite:6]{index=6}
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.model = SentenceTransformer(
            name,
            backend=backend,
            truncate_dim=truncate_dim,
            device="cpu",
            model_kwargs={
                "provider": provider,
                "file_name": onnx_file,
                "export": True,
                "session_options": so,
            },
        )
        dim = self.model.encode(["warmup"], convert_to_numpy=True, normalize_embeddings=True).shape[1]
        self._dim = int(dim)
 
    @property
    def dim(self) -> int:
        return self._dim
 
    def encode(self, texts: List[str], bs: int) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
 
# ----------------------------
# Manifest + Checkpoint
# ----------------------------
 
def build_or_load_manifest(data_root: Path, store_dir: Path, use_filelist: bool) -> List[Path]:
    manifest_path = Path(store_dir) / "manifest.txt"
    files: List[Path] = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            files = [Path(x.strip()) for x in f if x.strip()]
        return files
    # build once
    if use_filelist:
        with open(data_root, "r", encoding="utf-8") as f:
            files = [Path(x.strip()) for x in f if x.strip()]
    else:
        for dirpath, _, filenames in os.walk(str(data_root)):
            for fn in filenames:
                if fn.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz")):
                    files.append(Path(dirpath) / fn)
    files.sort()
    (Path(store_dir)).mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for p in files:
            f.write(str(p) + "\n")
    return files
 
class Checkpoint:
    def __init__(self, path: Path):
        self.path = Path(path)
 
    def load(self) -> Optional[dict]:
        if not self.path.exists(): return None
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)
 
    def save(self, state: dict):
        tmp = io.StringIO()
        json.dump(state, tmp)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(tmp.getvalue())
 
# ----------------------------
# Streaming helpers
# ----------------------------
 
def iter_docs_from_file(path: Path,
                        min_text_len: int = 100,
                        max_docs: Optional[int] = None,
                        skip_lines: int = 0) -> Iterator[dict]:
    open_fn = gzip.open if str(path).endswith(".gz") else open
    cnt = 0
    with open_fn(path, "rt", encoding="utf-8", errors="ignore") as f:
        # Fast-forward (for gzip, we *must* read/skip sequentially) :contentReference[oaicite:7]{index=7}
        for _ in range(skip_lines):
            if not f.readline():
                return
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = rec.get("text") or rec.get("content") or ""
            if len(text) < min_text_len:
                continue
            meta = {"source_file": path.name,
                    "url": rec.get("url", ""),
                    "timestamp": rec.get("timestamp", "")}
            yield {"content": text, "meta": meta}
            cnt += 1
            if max_docs and cnt >= max_docs:
                break
 
# ----------------------------
# Indexer with RESUME
# ----------------------------
 
class Indexer:
    def __init__(self,
                 store_dir: Path,
                 model_name: str,
                 omp_threads: int = 64,
                 backend: str = "onnx",
                 provider: str = "CPUExecutionProvider",
                 onnx_file: Optional[str] = None,
                 truncate_dim: Optional[int] = None,
                 mp_workers: int = 16,
                 mp_chunk: int = 6000,
                 embed_batch: int = 512,
                 ort_intra: int = 8,
                 ort_inter: int = 1,
                 start_id_offset: int = 0):
        setup_cpu_env(omp_threads)
        self.store_dir = Path(store_dir)
        self.ckpt = Checkpoint(self.store_dir / "checkpoint.json")
        self.docstore = DocStore(self.store_dir / "docstore", start_id_offset=start_id_offset)
        self.embedder = STEmbedder(model_name,
                                   backend=backend, provider=provider,
                                   onnx_file=onnx_file, truncate_dim=truncate_dim,
                                   ort_intra=ort_intra, ort_inter=ort_inter)
        self.fa = ExactFaiss(dim=self.embedder.dim, workdir=self.store_dir / "faiss")
        self.embed_bs = int(embed_batch)
        self.mp_workers = int(mp_workers)
        self.mp_chunk = int(mp_chunk)
        self.max_ram_gb = psutil.virtual_memory().total / (1024 ** 3) * 0.7
        self._pool = None
        if self.mp_workers > 0:
            self._pool = self.embedder.model.start_multi_process_pool(target_devices=["cpu"] * self.mp_workers)
 
    def _encode_block(self, texts: List[str]) -> np.ndarray:
        if self._pool:
            xb = self.embedder.model.encode(
                texts, pool=self._pool,
                batch_size=self.embed_bs, chunk_size=self.mp_chunk,
                show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            return xb.astype("float32", copy=False)
        parts = []
        for i in range(0, len(texts), self.embed_bs):
            parts.append(self.embedder.encode(texts[i:i+self.embed_bs], bs=self.embed_bs))
        return np.concatenate(parts, axis=0)
 
    def index_stream(self,
                     files: List[Path],
                     docs_per_flush: int = 40_000,
                     save_every: int = 200_000,
                     heartbeat: bool = False,
                     resume: bool = False):
        # Load checkpoint if any
        state = self.ckpt.load() if resume else None
        file_idx = int(state["file_idx"]) if state else 0
        line_idx = int(state["line_idx"]) if state else 0
 
        # Light consistency note: FAISS ntotal/docstore next_id are authoritative :contentReference[oaicite:8]{index=8}
        if state:
            print(f"[resume] file_idx={file_idx}, line_idx={line_idx}, ntotal={state.get('faiss_ntotal')}, next_id={state.get('docstore_next_id')}")
 
        texts, metas = [], []
        pbar = tqdm(desc="Indexing (Exact Flat, RESUME)", unit="docs")
        added = 0
        last_hb = time.time()
 
        def flush_batch():
            nonlocal texts, metas, added
            if not texts:
                return
            xb = self._encode_block(texts)
            docs = [{"content": c, "meta": m} for c, m in zip(texts, metas)]
            ids = self.docstore.add_docs(docs)
            self.fa.add_with_ids(xb, ids)
            added += len(ids)
            pbar.update(len(ids))
            # checkpoint index file periodically
            if (self.fa.index.ntotal % save_every) < len(ids):
                self.fa.save()
            texts.clear(); metas.clear()
            del xb, ids, docs
            gc.collect()
 
        try:
            # Iterate files with resume offsets
            for fi in range(file_idx, len(files)):
                path = files[fi]
                skip = line_idx if fi == file_idx else 0
                consumed_in_file = skip
                for rec in iter_docs_from_file(path, skip_lines=skip):
                    texts.append(rec["content"]); metas.append(rec.get("meta", {}))
                    consumed_in_file += 1
                    if len(texts) >= docs_per_flush:
                        flush_batch()
                        # Save resume point
                        self.ckpt.save({
                            "file_idx": fi,
                            "line_idx": consumed_in_file,
                            "faiss_ntotal": int(self.fa.index.ntotal),
                            "docstore_next_id": int(self.docstore.next_id),
                            "model": str(self.embedder.model.__class__.__name__),
                            "dim": int(self.embedder.dim),
                            "time": time.time()
                        })
                    if heartbeat and (time.time() - last_hb) >= 2.0:
                        rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
                        print(f"[hb] RSS={rss:.2f} GB | ntotal={self.fa.index.ntotal}", file=sys.stderr, flush=True)
                        last_hb = time.time()
                    used = psutil.virtual_memory().used / (1024 ** 3)
                    if used > self.max_ram_gb:
                        time.sleep(0.05)
                # end of file -> persist checkpoint at boundary
                flush_batch()
                self.ckpt.save({
                    "file_idx": fi + 1,  # next file
                    "line_idx": 0,
                    "faiss_ntotal": int(self.fa.index.ntotal),
                    "docstore_next_id": int(self.docstore.next_id),
                    "model": str(self.embedder.model.__class__.__name__),
                    "dim": int(self.embedder.dim),
                    "time": time.time()
                })
                # reset line_idx for next file
                line_idx = 0
 
        finally:
            if self._pool:
                try:
                    self.embedder.model.stop_multi_process_pool(self._pool)
                except Exception:
                    pass
            pbar.close()
            self.fa.save()
            self.docstore.close()
            print(f"✅ Done: ntotal={self.fa.index.ntotal}")
 
# ----------------------------
# Retrieval
# ----------------------------
 
class Retriever:
    def __init__(self,
                 store_dir: Path,
                 model_name: str,
                 omp_threads: int = 64,
                 backend: str = "onnx",
                 provider: str = "CPUExecutionProvider",
                 onnx_file: Optional[str] = None,
                 truncate_dim: Optional[int] = None,
                 ort_intra: int = 8,
                 ort_inter: int = 1):
        setup_cpu_env(omp_threads)
        self.docstore = DocStore(Path(store_dir) / "docstore")
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(ort_intra)
        so.inter_op_num_threads = int(ort_inter)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.model = SentenceTransformer(
            model_name,
            backend=backend,
            truncate_dim=truncate_dim,
            device="cpu",
            model_kwargs={"provider": provider, "file_name": onnx_file, "export": True, "session_options": so},
        )
        dim = int(self.model.encode(["warmup"], convert_to_numpy=True, normalize_embeddings=True).shape[1])
        self.fa = ExactFaiss(dim=dim, workdir=Path(store_dir) / "faiss")
 
    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        q = self.model.encode([query], batch_size=1, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
        ids, scores = self.fa.search(q, top_k)
        docs = self.docstore.fetch(ids)
        out = []
        for d, s in zip(docs, scores):
            d2 = dict(d); d2["score"] = s
            out.append(d2)
        return out
 
def build_prompt(question: str, docs: List[dict], max_chars_per_doc=500) -> str:
    ctx = []
    for i, d in enumerate(docs, 1):
        text = d.get("content", "")
        snippet = (text[:max_chars_per_doc] + "...") if len(text) > max_chars_per_doc else text
        ctx.append(f"Document {i}:\n{snippet}")
    context = "\n\n".join(ctx) if ctx else "(no context retrieved)"
    return f"""You are a helpful assistant. Answer ONLY using the context below.
If the answer is not in the context, say you don't know.
 
Context:
{context}
 
Question: {question}
 
Answer:"""
 
# ----------------------------
# CLI
# ----------------------------
 
def main():
    ap = argparse.ArgumentParser("Exact RAG (FAISS Flat + MP-ONNX + RESUME)")
    sub = ap.add_subparsers(dest="cmd", required=True)
 
    p_idx = sub.add_parser("index")
    p_idx.add_argument("--data-root", type=str, required=True,
                       help="Directory to walk OR a newline-delimited filelist")
    p_idx.add_argument("--filelist", action="store_true",
                       help="Interpret --data-root as a file list")
    p_idx.add_argument("--store-dir", type=str, default="./rag_flat_store")
    p_idx.add_argument("--model", type=str, default="sentence-transformers/static-retrieval-mrl-en-v1")
 
    # Embedding/MP
    p_idx.add_argument("--backend", type=str, default="onnx", choices=["onnx", "torch", "openvino"])
    p_idx.add_argument("--provider", type=str, default="CPUExecutionProvider")
    p_idx.add_argument("--onnx-file", type=str, default=None)
    p_idx.add_argument("--truncate-dim", type=int, default=None)
    p_idx.add_argument("--embed-batch", type=int, default=512)
    p_idx.add_argument("--mp-workers", type=int, default=16)
    p_idx.add_argument("--mp-chunk", type=int, default=6000)
    p_idx.add_argument("--ort-intra", type=int, default=8)
    p_idx.add_argument("--ort-inter", type=int, default=1)
 
    # FAISS / streaming
    p_idx.add_argument("--omp-threads", type=int, default=64)
    p_idx.add_argument("--docs-per-flush", type=int, default=40000)
    p_idx.add_argument("--save-every", type=int, default=200000)
    p_idx.add_argument("--heartbeat", action="store_true")
    p_idx.add_argument("--start-id-offset", type=int, default=0,
                       help="For sharded runs: ensure unique IDs")
    p_idx.add_argument("--resume", action="store_true", help="Resume from store_dir/checkpoint.json")
 
    p_q = sub.add_parser("query")
    p_q.add_argument("--store-dir", type=str, default="./rag_flat_store")
    p_q.add_argument("--model", type=str, default="sentence-transformers/static-retrieval-mrl-en-v1")
    p_q.add_argument("--backend", type=str, default="onnx", choices=["onnx", "torch", "openvino"])
    p_q.add_argument("--provider", type=str, default="CPUExecutionProvider")
    p_q.add_argument("--onnx-file", type=str, default=None)
    p_q.add_argument("--truncate-dim", type=int, default=None)
    p_q.add_argument("--omp-threads", type=int, default=64)
    p_q.add_argument("--ort-intra", type=int, default=8)
    p_q.add_argument("--ort-inter", type=int, default=1)
    p_q.add_argument("--top-k", type=int, default=5)
    p_q.add_argument("--question", type=str)
 
    args = ap.parse_args()
 
    if args.cmd == "index":
        files = build_or_load_manifest(Path(args.data_root), Path(args.store_dir), use_filelist=args.filelist)
        idx = Indexer(
            store_dir=Path(args.store_dir),
            model_name=args.model,
            omp_threads=args.omp_threads,
            backend=args.backend,
            provider=args.provider,
            onnx_file=args.onnx_file,
            truncate_dim=args.truncate_dim,
            mp_workers=args.mp_workers,
            mp_chunk=args.mp_chunk,
            embed_batch=args.embed_batch,
            ort_intra=args.ort_intra,
            ort_inter=args.ort_inter,
            start_id_offset=args.start_id_offset,
        )
        idx.index_stream(
            files=files,
            docs_per_flush=args.docs_per_flush,
            save_every=args.save_every,
            heartbeat=args.heartbeat,
            resume=args.resume,
        )
 
    elif args.cmd == "query":
        ret = Retriever(
            store_dir=Path(args.store_dir),
            model_name=args.model,
            omp_threads=args.omp_threads,
            backend=args.backend,
            provider=args.provider,
            onnx_file=args.onnx_file,
            truncate_dim=args.truncate_dim,
            ort_intra=args.ort_intra,
            ort_inter=args.ort_inter,
        )
        t1 = timeit.default_timer()
        docs = ret.retrieve(args.question, top_k=args.top_k)
        t2 = timeit.default_timer()
        print(t2-t1)
        prompt = build_prompt(args.question, docs)
        print("\n==== RAG PROMPT ====\n")
        print(prompt)
        print("\n====================\n")
        for i, d in enumerate(docs, 1):
            print(f"[{i}] score={d['score']:.4f} | {d['meta'].get('source_file','')}")
            print(d["content"][:300].replace("\n", " ") + ("..." if len(d["content"]) > 300 else ""))
 
if __name__ == "__main__":
    main()
 
 