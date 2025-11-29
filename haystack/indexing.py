"""
Large-scale FAISS Flat indexer tuned for multi-hundred-GB corpora (e.g. 300 GB).
 
Key differences vs `faisss_optimized_disk.py`:
- Focuses solely on *index building* (retrieval intentionally omitted).
- Adds asynchronous FAISS ingestion so embedding and index writes overlap.
- Introduces memory budget enforcement and adaptive flushing for long runs.
- Provides persistent checkpoints, resume support, and manifest reuse.
- Exposes validation helpers to inspect docstore / index consistency.
 
The actual FAISS index remains `IndexFlatIP` (exact) wrapped by `IndexIDMap2`
so that retrieval stays the bottleneck for analysis/benchmarking purposes.
"""
 
from __future__ import annotations
 
import argparse
import gc
import io
import json
import os
import queue
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple
 
try:  # Fast gzip if available
    from isal import igzip as gzip  # type: ignore
except Exception:  # pragma: no cover
    import gzip  # type: ignore
 
import faiss  # type: ignore
import numpy as np
import onnxruntime as ort  # type: ignore
import psutil
from sentence_transformers import SentenceTransformer  # type: ignore
from tqdm import tqdm
 
# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
 
 
def setup_cpu_env(omp_threads: int = 64) -> None:
    """Configure OpenMP / allocator knobs once per process."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    try:
        faiss.omp_set_num_threads(int(omp_threads))
    except Exception:  # pragma: no cover
        pass
 
 
# ---------------------------------------------------------------------------
# DocStore: SQLite + append-only JSONL shards
# ---------------------------------------------------------------------------
 
 
class DocStore:
    def __init__(self, root: Path, start_id_offset: int = 0):
        self.root = Path(root)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        self.db = self.root / "docs.sqlite3"
        self._conn = sqlite3.connect(self.db)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS docs(
              id INTEGER PRIMARY KEY,
              path TEXT NOT NULL,
              offset INTEGER NOT NULL,
              length INTEGER NOT NULL
            );
            """
        )
        self._conn.commit()
        self._fh = None
        self._shard = None
        row = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM docs").fetchone()
        self.next_id = max(int(row[0]) + 1, int(start_id_offset) + 1)
        self._open_new_shard()
 
    # ------------------------------
    def _open_new_shard(self) -> None:
        idx = (self.next_id // 200_000) + 1
        name = f"batch_{idx:06d}.jsonl"
        self._shard = str(self.root / "data" / name)
        self._fh = open(self._shard, "ab", buffering=0)
 
    # ------------------------------
    def add_docs(self, docs: List[dict]) -> np.ndarray:
        cur = self._conn.cursor()
        ids = np.empty(len(docs), dtype=np.int64)
        for i, d in enumerate(docs):
            rec = (
                json.dumps({"content": d["content"], "meta": d.get("meta", {})}, ensure_ascii=False)
                .encode("utf-8")
                + b"\n"
            )
            off = self._fh.tell()
            self._fh.write(rec)
            ln = len(rec)
            doc_id = self.next_id
            self.next_id += 1
            ids[i] = doc_id
            cur.execute(
                "INSERT INTO docs(id, path, offset, length) VALUES(?,?,?,?)",
                (int(doc_id), self._shard, off, ln),
            )
            if (doc_id % 200_000) == 0:
                self._fh.flush()
                self._fh.close()
                self._open_new_shard()
        self._conn.commit()
        return ids
 
    # ------------------------------
    def count_docs(self) -> int:
        (val,) = self._conn.execute("SELECT COUNT(*) FROM docs").fetchone()
        return int(val)
 
    # ------------------------------
    def close(self) -> None:
        try:
            if self._fh:
                self._fh.flush()
                self._fh.close()
        finally:
            self._conn.close()
 
 
# ---------------------------------------------------------------------------
# FAISS Flat index wrapper with async ingestion
# ---------------------------------------------------------------------------
 
 
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
            print(f"📂 Loaded existing FAISS Flat index: ntotal={self.index.ntotal}")
        else:
            base = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIDMap2(base)
            print("🔧 Created new FAISS IndexFlatIP (exact) wrapped in IDMap2")
 
    def save(self) -> None:
        faiss.write_index(self.index, str(self.path))
        print(f"💾 Saved index to {self.path} (ntotal={self.index.ntotal})")
 
    def add_with_ids(self, xb: np.ndarray, ids: np.ndarray) -> None:
        assert xb.dtype == np.float32 and ids.dtype == np.int64
        faiss.normalize_L2(xb)
        self.index.add_with_ids(xb, ids)
 
 
class AsyncFaissWriter:
    """Single consumer thread that feeds FAISS to overlap with embedding."""
 
    def __init__(self, exact_faiss: ExactFaiss, max_queue: int = 2):
        self.fa = exact_faiss
        self.queue: "queue.Queue[Optional[Tuple[np.ndarray, np.ndarray]]]" = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._exception: Optional[BaseException] = None
        self._ntotal = self.fa.index.ntotal
        self._thread.start()
 
    # ------------------------------
    def submit(self, xb: np.ndarray, ids: np.ndarray) -> None:
        self._raise_if_error()
        with self._pending_lock:
            self._pending += 1
        self.queue.put((xb, ids))
 
    # ------------------------------
    def flush(self) -> None:
        """Block until all submitted batches are committed."""
        self.queue.join()
        self._raise_if_error()
 
    # ------------------------------
    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.queue.put(None)  # Sentinel
            self.queue.join()
            self._thread.join()
            self._raise_if_error()
 
    # ------------------------------
    def pending(self) -> int:
        with self._pending_lock:
            return self._pending
 
    # ------------------------------
    @property
    def ntotal(self) -> int:
        return int(self._ntotal)
 
    # ------------------------------
    def _raise_if_error(self) -> None:
        if self._exception:
            raise RuntimeError("FAISS ingestion thread failed") from self._exception
 
    # ------------------------------
    def _drain(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                xb, ids = item
                self.fa.add_with_ids(xb, ids)
                with self._pending_lock:
                    self._pending -= 1
                self._ntotal += len(ids)
            except BaseException as exc:  # pragma: no cover
                self._exception = exc
            finally:
                del item
                gc.collect()
                self.queue.task_done()
 
 
# ---------------------------------------------------------------------------
# Embedding backend (SentenceTransformers with ONNX runtime by default)
# ---------------------------------------------------------------------------
 
 
class STEmbedder:
    def __init__(
        self,
        name: str,
        backend: str = "onnx",
        provider: str = "CPUExecutionProvider",
        onnx_file: Optional[str] = None,
        truncate_dim: Optional[int] = None,
        ort_intra: int = 8,
        ort_inter: int = 1,
    ):
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(ort_intra)
        so.inter_op_num_threads = int(ort_inter)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
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
        return (
            self.model.encode(
                texts,
                batch_size=bs,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
        )
 
 
# ---------------------------------------------------------------------------
# Manifest + checkpoint helpers
# ---------------------------------------------------------------------------
 
 
def build_or_load_manifest(data_root: Path, store_dir: Path, use_filelist: bool) -> List[Path]:
    manifest_path = Path(store_dir) / "manifest.txt"
    files: List[Path] = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            files = [Path(x.strip()) for x in f if x.strip()]
        return files
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
        if not self.path.exists():
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)
 
    def save(self, state: dict) -> None:
        tmp = io.StringIO()
        json.dump(state, tmp)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(tmp.getvalue())
 
 
# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------
 
 
def iter_docs_from_file(
    path: Path,
    min_text_len: int = 100,
    max_docs: Optional[int] = None,
    skip_lines: int = 0,
) -> Iterator[dict]:
    open_fn = gzip.open if str(path).endswith(".gz") else open
    cnt = 0
    with open_fn(path, "rt", encoding="utf-8", errors="ignore") as f:
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
            meta = {
                "source_file": path.name,
                "url": rec.get("url", ""),
                "timestamp": rec.get("timestamp", ""),
            }
            yield {"content": text, "meta": meta}
            cnt += 1
            if max_docs and cnt >= max_docs:
                break
 
 
@dataclass
class BatchProgress:
    file_idx: int
    last_line: int
    added_docs: int
 
 
# ---------------------------------------------------------------------------
# Large-scale Indexer
# ---------------------------------------------------------------------------
 
 
class LargeFlatIndexer:
    def __init__(
        self,
        store_dir: Path,
        model_name: str,
        omp_threads: int = 64,
        backend: str = "onnx",
        provider: str = "CPUExecutionProvider",
        onnx_file: Optional[str] = None,
        truncate_dim: Optional[int] = None,
        embed_batch: int = 512,
        mp_workers: int = 16,
        mp_chunk: int = 6000,
        ort_intra: int = 8,
        ort_inter: int = 1,
        start_id_offset: int = 0,
    ):
        setup_cpu_env(omp_threads)
        self.store_dir = Path(store_dir)
        self.ckpt = Checkpoint(self.store_dir / "checkpoint.json")
        self.docstore = DocStore(self.store_dir / "docstore", start_id_offset=start_id_offset)
        self.embedder = STEmbedder(
            model_name,
            backend=backend,
            provider=provider,
            onnx_file=onnx_file,
            truncate_dim=truncate_dim,
            ort_intra=ort_intra,
            ort_inter=ort_inter,
        )
        self.fa = ExactFaiss(dim=self.embedder.dim, workdir=self.store_dir / "faiss")
        self.embed_bs = int(embed_batch)
        self.mp_workers = int(mp_workers)
        self.mp_chunk = int(mp_chunk)
        self._pool = None
        if self.mp_workers > 0:
            self._pool = self.embedder.model.start_multi_process_pool(target_devices=["cpu"] * self.mp_workers)
        self._proc = psutil.Process(os.getpid())
 
    # ------------------------------
    def _encode_block(self, texts: List[str]) -> np.ndarray:
        if self._pool:
            xb = self.embedder.model.encode(
                texts,
                pool=self._pool,
                batch_size=self.embed_bs,
                chunk_size=self.mp_chunk,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return xb.astype("float32", copy=False)
        parts = []
        for i in range(0, len(texts), self.embed_bs):
            parts.append(self.embedder.encode(texts[i : i + self.embed_bs], bs=self.embed_bs))
        return np.concatenate(parts, axis=0)
 
    # ------------------------------
    def index_stream(
        self,
        files: List[Path],
        docs_per_flush: int = 30_000,
        save_every_docs: int = 200_000,
    resume: bool = False,
    mem_budget_gb: Optional[float] = None,
    heartbeat_secs: Optional[float] = None,
    min_text_len: int = 100,
    async_queue: int = 2,
    ) -> None:
        state = self.ckpt.load() if resume else None
        file_idx = int(state.get("file_idx", 0)) if state else 0
        line_idx = int(state.get("line_idx", 0)) if state else 0
        print(
            f"[index] resume={resume} | start file={file_idx} line={line_idx} | existing ntotal={self.fa.index.ntotal}"
        )
 
        writer = AsyncFaissWriter(self.fa, max_queue=async_queue)
        pbar = tqdm(desc="Indexing (Flat, async)", unit="docs", initial=self.fa.index.ntotal)
        last_save_total = self.fa.index.ntotal
        last_heartbeat = time.time()
 
        texts: List[str] = []
        metas: List[dict] = []
 
        def flush_batch(current_file: int, current_line: int) -> BatchProgress:
            if not texts:
                return BatchProgress(current_file, current_line, 0)
            xb = self._encode_block(texts)
            docs = [{"content": c, "meta": m} for c, m in zip(texts, metas)]
            ids = self.docstore.add_docs(docs)
            writer.submit(xb, ids)
            added = len(ids)
            pbar.update(added)
            texts.clear()
            metas.clear()
            gc.collect()
            return BatchProgress(current_file, current_line, added)
 
        try:
            for fi in range(file_idx, len(files)):
                path = files[fi]
                skip = line_idx if fi == file_idx else 0
                consumed = skip
                for rec in iter_docs_from_file(path, min_text_len=min_text_len, skip_lines=skip):
                    texts.append(rec["content"])
                    metas.append(rec.get("meta", {}))
                    consumed += 1
                    if len(texts) >= docs_per_flush:
                        prog = flush_batch(fi, consumed)
                        if (writer.ntotal - last_save_total) >= save_every_docs:
                            writer.flush()
                            self.fa.save()
                            last_save_total = writer.ntotal
                            self.ckpt.save(
                                {
                                    "file_idx": prog.file_idx,
                                    "line_idx": prog.last_line,
                                    "faiss_ntotal": writer.ntotal,
                                    "docstore_next_id": int(self.docstore.next_id),
                                    "time": time.time(),
                                }
                            )
                        if mem_budget_gb:
                            rss = self._proc.memory_info().rss / (1024 ** 3)
                            if rss > mem_budget_gb:
                                print(f"[mem] RSS {rss:.2f} GB > budget {mem_budget_gb:.2f} GB → flushing queue")
                                writer.flush()
                                gc.collect()
                        if heartbeat_secs and (time.time() - last_heartbeat) >= heartbeat_secs:
                            rss = self._proc.memory_info().rss / (1024 ** 3)
                            print(
                                f"[hb] file={fi} | line={consumed} | ntotal={writer.ntotal} | RSS={rss:.2f} GB",
                                file=sys.stderr,
                                flush=True,
                            )
                            last_heartbeat = time.time()
                # End of file → flush remaining docs for that file
                prog = flush_batch(fi, consumed)
                writer.flush()
                self.fa.save()
                last_save_total = writer.ntotal
                self.ckpt.save(
                    {
                        "file_idx": fi + 1,
                        "line_idx": 0,
                        "faiss_ntotal": writer.ntotal,
                        "docstore_next_id": int(self.docstore.next_id),
                        "time": time.time(),
                    }
                )
                line_idx = 0
        finally:
            try:
                writer.close()
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
 
    # ------------------------------
    def close(self) -> None:
        if self._pool:
            try:
                self.embedder.model.stop_multi_process_pool(self._pool)
            except Exception:
                pass
        self.docstore.close()
 
 
# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
 
 
def inspect_store(store_dir: Path, sample: int = 3) -> None:
    docstore = DocStore(Path(store_dir) / "docstore")
    total = docstore.count_docs()
    print(f"DocStore rows: {total}")
    cursor = docstore._conn.execute("SELECT id, path, offset, length FROM docs ORDER BY id LIMIT ?", (sample,))
    for row in cursor.fetchall():
        doc_id, path, off, ln = row
        with open(path, "rb") as fh:
            fh.seek(off)
            raw = fh.read(ln)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {"error": "decode failed"}
        print(f"- id={doc_id} | file={Path(path).name} | bytes={ln}")
        print(f"  meta keys: {list(payload.get('meta', {}).keys())}")
    docstore.close()
 
 
def index_stats(store_dir: Path) -> None:
    fa = ExactFaiss(dim=1, workdir=Path(store_dir) / "faiss")  # dim ignored on load
    print(f"FAISS index path: {fa.path}")
    print(f"Index ntotal: {fa.index.ntotal}")
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
 
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("FAISS Flat large-scale indexer (async)")
    sub = ap.add_subparsers(dest="cmd", required=True)
 
    p_idx = sub.add_parser("index", help="Build / resume the flat index")
    p_idx.add_argument("--data-root", type=str, required=True, help="Directory walk or newline-delimited file list")
    p_idx.add_argument("--filelist", action="store_true", help="Interpret --data-root as a manifest file")
    p_idx.add_argument("--store-dir", type=str, default="./rag_flat_store")
    p_idx.add_argument("--model", type=str, default="sentence-transformers/static-retrieval-mrl-en-v1")
    p_idx.add_argument("--backend", type=str, default="onnx", choices=["onnx", "torch", "openvino"])
    p_idx.add_argument("--provider", type=str, default="CPUExecutionProvider")
    p_idx.add_argument("--onnx-file", type=str, default=None)
    p_idx.add_argument("--truncate-dim", type=int, default=None)
    p_idx.add_argument("--embed-batch", type=int, default=512)
    p_idx.add_argument("--mp-workers", type=int, default=16)
    p_idx.add_argument("--mp-chunk", type=int, default=6000)
    p_idx.add_argument("--ort-intra", type=int, default=8)
    p_idx.add_argument("--ort-inter", type=int, default=1)
    p_idx.add_argument("--omp-threads", type=int, default=64)
    p_idx.add_argument("--docs-per-flush", type=int, default=30_000)
    p_idx.add_argument("--save-every-docs", type=int, default=200_000)
    p_idx.add_argument("--mem-budget-gb", type=float, default=None, help="Flush queue once RSS surpasses this budget")
    p_idx.add_argument("--heartbeat-secs", type=float, default=None, help="Emit stderr heartbeats at this cadence")
    p_idx.add_argument("--min-text-len", type=int, default=100)
    p_idx.add_argument("--async-queue", type=int, default=2, help="Queue size for async FAISS ingestion")
    p_idx.add_argument("--start-id-offset", type=int, default=0, help="Ensure unique IDs across shards")
    p_idx.add_argument("--resume", action="store_true", help="Resume from store_dir/checkpoint.json")
 
    p_stats = sub.add_parser("stats", help="Inspect docstore/index counts")
    p_stats.add_argument("--store-dir", type=str, default="./rag_flat_store")
    p_stats.add_argument("--sample", type=int, default=3)
 
    return ap
 
 
def cmd_index(args: argparse.Namespace) -> None:
    files = build_or_load_manifest(Path(args.data_root), Path(args.store_dir), use_filelist=args.filelist)
    indexer = LargeFlatIndexer(
        store_dir=Path(args.store_dir),
        model_name=args.model,
        omp_threads=args.omp_threads,
        backend=args.backend,
        provider=args.provider,
        onnx_file=args.onnx_file,
        truncate_dim=args.truncate_dim,
        embed_batch=args.embed_batch,
        mp_workers=args.mp_workers,
        mp_chunk=args.mp_chunk,
        ort_intra=args.ort_intra,
        ort_inter=args.ort_inter,
        start_id_offset=args.start_id_offset,
    )
    try:
        indexer.index_stream(
            files=files,
            docs_per_flush=args.docs_per_flush,
            save_every_docs=args.save_every_docs,
            resume=args.resume,
            mem_budget_gb=args.mem_budget_gb,
            heartbeat_secs=args.heartbeat_secs,
            min_text_len=args.min_text_len,
            async_queue=args.async_queue,
        )
    finally:
        indexer.close()
 
 
def cmd_stats(args: argparse.Namespace) -> None:
    inspect_store(Path(args.store_dir), sample=args.sample)
    index_stats(Path(args.store_dir))
 
 
def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.cmd == "index":
        cmd_index(args)
    elif args.cmd == "stats":
        cmd_stats(args)
    else:  # pragma: no cover
        raise SystemExit(f"Unknown command: {args.cmd}")
 
 
if __name__ == "__main__":
    main()
 
 