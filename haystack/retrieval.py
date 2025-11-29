"""Large-scale FAISS Flat disk retriever with Haystack RAG integration.
 
This module includes all base retrieval code directly and adds RAG (Retrieval-Augmented Generation)
capabilities using Haystack components and an LLM model.
 
Key features:
* Complete standalone implementation - no external retrieval imports needed
* Haystack pipeline integration for RAG workflows
* LLM-based answer generation using retrieved documents
* Flexible LLM backend support (OpenAI API compatible endpoints)
* Detailed timing breakdowns for retrieval and generation phases
* All optimizations from the base retriever preserved
"""
 
from __future__ import annotations
 
import argparse
import io
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import mmap
 
import numpy as np
import faiss
import requests
 
from haystack import Document as HaystackDocument
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
 
from faisss_optimized_disk import (
    ExactFaiss,
    STEmbedder,
    build_prompt,
    setup_cpu_env,
)
 
# ---------------------------------------------------------------------------
# Shard cache helpers (copied from base)
# ---------------------------------------------------------------------------
 
 
class ShardCache:
    """LRU cache that keeps JSONL shards open and optionally memory-mapped."""
 
    def __init__(self, max_open: int = 16, use_mmap: bool = True):
        self.max_open = max_open
        self.use_mmap = use_mmap
        self._fds: "OrderedDict[str, Tuple[io.BufferedReader, Optional[mmap.mmap]]]" = OrderedDict()
        self._lock = threading.Lock()
 
    def _evict_if_needed(self):
        while len(self._fds) > self.max_open:
            path, (fh, mm) = self._fds.popitem(last=False)
            try:
                if mm:
                    mm.close()
            finally:
                fh.close()
 
    def get(self, path: Path):
        key = str(path)
        with self._lock:
            if key in self._fds:
                fh, mm = self._fds.pop(key)
                self._fds[key] = (fh, mm)
                return fh, mm
            fh = open(path, "rb", buffering=0)
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) if self.use_mmap else None
            self._fds[key] = (fh, mm)
            self._evict_if_needed()
            return fh, mm
 
    def close(self):
        with self._lock:
            for fh, mm in self._fds.values():
                try:
                    if mm:
                        mm.close()
                finally:
                    fh.close()
            self._fds.clear()
 
 
# ---------------------------------------------------------------------------
# Read-only docstore with shard caching (copied from base)
# ---------------------------------------------------------------------------
 
 
class ReadOnlyDocStore:
    def __init__(self, root: Path, shard_cache_size: int = 24, use_mmap: bool = True):
        self.root = Path(root)
        self.db_path = self.root / "docs.sqlite3"
        if not self.db_path.exists():
            raise FileNotFoundError(f"Docstore not found at {self.db_path}")
        uri = f"file:{self.db_path}?mode=ro"
        # check_same_thread=False enables access from executor threads.
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
 
        # Try to configure journal mode - WAL mode is compatible with read-only access
        try:
            # Check current journal mode
            current_mode = self._conn.execute("PRAGMA journal_mode").fetchone()[0]
            # WAL mode is fine for read-only access, don't try to change it
            if current_mode.upper() != 'WAL':
                self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            # Already in WAL mode or can't change - that's fine for read-only
            pass
 
        self._conn.execute("PRAGMA synchronous=OFF")
        self._conn.execute("PRAGMA cache_size=16000")  # ~16MB page cache
        self._lock = threading.Lock()
        self._cache = ShardCache(max_open=shard_cache_size, use_mmap=use_mmap)
 
    def _fetch_chunk(self, ids: Sequence[int]) -> Dict[int, Tuple[str, int, int]]:
        marks = ",".join("?" for _ in ids)
        query = f"SELECT id, path, offset, length FROM docs WHERE id IN ({marks})"
        with self._lock:
            rows = self._conn.execute(query, list(map(int, ids))).fetchall()
        out: Dict[int, Tuple[str, int, int]] = {}
        for r in rows:
            out[int(r["id"])] = (r["path"], int(r["offset"]), int(r["length"]))
        return out
 
    def sample_ids(self, n: int) -> List[int]:
        with self._lock:
            rows = self._conn.execute("SELECT id FROM docs ORDER BY RANDOM() LIMIT ?", (int(n),)).fetchall()
        return [int(r[0]) for r in rows]
 
    def fetch_many(self, ids: Sequence[int], executor: Optional[ThreadPoolExecutor] = None) -> Dict[int, dict]:
        unique_ordered: List[int] = []
        seen = set()
        for _id in ids:
            if _id not in seen:
                unique_ordered.append(int(_id))
                seen.add(int(_id))
        if not unique_ordered:
            return {}
        chunks: List[Sequence[int]] = []
        CHUNK = 800  # keep margin under sqlite default max vars (999)
        for i in range(0, len(unique_ordered), CHUNK):
            chunks.append(unique_ordered[i : i + CHUNK])
 
        meta: Dict[int, Tuple[str, int, int]] = {}
        for chunk in chunks:
            meta.update(self._fetch_chunk(chunk))
 
        def read_one(item_id: int) -> Tuple[int, dict]:
            path, offset, length = meta[item_id]
            fh, mm = self._cache.get(Path(path))
            if mm:
                raw = mm[offset : offset + length]
            else:
                with self._lock:
                    fh.seek(offset)
                    raw = fh.read(length)
            return item_id, json.loads(raw)
 
        results: Dict[int, dict] = {}
        if executor:
            futures = {executor.submit(read_one, doc_id): doc_id for doc_id in unique_ordered if doc_id in meta}
            for fut in futures:
                doc_id, payload = fut.result()
                results[doc_id] = payload
        else:
            for doc_id in unique_ordered:
                if doc_id not in meta:
                    continue
                did, payload = read_one(doc_id)
                results[did] = payload
        return results
 
    def close(self):
        try:
            self._cache.close()
        finally:
            self._conn.close()
 
 
# ---------------------------------------------------------------------------
# Profiling helpers (copied from base)
# ---------------------------------------------------------------------------
 
 
class StageTimer:
    def __init__(self):
        self._events = defaultdict(float)
 
    @contextmanager
    def track(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._events[name] += time.perf_counter() - start
 
    def snapshot(self):
        return dict(self._events)
 
 
# ---------------------------------------------------------------------------
# Large-scale retriever (copied from base)
# ---------------------------------------------------------------------------
 
 
class LargeScaleRetriever:
    def __init__(
        self,
        store_dir: Path,
        model_name: str = "sentence-transformers/static-retrieval-mrl-en-v1",
        backend: str = "onnx",
        provider: str = "CPUExecutionProvider",
        onnx_file: Optional[str] = None,
        truncate_dim: Optional[int] = None,
        omp_threads: int = 64,
        ort_intra: int = 8,
        ort_inter: int = 1,
        embed_batch: int = 128,
        doc_workers: int = 4,
        shard_cache: int = 24,
        use_mmap: bool = True,
    ):
        setup_cpu_env(omp_threads)
        self.store_dir = Path(store_dir)
        self.docstore = ReadOnlyDocStore(self.store_dir / "docstore", shard_cache_size=shard_cache, use_mmap=use_mmap)
        # SentenceTransformer wrapper (uses ONNX by default)
        self.embedder = STEmbedder(
            name=model_name,
            backend=backend,
            provider=provider,
            onnx_file=onnx_file,
            truncate_dim=truncate_dim,
            ort_intra=ort_intra,
            ort_inter=ort_inter,
        )
        self.fa = ExactFaiss(dim=self.embedder.dim, workdir=self.store_dir / "faiss")
        self.embed_batch = int(embed_batch)
        self.doc_workers = int(doc_workers)
        self._executor = ThreadPoolExecutor(max_workers=self.doc_workers) if self.doc_workers > 0 else None
 
    def _encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self.embedder.encode(list(queries), bs=self.embed_batch)
 
    def retrieve_batch(
        self,
        queries: Sequence[str],
        top_k: int = 5,
    ) -> Tuple[List[List[dict]], List[Dict[str, float]]]:
        timer = StageTimer()
        with timer.track("embed"):
            q_vectors = self._encode_queries(queries)
        with timer.track("search"):
            q = q_vectors.astype(np.float32, copy=False)
            faiss.normalize_L2(q)
            # FAISS batch search is already parallelized - don't change thread count
            # Let FAISS use its default OpenMP threads for optimal performance
            D, I = self.fa.index.search(q, top_k)
        # Flatten IDs for one-shot docstore fetch
        doc_ids: List[int] = []
        for arr in I:
            doc_ids.extend([int(x) for x in arr if x != -1])
        with timer.track("doc_fetch"):
            fetched = self.docstore.fetch_many(doc_ids, executor=self._executor)
        results: List[List[dict]] = []
        for q_idx in range(len(queries)):
            docs: List[dict] = []
            seen = set()
            for doc_id, score in zip(I[q_idx], D[q_idx]):
                doc_id = int(doc_id)
                if doc_id == -1 or doc_id in seen or doc_id not in fetched:
                    continue
                payload = dict(fetched[doc_id])
                payload["score"] = float(score)
                docs.append(payload)
                seen.add(doc_id)
            results.append(docs)
        return results, [timer.snapshot() for _ in queries]
 
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[dict], Dict[str, float]]:
        docs, stats = self.retrieve_batch([query], top_k=top_k)
        return docs[0], stats[0]
 
    def close(self):
        try:
            if self._executor:
                self._executor.shutdown(wait=True, cancel_futures=False)
        finally:
            self.docstore.close()
 
 
def _format_stats(stats: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v*1000:.1f}ms" for k, v in stats.items())
 
 
# ---------------------------------------------------------------------------
# RAG Components using Haystack
# ---------------------------------------------------------------------------
 
 
class HaystackRAGGenerator:
    """RAG generator using Haystack components with OpenAI-compatible API endpoints."""
 
    def __init__(
        self,
        api_base_url: str = "http://localhost:5000/v1",
        model_name: str = "openai/gpt-oss-20b",
        api_key: str = "EMPTY",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ):
        """Initialize Haystack RAG generator.
 
        Args:
            api_base_url: Base URL for OpenAI-compatible API endpoint (e.g., vLLM, text-generation-inference)
            model_name: Model name/path to use for generation
            api_key: API key (use "EMPTY" for local endpoints)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.api_base_url = api_base_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
 
        # Test connection to the API endpoint
        try:
            response = requests.get(f"{api_base_url.rstrip('/v1')}/v1/models", timeout=5)
            if response.status_code == 200:
                print(f"✅ Connected to LLM endpoint: {api_base_url}")
            else:
                print(f"⚠️  LLM endpoint connection warning: status {response.status_code}")
        except Exception as e:
            print(f"⚠️  Could not verify LLM endpoint connection: {e}")
 
        # Initialize Haystack components
        self._init_haystack_components(api_key)
 
    def _init_haystack_components(self, api_key: str):
        """Initialize Haystack prompt builder and generator."""
 
        # Prompt template for RAG
        self.prompt_template = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer the question. If the context doesn't contain enough information, say so clearly.
 
Context:
{% for doc in documents %}
Document {{ loop.index }}:
{{ doc.content }}
---
{% endfor %}
 
Question: {{ question }}
 
Please provide a clear, concise answer based only on the context provided above.
 
Answer:"""
 
        # Initialize prompt builder with required variables specified
        self.prompt_builder = PromptBuilder(
            template=self.prompt_template,
            required_variables=["documents", "question"]
        )
 
        # Initialize generator with OpenAI-compatible API
        # Wrap API key in Secret object as required by Haystack
        self.generator = OpenAIGenerator(
            api_key=Secret.from_token(api_key) if isinstance(api_key, str) else api_key,
            model=self.model_name,
            api_base_url=self.api_base_url,
            generation_kwargs={
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        )
 
    def generate_answer(
        self,
        question: str,
        documents: List[dict],
        max_chars_per_doc: int = 500,
    ) -> Dict:
        """Generate answer using retrieved documents.
 
        Args:
            question: User question
            documents: List of retrieved documents (from FAISS retrieval)
            max_chars_per_doc: Maximum characters to include from each document
 
        Returns:
            Dictionary with answer, metadata, and timing information
        """
        start_time = time.perf_counter()
 
        if not documents:
            return {
                "answer": "No relevant documents found to answer this question.",
                "prompt": "",
                "num_documents": 0,
                "generation_time": 0.0,
                "prompt_build_time": 0.0,
                "llm_inference_time": 0.0,
            }
 
        # Convert retrieved documents to Haystack Document format
        conversion_start = time.perf_counter()
        haystack_docs = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")[:max_chars_per_doc]
            meta = doc.get("meta", {})
 
            haystack_docs.append(
                HaystackDocument(
                    content=content,
                    meta={
                        "source_file": meta.get("source_file", ""),
                        "timestamp": meta.get("timestamp", ""),
                        "score": doc.get("score", 0.0),
                    }
                )
            )
        conversion_time = time.perf_counter() - conversion_start
 
        # Build prompt
        prompt_start = time.perf_counter()
        prompt_result = self.prompt_builder.run(
            documents=haystack_docs,
            question=question
        )
        prompt = prompt_result["prompt"]
        prompt_build_time = time.perf_counter() - prompt_start
 
        # Generate answer
        llm_start = time.perf_counter()
        try:
            generation_result = self.generator.run(prompt=prompt)
 
            # Extract answer from generation result
            if "replies" in generation_result and generation_result["replies"]:
                answer = generation_result["replies"][0]
            else:
                answer = "No answer generated by the model."
 
            llm_inference_time = time.perf_counter() - llm_start
            total_generation_time = time.perf_counter() - start_time
 
            return {
                "answer": answer.strip(),
                "prompt": prompt,
                "num_documents": len(haystack_docs),
                "generation_time": total_generation_time,
                "prompt_build_time": prompt_build_time,
                "llm_inference_time": llm_inference_time,
                "doc_conversion_time": conversion_time,
            }
 
        except Exception as e:
            llm_inference_time = time.perf_counter() - llm_start
            total_generation_time = time.perf_counter() - start_time
 
            return {
                "answer": f"Error generating answer: {str(e)}",
                "prompt": prompt,
                "num_documents": len(haystack_docs),
                "generation_time": total_generation_time,
                "prompt_build_time": prompt_build_time,
                "llm_inference_time": llm_inference_time,
                "doc_conversion_time": conversion_time,
            }

class LargeScaleRAGRetriever:
    """Extends LargeScaleRetriever with RAG capabilities using Haystack."""
 
    def __init__(
        self,
        store_dir: Path,
        api_base_url: str = "http://localhost:5000/v1",
        model_name: str = "openai/gpt-oss-20b",
        api_key: str = "EMPTY",
        llm_max_tokens: int = 512,
        llm_temperature: float = 0.1,
        # Retriever parameters (passed to base retriever)
        embedding_model: str = "sentence-transformers/static-retrieval-mrl-en-v1",
        backend: str = "onnx",
        provider: str = "CPUExecutionProvider",
        onnx_file: Optional[str] = None,
        truncate_dim: Optional[int] = None,
        omp_threads: int = 64,
        ort_intra: int = 8,
        ort_inter: int = 1,
        embed_batch: int = 128,
        doc_workers: int = 4,
        shard_cache: int = 24,
        use_mmap: bool = True,
    ):
        """Initialize RAG retriever.
 
        This combines the base retriever (unchanged) with RAG generation capabilities.
        """
        # Initialize base retriever (unchanged from original)
        self.base_retriever = LargeScaleRetriever(
            store_dir=store_dir,
            model_name=embedding_model,
            backend=backend,
            provider=provider,
            onnx_file=onnx_file,
            truncate_dim=truncate_dim,
            omp_threads=omp_threads,
            ort_intra=ort_intra,
            ort_inter=ort_inter,
            embed_batch=embed_batch,
            doc_workers=doc_workers,
            shard_cache=shard_cache,
            use_mmap=use_mmap,
        )
 
        # Initialize RAG generator
        self.rag_generator = HaystackRAGGenerator(
            api_base_url=api_base_url,
            model_name=model_name,
            api_key=api_key,
            max_tokens=llm_max_tokens,
            temperature=llm_temperature,
        )
 
        print(f"🚀 Large-scale RAG retriever initialized with Haystack")
 
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> Tuple[List[dict], Dict[str, float]]:
        """Retrieve documents (uses base retriever, unchanged)."""
        return self.base_retriever.retrieve(query, top_k=top_k)
 
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> Tuple[List[List[dict]], List[Dict[str, float]]]:
        """Retrieve documents in batch (uses base retriever, unchanged)."""
        return self.base_retriever.retrieve_batch(queries, top_k=top_k)
 
    def retrieve_and_generate(
        self,
        question: str,
        top_k: int = 5,
        max_chars_per_doc: int = 500,
    ) -> Dict:
        """Retrieve documents and generate answer using RAG.
 
        Args:
            question: User question
            top_k: Number of documents to retrieve
            max_chars_per_doc: Maximum characters per document for context
 
        Returns:
            Dictionary with answer, retrieved documents, and detailed timing stats
        """
        # Step 1: Retrieve documents using base retriever
        retrieval_start = time.perf_counter()
        docs, retrieval_stats = self.retrieve(question, top_k=top_k)
        retrieval_total_time = time.perf_counter() - retrieval_start
 
        # Step 2: Generate answer using RAG
        generation_start = time.perf_counter()
        rag_result = self.rag_generator.generate_answer(
            question=question,
            documents=docs,
            max_chars_per_doc=max_chars_per_doc,
        )
        generation_total_time = time.perf_counter() - generation_start
 
        # Calculate total time
        total_time = retrieval_total_time + generation_total_time
 
        return {
            "question": question,
            "answer": rag_result["answer"],
            "retrieved_documents": docs,
            "num_documents": len(docs),
            # Retrieval timing (from base retriever)
            "retrieval_stats": retrieval_stats,
            "retrieval_total_time": retrieval_total_time,
            # Generation timing (detailed breakdown)
            "generation_total_time": generation_total_time,
            "prompt_build_time": rag_result.get("prompt_build_time", 0.0),
            "llm_inference_time": rag_result.get("llm_inference_time", 0.0),
            "doc_conversion_time": rag_result.get("doc_conversion_time", 0.0),
            # Overall timing
            "total_time": total_time,
            "prompt": rag_result["prompt"],
        }
 
    def close(self):
        """Clean up resources."""
        self.base_retriever.close()
 
 
# ---------------------------------------------------------------------------
# CLI with RAG support
# ---------------------------------------------------------------------------
 
 
def single_query_rag(args: argparse.Namespace):
    """Single query with RAG answer generation."""
    rag_retriever = LargeScaleRAGRetriever(
        store_dir=Path(args.store_dir),
        api_base_url=args.llm_api_url,
        model_name=args.llm_model,
        api_key=args.llm_api_key,
        llm_max_tokens=args.llm_max_tokens,
        llm_temperature=args.llm_temperature,
        embedding_model=args.model,
        backend=args.backend,
        provider=args.provider,
        onnx_file=args.onnx_file,
        truncate_dim=args.truncate_dim,
        omp_threads=args.omp_threads,
        ort_intra=args.ort_intra,
        ort_inter=args.ort_inter,
        embed_batch=args.embed_batch,
        doc_workers=args.doc_workers,
        shard_cache=args.shard_cache,
        use_mmap=not args.disable_mmap,
    )
 
    try:
        # Retrieve and generate answer
        result = rag_retriever.retrieve_and_generate(
            question=args.question,
            top_k=args.top_k,
            max_chars_per_doc=args.max_chars_per_doc,
        )
 
        print("\n" + "=" * 80)
        print("QUESTION")
        print("=" * 80)
        print(result["question"])
 
        print("\n" + "=" * 80)
        print("GENERATED ANSWER")
        print("=" * 80)
        print(result["answer"])
 
        print("\n" + "=" * 80)
        print("RETRIEVED DOCUMENTS")
        print("=" * 80)
        for i, doc in enumerate(result["retrieved_documents"], 1):
            meta = doc.get("meta", {})
            print(f"\n[{i}] Score: {doc.get('score', 0.0):.4f}")
            print(f"    File: {meta.get('source_file', 'N/A')}")
            print(f"    Timestamp: {meta.get('timestamp', 'N/A')}")
            snippet = doc.get("content", "")[:args.preview_chars]
            print(f"    Content: {snippet.replace(chr(10), ' ')}...")
 
        print(f"\n{'=' * 80}")
        print("PERFORMANCE BREAKDOWN")
        print("=" * 80)
        print(f"\n📊 RETRIEVAL PHASE: {result['retrieval_total_time']*1000:.1f}ms")
        print(f"    └─ Detailed: {_format_stats(result['retrieval_stats'])}")
        print(f"\n🤖 GENERATION PHASE: {result['generation_total_time']*1000:.1f}ms")
        print(f"    ├─ Document conversion: {result['doc_conversion_time']*1000:.1f}ms")
        print(f"    ├─ Prompt building: {result['prompt_build_time']*1000:.1f}ms")
        print(f"    └─ LLM inference: {result['llm_inference_time']*1000:.1f}ms")
        print(f"\n⏱️  TOTAL TIME: {result['total_time']*1000:.1f}ms")
        print(f"📚 Documents retrieved: {result['num_documents']}")
        print("=" * 80 + "\n")
 
    finally:
        rag_retriever.close()
 
 
def batch_query_rag(args: argparse.Namespace):
    """Batch query with RAG answer generation - parallelized version."""
    queries: List[str] = []
    with open(args.query_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            queries.append(line)
 
    if not queries:
        raise SystemExit("Query file is empty")
 
    rag_retriever = LargeScaleRAGRetriever(
        store_dir=Path(args.store_dir),
        api_base_url=args.llm_api_url,
        model_name=args.llm_model,
        api_key=args.llm_api_key,
        llm_max_tokens=args.llm_max_tokens,
        llm_temperature=args.llm_temperature,
        embedding_model=args.model,
        backend=args.backend,
        provider=args.provider,
        onnx_file=args.onnx_file,
        truncate_dim=args.truncate_dim,
        omp_threads=args.omp_threads,
        ort_intra=args.ort_intra,
        ort_inter=args.ort_inter,
        embed_batch=args.embed_batch,
        doc_workers=args.doc_workers,
        shard_cache=args.shard_cache,
        use_mmap=not args.disable_mmap,
    )
 
    try:
        overall_start = time.perf_counter()
 
        # Track aggregate timing stats
        all_retrieval_times = []
        all_generation_times = []
        all_total_times = []
 
        # PARALLELIZATION STRATEGY:
        # 1. First, batch retrieval for all queries (already optimized in base retriever)
        # 2. Then, parallelize the generation phase using ThreadPoolExecutor
 
        print(f"\n{'=' * 80}")
        print(f"Processing {len(queries)} queries with parallelism={args.rag_workers}")
        print("=" * 80)
 
        # Phase 1: Batch retrieval (embeddings + FAISS search + doc fetch)
        print(f"\n📊 Phase 1: Batch Retrieval")
        retrieval_start = time.perf_counter()
        all_docs, all_retrieval_stats = rag_retriever.retrieve_batch(
            queries=queries,
            top_k=args.top_k
        )
        batch_retrieval_time = time.perf_counter() - retrieval_start
        print(f"   ✓ Retrieved documents for {len(queries)} queries in {batch_retrieval_time*1000:.1f}ms")
        print(f"   ✓ Average per query: {batch_retrieval_time/len(queries)*1000:.1f}ms")
 
        # Phase 2: Parallel generation
        print(f"\n🤖 Phase 2: Parallel Generation (workers={args.rag_workers})")
        generation_start = time.perf_counter()
 
        def generate_for_query(query_idx: int) -> Dict:
            """Generate answer for a single query with retrieved documents."""
            query = queries[query_idx]
            docs = all_docs[query_idx]
            retrieval_stats = all_retrieval_stats[query_idx]
 
            # Generate answer using already-retrieved documents
            gen_start = time.perf_counter()
            rag_result = rag_retriever.rag_generator.generate_answer(
                question=query,
                documents=docs,
                max_chars_per_doc=args.max_chars_per_doc,
            )
            gen_time = time.perf_counter() - gen_start
 
            return {
                "query_idx": query_idx,
                "question": query,
                "answer": rag_result["answer"],
                "retrieved_documents": docs,
                "num_documents": len(docs),
                "retrieval_stats": retrieval_stats,
                "retrieval_total_time": 0.0,  # Amortized in batch
                "generation_total_time": gen_time,
                "prompt_build_time": rag_result.get("prompt_build_time", 0.0),
                "llm_inference_time": rag_result.get("llm_inference_time", 0.0),
                "doc_conversion_time": rag_result.get("doc_conversion_time", 0.0),
                "total_time": gen_time,
            }
 
        # Execute generation in parallel
        results = []
        if args.rag_workers > 1:
            with ThreadPoolExecutor(max_workers=args.rag_workers) as executor:
                futures = [executor.submit(generate_for_query, i) for i in range(len(queries))]
                for future in futures:
                    results.append(future.result())
        else:
            # Sequential fallback
            for i in range(len(queries)):
                results.append(generate_for_query(i))
 
        # Sort results by original query order
        results.sort(key=lambda x: x["query_idx"])
 
        batch_generation_time = time.perf_counter() - generation_start
        print(f"   ✓ Generated {len(queries)} answers in {batch_generation_time*1000:.1f}ms")
        print(f"   ✓ Average per query: {batch_generation_time/len(queries)*1000:.1f}ms")
 
        overall_time = time.perf_counter() - overall_start
 
        # Print individual results
        print(f"\n{'=' * 80}")
        print("RESULTS")
        print("=" * 80)
 
        for i, result in enumerate(results, 1):
            print(f"\n{'=' * 80}")
            print(f"Query {i}/{len(queries)}")
            print("=" * 80)
 
            # Track timings (using amortized retrieval time)
            amortized_retrieval = batch_retrieval_time / len(queries)
            all_retrieval_times.append(amortized_retrieval)
            all_generation_times.append(result['generation_total_time'])
            all_total_times.append(amortized_retrieval + result['generation_total_time'])
 
            print(f"\nQ: {result['question']}")
            print(f"\nA: {result['answer']}")
            print(f"\n📊 Timing Breakdown:")
            print(f"   Retrieval (amortized): {amortized_retrieval*1000:.1f}ms")
            print(f"   Generation: {result['generation_total_time']*1000:.1f}ms (LLM: {result['llm_inference_time']*1000:.1f}ms)")
            print(f"   Per-query total: {(amortized_retrieval + result['generation_total_time'])*1000:.1f}ms")
            print(f"\n📚 Retrieved {result['num_documents']} documents")
 
            if args.echo_results:
                print("\nRetrieved documents:")
                for j, doc in enumerate(result["retrieved_documents"], 1):
                    meta = doc.get("meta", {})
                    print(f"  [{j}] score={doc.get('score', 0.0):.4f} | file={meta.get('source_file', '')}")
 
        # Print aggregate statistics
        if len(queries) > 1:
            print(f"\n{'=' * 80}")
            print("AGGREGATE STATISTICS")
            print("=" * 80)
            print(f"\n⚡ PARALLELIZATION SUMMARY:")
            print(f"   Batch Retrieval Time: {batch_retrieval_time*1000:.1f}ms")
            print(f"   Batch Generation Time: {batch_generation_time*1000:.1f}ms")
            print(f"   Overall Wall-Clock Time: {overall_time*1000:.1f}ms")
            print(f"\n📊 PER-QUERY AVERAGES:")
            print(f"   Average Retrieval (amortized): {sum(all_retrieval_times)/len(all_retrieval_times)*1000:.1f}ms")
            print(f"   Average Generation: {sum(all_generation_times)/len(all_generation_times)*1000:.1f}ms")
            print(f"   Average Total: {sum(all_total_times)/len(all_total_times)*1000:.1f}ms")
            print(f"\n📈 THROUGHPUT:")
            print(f"   Total Queries: {len(queries)}")
            print(f"   Wall-Clock Throughput: {len(queries)/overall_time:.2f} queries/sec")
            print(f"   Sequential Equivalent Time: {sum(all_total_times):.2f}s")
            print(f"   Speedup: {sum(all_total_times)/overall_time:.2f}x")
            print("=" * 80 + "\n")
 
    finally:
        rag_retriever.close()
 
 
def main():
    ap = argparse.ArgumentParser(
        "Large-scale FAISS Flat disk retriever with Haystack RAG",
        description="Extends the base retriever with RAG capabilities using Haystack and LLM",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
 
    def add_common(p):
        # Base retriever parameters
        p.add_argument("--store-dir", type=str, default="./rag_flat_store")
        p.add_argument("--model", type=str, default="sentence-transformers/static-retrieval-mrl-en-v1")
        p.add_argument("--backend", type=str, default="onnx", choices=["onnx", "torch", "openvino"])
        p.add_argument("--provider", type=str, default="CPUExecutionProvider")
        p.add_argument("--onnx-file", type=str, default=None)
        p.add_argument("--truncate-dim", type=int, default=None)
        p.add_argument("--omp-threads", type=int, default=64)
        p.add_argument("--ort-intra", type=int, default=8)
        p.add_argument("--ort-inter", type=int, default=1)
        p.add_argument("--embed-batch", type=int, default=128)
        p.add_argument("--doc-workers", type=int, default=4)
        p.add_argument("--shard-cache", type=int, default=24)
        p.add_argument("--disable-mmap", action="store_true")
        p.add_argument("--top-k", type=int, default=5)
 
        # RAG/LLM parameters
        p.add_argument("--llm-api-url", type=str, default="http://localhost:5000/v1",
                      help="OpenAI-compatible API endpoint (e.g., vLLM, text-generation-inference)")
        p.add_argument("--llm-model", type=str, default="openai/gpt-oss-20b",
                      help="Model name for generation")
        p.add_argument("--llm-api-key", type=str, default="EMPTY",
                      help="API key (use EMPTY for local endpoints)")
        p.add_argument("--llm-max-tokens", type=int, default=2024,
                      help="Maximum tokens to generate")
        p.add_argument("--llm-temperature", type=float, default=0.1,
                      help="Sampling temperature")
 
    # Query with RAG command
    p_query_rag = sub.add_parser("query-rag", help="Single query with RAG answer generation")
    add_common(p_query_rag)
    p_query_rag.add_argument("--preview-chars", type=int, default=320)
    p_query_rag.add_argument("--max-chars-per-doc", type=int, default=500)
    p_query_rag.add_argument("--question", type=str)
 
    # Batch query with RAG command
    p_batch_rag = sub.add_parser("batch-query-rag", help="Batch queries with RAG answer generation")
    add_common(p_batch_rag)
    p_batch_rag.add_argument("--query-file", type=str, required=True)
    p_batch_rag.add_argument("--max-chars-per-doc", type=int, default=500)
    p_batch_rag.add_argument("--echo-results", action="store_true")
    p_batch_rag.add_argument("--rag-workers", type=int, default=4,
                            help="Number of parallel workers for RAG generation (default: 4)")
 
    args = ap.parse_args()
 
    if args.cmd == "query-rag":
        single_query_rag(args)
    elif args.cmd == "batch-query-rag":
        batch_query_rag(args)
    else:
        raise SystemExit(f"Unsupported command: {args.cmd}")
 
 
if __name__ == "__main__":
    main()