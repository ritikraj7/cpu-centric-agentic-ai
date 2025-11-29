# Haystack RAG System
 
A high-performance Retrieval-Augmented Generation (RAG) system built on FAISS for large-scale document retrieval with LLM-based answer generation using Haystack integration.
 
## Overview
 
This system combines efficient disk-based document retrieval using FAISS with language model generation capabilities to provide accurate, context-aware answers from large document collections. It features optimized retrieval performance through parallel processing, memory-mapped I/O, and intelligent caching strategies.
 
## Key Features
 
### Retrieval Engine
- **FAISS Flat Index**: Exact nearest-neighbor search for precise document retrieval
- **Disk-Based Storage**: Handles large-scale document collections that exceed memory limits
- **Memory-Mapped I/O**: Optimized file access with configurable shard caching (LRU cache)
- **Parallel Document Fetching**: Multi-threaded document retrieval for improved throughput
- **ONNX Runtime Integration**: Accelerated embedding generation with CPU optimization
 
### RAG Capabilities
- **Haystack Integration**: Modular RAG pipeline using Haystack components
- **Flexible LLM Backend**: OpenAI-compatible API support (vLLM, text-generation-inference, etc.)
- **Batch Processing**: Efficient parallel processing of multiple queries
- **Detailed Performance Metrics**: Comprehensive timing breakdowns for retrieval and generation phases
 
### Performance Optimizations
- Batch query embedding generation
- Parallel document retrieval with thread pooling
- Configurable worker pools for RAG generation
- Intelligent shard caching with memory mapping
- CPU-optimized ONNX runtime with configurable threading
 
## Architecture
 
```
Query → Embedding Model → FAISS Index → Document Retrieval → LLM Generation → Answer
         (ONNX/torch)     (Flat/Exact)   (SQLite+JSONL)      (Haystack)
```
 
### Components
 
1. **STEmbedder**: Sentence transformer wrapper for query embedding generation
2. **ExactFaiss**: FAISS index manager for similarity search
3. **ReadOnlyDocStore**: SQLite-backed document storage with JSONL shards
4. **ShardCache**: LRU cache for open file handles with mmap support
5. **HaystackRAGGenerator**: RAG pipeline using Haystack components
6. **LargeScaleRAGRetriever**: Main orchestrator combining retrieval and generation
 
## Dependencies
 
### Python Requirements
If already setup, activate langchain conda environment-

`conda activate main`

Or, if you want to setup from scratch-

```bash
pip install numpy faiss-cpu sentence-transformers onnxruntime haystack-ai requests
```
 
### C4 Documents
The RAG system is based on indexing a large scale document. In this work, we use C4 document
corpus (305 GB english variant). Follow the following steps:
1. Download the C4 document corpus from [hugging face website](https://huggingface.co/datasets/allenai/c4). A download script is provided
in haystack/download.py file.
2. Run the indexing file provided in haystack/indexing.py. Change the `--data-root` option to the
path of downloaded documents.
3. It can take multiple days (approximately 4-6) to index the whole document depending on the
system. Due to time limitations, this work only used 100 GB of indexing data.

### Optional Dependencies
- `onnxruntime-gpu`: For GPU-accelerated embedding generation
- `torch`: Alternative embedding backend (if not using ONNX)
 
## Usage
 
### Single Query with RAG
 
```bash
python retrieval.py query-rag \
    --store-dir ./rag_flat_store \
    --question "What is machine learning?" \
    --top-k 5 
```
 
### Batch Query Processing
 
```bash
python retrieval.py batch-query-rag \
    --store-dir ./rag_flat_store \
    --query-file questions.txt \
    --top-k 5 \
    --rag-workers 4
```
 
## Configuration
 
### Retrieval Parameters
 
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--store-dir` | `./rag_flat_store` | Directory containing FAISS index and document store |
| `--model` | `sentence-transformers/static-retrieval-mrl-en-v1` | Embedding model name |
| `--backend` | `onnx` | Embedding backend (onnx/torch/openvino) |
| `--top-k` | `5` | Number of documents to retrieve |
| `--embed-batch` | `128` | Batch size for embedding generation |
| `--doc-workers` | `4` | Thread pool size for document fetching |
| `--shard-cache` | `24` | Number of JSONL shards to keep open |
| `--omp-threads` | `64` | OpenMP threads for FAISS/ONNX |
| `--ort-intra` | `8` | ONNX Runtime intra-op threads |
| `--ort-inter` | `1` | ONNX Runtime inter-op threads |
 
### LLM Parameters
 
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--llm-api-url` | `http://172.27.149.251:8000/v1` | OpenAI-compatible API endpoint |
| `--llm-model` | Model path/name | LLM model identifier |
| `--llm-api-key` | `EMPTY` | API key (use "EMPTY" for local endpoints) |
| `--llm-max-tokens` | `2024` | Maximum tokens to generate |
| `--llm-temperature` | `0.1` | Sampling temperature for generation |
| `--max-chars-per-doc` | `500` | Maximum characters per document in context |
 
### Batch Processing Parameters
 
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rag-workers` | `4` | Number of parallel workers for RAG generation |
| `--query-file` | Required | Path to file containing queries (one per line) |
| `--echo-results` | False | Print detailed results for each query |
 
## Performance Tuning
 
### CPU Optimization
 
For CPU-based deployments, adjust threading parameters:
 
```bash
# High-core-count systems (64+ cores)
--omp-threads 64 --ort-intra 8 --ort-inter 1 --doc-workers 8
 
# Medium systems (16-32 cores)
--omp-threads 16 --ort-intra 4 --ort-inter 1 --doc-workers 4
 
# Low-core systems (4-8 cores)
--omp-threads 4 --ort-intra 2 --ort-inter 1 --doc-workers 2
```
 
### Memory Optimization
 
```bash
# Reduce memory usage
--shard-cache 8 --embed-batch 64 --disable-mmap
 
# Maximize throughput (high memory)
--shard-cache 48 --embed-batch 256
```
 
### Batch Processing Optimization
 
```bash
# Maximize parallelism for batch queries
--rag-workers 8 --doc-workers 8 --embed-batch 256
 
# Memory-constrained batch processing
--rag-workers 2 --doc-workers 2 --embed-batch 64
```
 
## Data Format
 
### Document Store Structure
 
```
rag_flat_store/
├── faiss/
│   └── index.faiss          # FAISS flat index
├── docstore/
│   ├── docs.sqlite3         # SQLite index (id, path, offset, length)
│   └── shard_*.jsonl        # JSONL document shards
```
 
### Document Format
 
Each document in the JSONL shards:
 
```json
{
  "content": "Document text content...",
  "meta": {
    "source_file": "path/to/source.txt",
    "timestamp": "2025-01-01T00:00:00",
    "custom_field": "value"
  }
}
```
 
### Query File Format
 
One query per line:
 
```
What is machine learning?
How does neural network training work?
Explain gradient descent
```

## Dependencies
 
Built with:
- **FAISS**: Similarity search and clustering
- **Haystack**: RAG pipeline components
- **ONNX Runtime**: Optimized model inference
- **Sentence Transformers**: Text embedding models
- **SQLite**: Document metadata indexing
- **NumPy**: Numerical operations
 
