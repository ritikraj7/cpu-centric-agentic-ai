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

### Tested Systems

#### Intel® Arc™ B580 Graphics

- CPU: Intel® Core™ Ultra 9 Processor 285K
- CPU Cores: 24 (8 Performance-cores and 16 Efficient-cores)
- CPU Threads: 24
- Memory: 64 GB
- GPU: Intel® Arc™ B580 Graphics
- GPU Memory: 12 GB
- Storage: 500 GB

#### Software Dependencies

- OS Ubuntu 26.04.1 LTS
- Intel Graphics Compute Runtime 26.31.39395.13
- Python 3.14.4

 
### Python Requirements

If already setup, activate haystack environment-

```bash
source agentic_haystack_rag_env/bin/activate
cd cpu-centric-agentic-ai/haystack
```

Or, if you want to setup from scratch-

```bash
cd [Workspace]
python3 -m venv agentic_haystack_rag_env
source agentic_haystack_rag_env/bin/activate
python -m pip install --upgrade pip

pip install numpy faiss-cpu==1.15.1 sentence-transformers==6.0.1 \
  onnxruntime==1.30.0 haystack-ai==2.31.0 requests==2.34.2 \
  datasets==5.0.1 psutil==7.2.2

git clone https://github.com/ritikraj7/cpu-centric-agentic-ai.git
cd cpu-centric-agentic-ai/haystack
```
 
### C4 Documents
The RAG system is based on indexing a large scale document. In this work, we use C4 document
corpus (15 GB realnewslike) instead of the 305 GB english variant. Follow the following steps:
1. Download the C4 document corpus from [hugging face website](https://huggingface.co/datasets/allenai/c4).
```bash
mkdir datasets
hf download allenai/c4 --repo-type dataset --include "realnewslike/*" --local-dir datasets
```
2. Run the indexing file provided in haystack/indexing.py. Change the `--data-root` option to the
path of downloaded documents.
```bash
python indexing.py index --data-root datasets/realnewslike
```
Output:
```
...
Indexing (Flat, async): 12573217docs [2:34:20, 1357.73docs/s]
💾 Saved index to rag_flat_store/faiss/flat.index (ntotal=12546302)
✅ Done: ntotal=12546302
```
3. It can take multiple hours to index the whole document depending on the
system.

### Optional Dependencies
- `onnxruntime-gpu`: For GPU-accelerated embedding generation
- `torch`: Alternative embedding backend (if not using ONNX)
 
## Usage

### LLM Server

Open a new Terminal to run local LLM server running at `http://localhost:8000/v1`.

Build vLLM image.

```bash
cd [Workspace]
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 3ca6ca2
sudo docker buildx build -f docker/Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .
```

```bash
sudo docker images
```
Output:
```
IMAGE                 ID             DISK USAGE   CONTENT SIZE
vllm-xpu-env:latest   3dfdbed7f610       37.9GB         9.26GB
```

Run vLLM container.

```bash
sudo docker run -it \
  --rm \
  --network=host \
  --device /dev/dri:/dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host \
  --privileged \
  --entrypoint bash \
  vllm-xpu-env
```

Inside vLLM container, spin up vLLM server.

```bash
vllm serve \
  Qwen/Qwen3.5-4B \
  --dtype=bfloat16 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --attention-backend TRITON_ATTN \
  --gpu-memory-utilization 0.85 \
  --max-model-len 5120 \
  --kv-cache-memory-bytes 500M
```
Output:
```
(EngineCore pid=1548) INFO 09-18 17:01:23 [gpu_worker.py:381] Initial free memory 10.06 GiB, reserved 0.49 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling. This does not respect the gpu_memory_utilization config. Only use kv_cache_memory_bytes config when you want manual control of KV cache memory size. If OOM'ed, check the difference of initial free memory between the current run and the previous run where kv_cache_memory_bytes is suggested and update it correspondingly.
(EngineCore pid=1548) INFO 09-18 17:01:23 [kv_cache_utils.py:1710] GPU KV cache size: 11,520 tokens
(EngineCore pid=1548) INFO 09-18 17:01:23 [kv_cache_utils.py:1711] Maximum concurrency for 5,120 tokens per request: 2.25x
```

Test vLLM server from another Terminal.

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data '{
        "model": "Qwen/Qwen3.5-4B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Write fibonacci code in python. And explain."
                    }
                ]
            }
        ]
    }'
```
 
### Single Query with RAG

Go back to haystack environment.
 
```bash
export OPENAI_TIMEOUT=600
python retrieval.py query-rag \
  --store-dir ./rag_flat_store \
  --llm-api-url "http://localhost:8000/v1" \
  --llm-model "Qwen/Qwen3.5-4B" \
  --llm-max-tokens 1024 \
  --question "What is machine learning?" \
  --top-k 5
```
Output:
```
📂 Loaded FAISS Flat index: ntotal=12546302
✅ Connected to LLM endpoint: http://localhost:8000/v1
🚀 Large-scale RAG retriever initialized with Haystack

================================================================================
QUESTION
================================================================================
What is machine learning?

================================================================================
GENERATED ANSWER
================================================================================
Thinking Process:
...

================================================================================
RETRIEVED DOCUMENTS
================================================================================

[1] Score: 0.8376
    File: c4-train.00440-of-00512.json.gz
    Timestamp: 2019-04-20T12:45:32Z
    Content: Why is Machine Learning so Hard? Machine Learning is a fascinating field that is rapidly emerging and heavily marketed as solution to many of today’s problems. Yet, the application of Machine Learning in a real-world production setting can be quite difficult to execute with promising results. In this post let’s examine...
...
[5] Score: 0.7769
    File: c4-train.00330-of-00512.json.gz
    Timestamp: 2019-04-22T09:27:31Z
    Content: Machine learning is poised to have a profound impact on your business but the hype is sowing confusion. Here’s a clear-eyed look at what machine learning is and how it can be used today. Machine learning is transforming business. But even as the technology advances, companies still struggle to take advantage of it, lar...

================================================================================
PERFORMANCE BREAKDOWN
================================================================================

📊 RETRIEVAL PHASE: 962.3ms
    └─ Detailed: embed=66.4ms, search=889.5ms, doc_fetch=5.9ms

🤖 GENERATION PHASE: 35494.7ms
    ├─ Document conversion: 0.8ms
    ├─ Prompt building: 0.1ms
    └─ LLM inference: 35493.8ms

⏱️  TOTAL TIME: 36456.9ms
📚 Documents retrieved: 5
================================================================================
```
 
### Batch Query Processing
 
```bash
export OPENAI_TIMEOUT=600
python retrieval.py batch-query-rag \
  --store-dir ./rag_flat_store \
  --llm-api-url "http://localhost:8000/v1" \
  --llm-model "Qwen/Qwen3.5-4B" \
  --llm-max-tokens 1024 \
  --query-file queries.txt \
  --top-k 5 \
  --rag-workers 1 \
  --echo-results
```
Output:
```
...
================================================================================
AGGREGATE STATISTICS
================================================================================

⚡ PARALLELIZATION SUMMARY:
   Batch Retrieval Time: 10098.1ms
   Batch Generation Time: 3604678.6ms
   Overall Wall-Clock Time: 3614776.7ms

📊 PER-QUERY AVERAGES:
   Average Retrieval (amortized): 78.9ms
   Average Generation: 28161.5ms
   Average Total: 28240.4ms

📈 THROUGHPUT:
   Total Queries: 128
   Wall-Clock Throughput: 0.04 queries/sec
   Sequential Equivalent Time: 3614.78s
   Speedup: 1.00x
================================================================================
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
 
