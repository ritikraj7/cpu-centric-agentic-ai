# CPU-Centric Agentic AI Benchmarking Suite

A comprehensive benchmarking framework for evaluating and optimizing CPU-centric agentic AI systems across multiple workloads, reproducing results from the research paper: **"A CPU-centric Perspective on Agentic AI"** ([arXiv:2511.00739](https://arxiv.org/pdf/2511.00739)).

[![Paper](https://img.shields.io/badge/arXiv-2511.00739-b31b1b.svg)](https://arxiv.org/pdf/2511.00739)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository provides a complete benchmarking suite for characterizing latency, throughput, and performance characteristics of CPU-centric agentic AI workloads. It includes five diverse real-world applications spanning web search, retrieval-augmented generation (RAG), code generation, mathematical problem-solving, and chemistry research.

## Table of Contents

- [Workloads](#workloads)
- [Quick Start](#quick-start)
- [Reproducing Paper Figures](#reproducing-paper-figures)
- [Directory Structure](#directory-structure)
- [Performance Tuning](#performance-tuning)
- [System Requirements](#system-requirements)
- [Citation](#citation)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Workloads

### 1. LangChain (Web Search Orchestrator)

**Description**: Multi-stage pipeline for web-queried question answering using Google Search, web scraping, summarization, and LLM inference.

**Key Characteristics**:
- Pipeline: Web Search → Content Fetching → Summarization → LLM Generation
- Benchmarks: FreshQA, MuSiQue, QASC
- Model (customizable): GPT-OSS-20B (via vLLM) 
- Optimization: Parallel URL fetching, batch LLM inference

**Location**: [`langchain/`](langchain/) | [README](langchain/README.md)

### 2. Haystack (RAG System)

**Description**: Large-scale FAISS-based retrieval-augmented generation system for document search and question answering.

**Key Characteristics**:
- Architecture: Query Embedding → FAISS Search → Document Retrieval → LLM Generation
- Index Type: FAISS Flat (exact nearest neighbors)
- Scale: 10M+ documents with 768-dim embeddings
- Optimization: Memory-mapped I/O, LRU shard caching, parallel retrieval

**Location**: [`haystack/`](haystack/) | [README](haystack/README.md)

### 3. Mini-SWE-Agent (Code Generation)

**Description**: Software engineering agent for automated code generation, debugging, and computational tasks.

**Key Characteristics**:
- Tasks: Sorting, Fibonacci, integration, KNN, FFT, Sudoku, SWE-bench
- Model: Qwen2.5-Coder-32B-Instruct (via vLLM)
- Features: Bash execution, multi-step reasoning, latency profiling
- Datasets: SWE-bench, SciCode, LiveCodeBench

**Location**: [`mini-swe-agent/`](mini-swe-agent/) | [README](mini-swe-agent/README.md)

### 4. Toolformer (Math Problem-Solving)

**Description**: GPT-J-6B augmented with calculator tools for mathematical word problem solving.

**Key Characteristics**:
- Model: GPT-J-6B with Toolformer methodology
- Tools: Wolfram Alpha API, local AST calculator
- Datasets: ASDiv, SVAMP, MAWPS
- Features: Self-supervised tool usage, parallel processing

**Location**: [`toolformer/`](toolformer/) | [README](toolformer/README.md)

### 5. ChemCrow

**Description**: Chemistry-focused research agent for scientific literature search and molecular analysis.

**Key Characteristics**:
- Model: GPT-4 (OpenAI API) with chemistry tools
- Tasks: Literature search, molecular properties, reaction prediction
- Tools: LiteratureSearch (paperscraper), RDKit, PubChem
- Features: Tool usage analysis, detailed timing breakdowns

**Location**: [`chemcrow/`](chemcrow/) | [README](chemcrow/README.md)

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ GPU memory (for LLM inference)
- 64GB+ RAM (for RAG workloads)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
```

2. **Set up conda environments** (3 separate environments required):

```bash
# Main environment (for vLLM, Haystack, Toolformer)
conda env create -f conda_main_env.yml
conda activate main

# LangChain environment
conda env create -f conda_langchain_env.yml

# SWE-Agent environment
conda env create -f conda_swe_env.yml

# Optionally, ChemCrow environment
conda env create -f chemcrow/chemcrow_env.yml
```

3. **Configure API keys**:

```bash
# Google Custom Search (for LangChain)
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_CX="your-custom-search-engine-id"

# Wolfram Alpha (for Toolformer)
export WOLFRAM_ALPHA_APPID="your-wolfram-appid"

# OpenAI (for ChemCrow)
export OPENAI_API_KEY="your-openai-api-key"
```

4. **Download models**:

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/huggingface/cache

# Models will be downloaded automatically on first use
```

5. **Additional Dependencies for Individual Workloads**

Please refer to the README of individual workloads for additional dependencies and setup instructions.

### Running a Simple Benchmark

**Example 1: LangChain Web Search**

```bash
conda activate langchain
export GOOGLE_API_KEY="..."
export GOOGLE_CX="..."

# Start vLLM server
vllm serve openai/gpt-oss-20b --port 8000 &

# Run benchmark
python langchain/orchestrator.py --batch-size 1 --benchmark freshQA
```

**Example 2: Haystack RAG**

```bash
conda activate main

# Start vLLM server
vllm serve openai/gpt-oss-20b --port 8000 &

# Single query
python haystack/retrieval.py query-rag \
    --store-dir ./rag_flat_store \
    --question "What is machine learning?"
```

**Example 3: Mini-SWE-Agent**

```bash
conda activate swe

# Start vLLM server
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 5000 &

# Run sorting benchmark
python mini-swe-agent/benchmark_latency.py --benchmark-type sorting
```

## Reproducing Paper Figures

Most of the figures from the paper can be reproduced using scripts in the [`scripts/`](scripts/) directory. Each script is self-contained and handles vLLM server management automatically. Please refer `energy/README.md` for reproduction of figure 5 in the paper. We will add the scripts to reproduce remaining figures soon. 

**Estimated Time**: 1-2 hours


### Figure 2: Latency Breakdown Across Workloads

**Description**: Compares end-to-end latency breakdown (LLM inference, tool execution, framework overhead) across all five workloads. The script only contains reproduction of CPU-bound workloads: Langchain, Haystack and Mini-SWE-agent.

```bash
cd scripts
bash figure_2.sh
```

**Output**: `figures/figure_2.png`


### Figure 3: LangChain Parallelization Strategies

**Description**: Evaluates sequential, multi-threading, and multi-processing approaches for LangChain across batch sizes 1-128.

```bash
cd scripts
bash figure_3.sh
```

**Output**: `figures/figure_3.png`

### Figure 4a: Throughput vs Batch Size

**Description**: LLM inference throughput (requests/sec) vs batch size for different input/output token configurations.

```bash
cd scripts
bash figure_4a.sh
```

**Output**: `figures/figure_4a.png`

### Figure 4b: Agentic Workload Throughput

**Description**: End-to-end throughput analysis for LangChain and Haystack with varying batch sizes.

```bash
cd scripts
bash figure_4b.sh
```

**Output**: `figures/figure_4b.png`

### Figure 4c: LangChain Stage-Level Latency

**Description**: Time-per-call for different stages (web search/summarization/LLM inference) of Langchain workload across batch sizes 1-128.

```bash
cd scripts
bash figure_4c.sh
```

**Output**: `figures/figure_4c.png`

### Figure 7a: Evaluation of CGAM Optimization for Langchain

**Description**: Percentile latency analysis (with highlighted P50 and P99) for LangChain workload with CGAM and CGAM overlap optimzation compared against multi-processing baseline.

```bash
cd scripts
bash figure_7a.sh
```

**Output**: `figures/figure_7a.png`

### Run All Experiments

```bash
cd scripts
bash figure_2.sh
bash figure_3.sh
bash figure_4a.sh
bash figure_4b.sh
bash figure_4c.sh
bash figure_7a.sh
```

## Directory Structure

```
├── langchain/              # Web search orchestrator
│   ├── orchestrator.py     # Main pipeline implementation
│   ├── plot_*.py           # Visualization scripts
│   └── README.md           # Detailed documentation
├── haystack/               # RAG system
│   ├── retrieval.py        # FAISS-based retrieval + LLM generation
│   └── README.md
├── mini-swe-agent/         # Code generation agent
│   ├── benchmark_latency.py
│   ├── src/minisweagent/   # Agent implementation
│   └── README.md
├── toolformer/             # Math problem solver
│   ├── math_toolformer.py  # Single-batch latency
│   ├── batch_toolformer.py # Multi-batch throughput
│   └── README.md
├── chemcrow/               # Chemistry research agent
│   ├── benchmark_chemcrow_tasks.py
│   └── README.md
├── scripts/                # Paper figure reproduction scripts
│   ├── figure_2.sh
│   ├── figure_3.sh
│   ├── figure_4a.sh
│   ├── figure_4b.sh
│   ├── figure_4c.sh
│   └── figure_7a.sh
├── figures/                # Generated plots
├── energy/                 # Energy measurement utilities
├── throughput.py           # LLM throughput benchmarking
├── plot_latency.py         # Latency visualization
├── plot_throughput.py      # Throughput visualization
├── plot_agentic_throughput.py  # Agentic workload analysis
└── README.md               # This file
```

## Performance Tuning

### NUMA Affinity

For multi-socket systems, use NUMA binding:

```bash
numactl --cpunodebind=0 --membind=0 python langchain/orchestrator.py
```

### Parallel Processing

Adjust worker counts based on available CPU cores:

```python
# LangChain
python langchain/orchestrator.py --batch-size 32 --workers 16

# Haystack
python haystack/retrieval.py batch-query-rag --rag-workers 8
```

<!-- ## Troubleshooting

### Common Issues

**Issue 1: CUDA Out of Memory**
```
Solution: Reduce batch size or max-model-len in vLLM
vllm serve MODEL --max-model-len 2048 --gpu-memory-utilization 0.8
```

**Issue 2: vLLM Server Connection Refused**
```
Solution: Check if server is running
curl http://localhost:8000/health
```

**Issue 3: Google API Rate Limiting**
```
Solution: Add delay between requests or use multiple API keys
```

**Issue 4: Conda Environment Conflicts**
```
Solution: Use separate environments as specified
conda activate main  # For vLLM
conda activate langchain  # For LangChain
conda activate swe  # For SWE-Agent
```

### Debug Mode

Enable verbose logging:

```bash
# LangChain
python langchain/orchestrator.py --verbose

# Haystack
python haystack/retrieval.py query-rag --echo-results

# SWE-Agent
python mini-swe-agent/benchmark_latency.py --log-level DEBUG
```

### Performance Profiling

Use NVIDIA Nsight Systems for detailed profiling:

```bash
nsys profile -o profile.nsys-rep \
    -t cuda,nvtx,osrt,python \
    python langchain/orchestrator.py --batch-size 8

nsys-ui profile.nsys-rep
``` -->

## Citation

If you use this benchmark suite in your research, please cite the paper:

```bibtex
@article{agentic2025,
  title={Characterizing and Optimizing Agentic AI Workloads},
  author={[Authors]},
  journal={arXiv preprint arXiv:2511.00739},
  year={2025},
  url={https://arxiv.org/pdf/2511.00739}
}
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-workload`)
3. Add tests and documentation
4. Submit a pull request

### Areas for Contribution

- Additional workloads (agents, tools, frameworks)
- Performance optimizations
- New benchmark datasets
- Visualization improvements
- Documentation enhancements
- Bug fixes and error handling

## Acknowledgments

This benchmark suite builds upon excellent open-source projects:

- **vLLM**: High-performance LLM inference ([vllm-project/vllm](https://github.com/vllm-project/vllm))
- **LangChain**: Agentic AI orchestration framework ([langchain-ai/langchain](https://github.com/langchain-ai/langchain))
- **Haystack**: NLP framework for RAG ([deepset-ai/haystack](https://github.com/deepset-ai/haystack))
- **FAISS**: Similarity search library ([facebookresearch/faiss](https://github.com/facebookresearch/faiss))
- **mini-swe-agent**: Software engineering AI agent ([SWE-agent/mini-swe-agent](https://github.com/princeton-nlp/SWE-bench))
- **SWE-bench**: Software engineering benchmark ([princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench))
- **ChemCrow**: Chemistry research agent ([ur-whitelab/chemcrow-public](https://github.com/ur-whitelab/chemcrow-public))

Special thanks to the research community for datasets and evaluation frameworks.

**Note**: Parts of the code in this repository have been generated using AI tools including ChatGPT and Claude AI.

## Contact

For questions, issues, or collaboration, please contact the author-

Ritik Raj (ritik.raj@gatech.edu)
