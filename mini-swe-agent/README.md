# Mini-SWE-Agent Coding Framework
 
A comprehensive coding framework for evaluating LLM-based software engineering agents with detailed latency profiling and performance analysis across multiple computational and coding tasks.
 
## Overview
 
This benchmark measures the performance of autonomous software engineering agents built on large language models (LLMs) using the vLLM inference server. It provides detailed timing breakdowns for LLM inference, bash command execution, and overall task completion across diverse problem domains.
 
## Architecture
 
```
User → LatencyBenchmarker → DefaultAgent → VLLMModel (vLLM Server)
                                    ↓
                              LocalEnvironment (Bash Execution)
                                    ↓
                              Incremental Results Logging
```
 
### Processing Pipeline
 
1. **Task Setup**: Load benchmark configuration and initialize agent
2. **Agent Execution**: Multi-step reasoning with LLM and bash tools
3. **Latency Tracking**: Record timestamps for all LLM calls and bash executions
4. **Result Aggregation**: Compute timing summaries and success metrics
5. **Incremental Saving**: Persist results to prevent data loss
 
## Installation
 
### Requirements

If already setup, activate langchain conda environment-

`conda activate swe`

Or, if you want to setup from scratch-

```bash
pip install datasets vllm pyyaml min
```
 
### Dependencies
 
- **datasets**: HuggingFace datasets library for SWE-bench, SciCode, LiveCodeBench
- **vllm**: GPU-accelerated LLM inference server
- **pyyaml**: YAML configuration parsing
- **Custom modules**: `vllm_model`, `minisweagent` (included in repository)
 
### vLLM Server Setup
 
Start vLLM server with Qwen2.5-Coder or similar coding model:
 
```bash
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
    --host 0.0.0.0 \
    --port 5000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95
```
 
## Usage
 
### Basic Benchmark Execution
 
```bash
python benchmark_latency.py --benchmark-type sorting
```

## Benchmark Types
 
### 1. CPU-Intensive Benchmarks
 
#### Sorting Algorithms
Tests agent's ability to implement and benchmark sorting algorithms (bubble sort on 10K-20K elements).
 
**Metrics**: Implementation correctness, timing accuracy, code quality
 
```bash
python benchmark_latency.py --benchmark-type sorting
```
 
#### Numerical Integration
Implements trapezoidal rule with numpy and scipy for sin(x) integration.
 
**Metrics**: Numerical accuracy, performance comparison, step count handling
 
```bash
python benchmark_latency.py --benchmark-type integration
```
 
#### K-Nearest Neighbors
KNN classifier implementation from scratch or with scikit-learn.
  
```bash
python benchmark_latency.py --benchmark-type knn
```
 
### 2. Software Engineering Benchmarks
 
#### SWE-bench
Real-world GitHub issue resolution from popular repositories.
 
**Dataset**: SWE-bench verified instances
**Source**: HuggingFace `princeton-nlp/SWE-bench_Lite`
 
#### SciCode
Scientific computing problems requiring domain expertise.
 
**Dataset**: SciCode problems from research domains
**Difficulty**: Advanced scientific programming

 
 
## Detailed Logging
 
All benchmarks track:
- **LLM API calls**: Timestamp, duration, prompt/completion tokens, cost
- **Bash executions**: Command, stdout/stderr, exit code, duration
- **Agent messages**: Full conversation history
- **Timing breakdown**: LLM vs. bash vs. overhead percentages

 
## Acknowledgments
 
- **SWE-bench Team** at Princeton NLP for the software engineering benchmark
- **Qwen Team** at Alibaba for the Qwen2.5-Coder models
- **vLLM Team** for high-performance inference framework
- **HuggingFace** for datasets infrastructure
- **SciCode** and **LiveCodeBench** contributors
 
 