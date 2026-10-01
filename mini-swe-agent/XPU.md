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
 
### Requirements

If already setup, activate the environment-

```bash
source agentic_swe_codegen_env/bin/activate
cd cpu-centric-agentic-ai/mini-swe-agent
```

Or, if you want to setup from scratch-

```bash
cd [Workspace]
python3 -m venv agentic_swe_codegen_env
source agentic_swe_codegen_env/bin/activate
python -m pip install --upgrade pip

pip install datasets==5.0.1 pyyaml==6.0.3 python-dotenv==1.2.3 \
  platformdirs==4.11.11 rich==15.0.0 jinja2==3.1.6

git clone https://github.com/ritikraj7/cpu-centric-agentic-ai.git
cd cpu-centric-agentic-ai/mini-swe-agent
```
 
### Dependencies
 
- **datasets**: HuggingFace datasets library for SWE-bench, SciCode, LiveCodeBench
- **vllm**: GPU-accelerated LLM inference server
- **pyyaml**: YAML configuration parsing
- **Custom modules**: `vllm_model`, `minisweagent` (included in repository)
 
### vLLM Server Setup
 
Start vLLM server with Qwen2.5-Coder or similar coding model on different Terminal.
 
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
  Qwen/Qwen2.5-Coder-3B-Instruct \
  --dtype=bfloat16 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --attention-backend TRITON_ATTN \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --kv-cache-memory-bytes 750M
```
Output:
```
(EngineCore pid=2122) INFO 09-19 12:40:20 [default_loader.py:391] Loading weights took 0.69 seconds
(EngineCore pid=2122) INFO 09-19 12:40:20 [gpu_model_runner.py:4883] Model loading took 5.79 GiB memory and 2.949057 seconds
(EngineCore pid=2122) INFO 09-19 12:40:21 [gpu_worker.py:381] Initial free memory 10.18 GiB, reserved 0.73 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling. This does not respect the gpu_memory_utilization config. Only use kv_cache_memory_bytes config when you want manual control of KV cache memory size. If OOM'ed, check the difference of initial free memory between the current run and the previous run where kv_cache_memory_bytes is suggested and update it correspondingly.
(EngineCore pid=2122) INFO 09-19 12:40:21 [kv_cache_utils.py:1710] GPU KV cache size: 21,328 tokens
(EngineCore pid=2122) INFO 09-19 12:40:21 [kv_cache_utils.py:1711] Maximum concurrency for 16,384 tokens per request: 1.30x
```

Test vLLM server from another Terminal.

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data '{
        "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
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
 
## Usage
 
### Basic Benchmark Execution
 
```bash
export OPENAI_TIMEOUT=600
python benchmark_latency.py \
  --base-url "http://localhost:8000" --model-path "Qwen/Qwen2.5-Coder-3B-Instruct" \
  --max-tokens 8192 \
  --benchmark-type sorting
```

## Benchmark Types
 
### 1. CPU-Intensive Benchmarks
 
#### Sorting Algorithms
Tests agent's ability to implement and benchmark sorting algorithms (bubble sort on 10K-20K elements).
 
**Metrics**: Implementation correctness, timing accuracy, code quality
 
```bash
export OPENAI_TIMEOUT=600
python benchmark_latency.py \
  --base-url "http://localhost:8000" --model-path "Qwen/Qwen2.5-Coder-3B-Instruct" \
  --max-tokens 8192 \
  --benchmark-type sorting
```
```bash
head -n 30 benchmark_results/sorting_benchmark.json
```
Output:
```
{
  "dataset": "sorting_algorithms",
  "exit_status": "LimitsExceeded",
  "result": "",
  "total_runtime": 91.50305032730103,
  "total_wall_time": 91.50305032730103,
  "task_preview": "\nSorting Algorithms Benchmark\n\nProblem Description:\nWrite Python code to implement and benchmark bubble sort on arrays of sizes 10000 and 20000 elements.\n\nInstructions:\n1. Create a Python script with ...",
  "model_calls": 15,
  "model_cost": 0.15,
  "timing_summary": {
    "total_llm_time_seconds": 81.30268001556396,
    "total_bash_time_seconds": 10.176735401153564,
    "average_llm_time_seconds": 5.420178667704264,
    "average_bash_time_seconds": 0.678449026743571,
    "total_llm_calls": 15,
    "total_bash_calls": 15,
    "llm_time_percentage": 88.85242592979039,
    "bash_time_percentage": 11.121744427920142,
    "other_time_seconds": 0.023634910583496094
  },
  "detailed_logs": {
...
```
 
#### Numerical Integration
Implements trapezoidal rule with numpy and scipy for sin(x) integration.
 
**Metrics**: Numerical accuracy, performance comparison, step count handling
 
```bash
export OPENAI_TIMEOUT=600
python benchmark_latency.py \
  --base-url "http://localhost:8000" --model-path "Qwen/Qwen2.5-Coder-3B-Instruct" \
  --max-tokens 8192 \
  --benchmark-type integration
```
```bash
head -n 30 benchmark_results/integration_benchmark.json
```
Output:
```
{
  "dataset": "numerical_integration",
  "exit_status": "LimitsExceeded",
  "result": "",
  "total_runtime": 54.39385223388672,
  "total_wall_time": 54.39385223388672,
  "task_preview": "\nNumerical Integration Benchmark\n\nProblem Description:\nWrite Python code to compute numerical integration of sin(x) from 0 to pi using trapezoidal rule with 1000000 and 10000000 steps. Compare numpy.t...",
  "model_calls": 12,
  "model_cost": 0.11999999999999998,
  "timing_summary": {
    "total_llm_time_seconds": 49.00807452201843,
    "total_bash_time_seconds": 5.363288402557373,
    "average_llm_time_seconds": 4.084006210168202,
    "average_bash_time_seconds": 0.44694070021311444,
    "total_llm_calls": 12,
    "total_bash_calls": 12,
    "llm_time_percentage": 90.09855435737458,
    "bash_time_percentage": 9.860100328058966,
    "other_time_seconds": 0.022489309310913086
  },
  "detailed_logs": {
...
```
 
#### K-Nearest Neighbors
KNN classifier implementation from scratch or with scikit-learn.
  
```bash
export OPENAI_TIMEOUT=600
python benchmark_latency.py \
  --base-url "http://localhost:8000" --model-path "Qwen/Qwen2.5-Coder-3B-Instruct" \
  --max-tokens 8192 \
  --benchmark-type knn
```
```bash
head -n 30 benchmark_results/knn_benchmark.json
```
Output:
```
{
  "dataset": "knn_numpy",
  "exit_status": "LimitsExceeded",
  "result": "",
  "total_runtime": 27.27873921394348,
  "total_wall_time": 27.27873921394348,
  "task_preview": "\nk-NN Benchmark \n\nProblem Description:\nImplement a naive k-Nearest Neighbors classifier for k=5 on random datasets with shapes (4k\u00d732) and (6k\u00d732). Report latency and memory usage.\n\n\nPlease implement ...",
  "model_calls": 8,
  "model_cost": 0.08,
  "timing_summary": {
    "total_llm_time_seconds": 24.323007822036743,
    "total_bash_time_seconds": 2.939878463745117,
    "average_llm_time_seconds": 3.040375977754593,
    "average_bash_time_seconds": 0.36748480796813965,
    "total_llm_calls": 8,
    "total_bash_calls": 8,
    "llm_time_percentage": 89.16470673836744,
    "bash_time_percentage": 10.777178669028821,
    "other_time_seconds": 0.015852928161621094
  },
  "detailed_logs": {
...
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
 
 