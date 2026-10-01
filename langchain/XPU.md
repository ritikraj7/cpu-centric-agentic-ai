# LangChain Orchestrator
 
A high-performance batch LLM orchestrator built with LangGraph that implements a complete web-queried chatbot pipeline.
 
## Overview
 
This orchestrator processes multiple queries in batches through a four-stage pipeline: web search, content fetching, summarization, and LLM inference. Each stage is instrumented with performance analysis and supports parallel processing for optimal throughput. The system implements a stateful graph workflow with the following stages:
 
```
web_search → fetch_url → summarize → final_answer
```
 
### Pipeline Stages
 
1. **Web Search** - Uses Google Custom Search API to retrieve relevant URLs
2. **Content Fetching** - Downloads and extracts plain text from web pages (parallel processing)
3. **Summarization** - Generates extractive summaries using LexRank algorithm (parallel processing)
4. **LLM Inference** - Produces final answers using a local VLLM-hosted language model
 
 
## Requirements

### Tested Systems

#### Intel® Arc™ B580 Graphics

- CPU: Intel® Core™ Ultra 9 Processor 285K
- CPU Cores: 24 (8 Performance-cores and 16 Efficient-cores)
- CPU Threads: 24
- Memory: 64 GB
- GPU: Intel® Arc™ B580 Graphics
- GPU Memory: 12 GB
- Storage: 500 GB

### Software Dependencies

- OS Ubuntu 26.04.1 LTS
- Intel Graphics Compute Runtime 26.31.39395.13
- Python 3.14.4
 
#### Python Dependencies
 
If already setup, activate langchain environment-

```bash
source agentic_langchain_orchestrator_env/bin/activate
cd cpu-centric-agentic-ai/langchain
```

Or, if you want to setup from scratch-

```bash
cd [Workspace]
python3 -m venv agentic_langchain_orchestrator_env
source agentic_langchain_orchestrator_env/bin/activate
python -m pip install --upgrade pip

pip install langchain==0.3.27 langgraph==0.6.10 langchain-core==0.3.79 langchain-community==0.3.31
pip install requests==2.34.2 beautifulsoup4==4.14.2 sumy==0.11.0
pip install nvtx
pip install openai==3.13.0

python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

git clone https://github.com/ritikraj7/cpu-centric-agentic-ai.git
cd cpu-centric-agentic-ai/langchain
```

Change line 142 in orchestrator.py with the vLLM URL and model.

```python
def final_answer(state: GraphState) -> GraphState:
    ...
    llm = VLLMOpenAI(
        base_url='http://localhost:8000/v1',
        model="Qwen/Qwen3.5-4B",
        openai_api_key='EMPTY'  # no API key required for local VLLM
    )
```

Spin up vLLM server before test run the following command.
```bash
python orchestrator.py --batch-size 1 --benchmark freshQA --skip-web-search
```
Output:
```
1: [TIMING] start: 97706.9427s
1: [TIMING] end: 34.5680s
```

### External Services
 
#### Google Custom Search API (optional)

Requires `GOOGLE_API_KEY` and `GOOGLE_CX` environment variables
- Go to [Google API website](https://developers.google.com/custom-search/v1/introduction) to request an API key.
    - Click on 'Gey a Key' button.
    - Select or Create a new project.
    - Click on 'CONFIRM AND CONTINUE' button.
    - Click on 'SHOW KEY' button.
- Go to [Google CX website](https://programmablesearchengine.google.com/controlpanel/all) to request Google CX code.
    - Select your search engine or Create one and go into that.
    - You can find the CX id titled as "Search engine ID".
    - Public URL also has the cx id in the Query param as ?cx=**.

#### vLLM Server

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
  --max-model-len 1024 \
  --kv-cache-memory-bytes 500M
```
Output:
```
(EngineCore pid=2461) INFO 09-11 16:26:47 [gpu_worker.py:381] Initial free memory 10.04 GiB, reserved 0.49 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling. This does not respect the gpu_memory_utilization config. Only use kv_cache_memory_bytes config when you want manual control of KV cache memory size. If OOM'ed, check the difference of initial free memory between the current run and the previous run where kv_cache_memory_bytes is suggested and update it correspondingly.
(EngineCore pid=2461) INFO 09-11 16:26:47 [kv_cache_utils.py:1710] GPU KV cache size: 5,529 tokens
(EngineCore pid=2461) INFO 09-11 16:26:47 [kv_cache_utils.py:1711] Maximum concurrency for 1,024 tokens per request: 5.40x
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

## Environment Setup

Optional:
```bash
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_CX="your-custom-search-engine-id"
```

## Usage
 
### Basic Usage
 
```bash
python orchestrator.py \
  --batch-size 1 \
  --benchmark freshQA --skip-web-search \
  --verbose
```
Output:
```
1: [TIMING] start: 97773.1318s
1: [TIMING] end: 29.8126s

======================================================================
TIMING STATISTICS (across all batches)
======================================================================
Stage                Count      Avg (s)      Min (s)      Max (s)     
----------------------------------------------------------------------
web_search           1          0.0000       0.0000       0.0000      
fetch_url            1          21.8449      21.8449      21.8449     
summarize            1          0.1813       0.1813       0.1813      
llm_inference        1          7.7843       7.7843       7.7843      
======================================================================
```

### Sample Timeline Profiling

```bash
python orchestrator.py \
  --benchmark freshQA --skip-web-search \
  --sequential --batch-size 10 \
  --verbose 
```
Output:
```
1: [TIMING] start: 72716.2859s
1: [TIMING] end: 273.2346s

======================================================================
TIMING STATISTICS (across all batches)
======================================================================
Stage                Count      Avg (s)      Min (s)      Max (s)     
----------------------------------------------------------------------
web_search           10         0.0000       0.0000       0.0000      
fetch_url            10         21.7297      21.5860      21.8841     
summarize            10         0.2029       0.1687       0.4649      
llm_inference        10         5.3893       1.9922       7.8809      
======================================================================
```

### Command-Line Arguments
 
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch_size` | int | 1 | Number of queries to process in parallel |
| `--benchmark` | str | freshQA | Benchmark dataset to use (freshQA, musique, QASC) |
 
 
## Detailed Performance Monitoring
 
### Timing Statistics
 
The system tracks execution time for each stage:
- **web_search**: Google API query time
- **fetch_url**: Web page download and parsing time
- **summarize**: LexRank summarization time
- **llm_inference**: LLM response generation time
 
Statistics include count, average, minimum, and maximum execution times across all batches.
 
## Configuration
 
### LLM Model Configuration
 
Edit the `final_answer()` function to configure the LLM:
 
```python
llm = VLLMOpenAI(
    base_url='http://localhost:8000/v1',
    model="Qwen/Qwen3.5-4B",  # Change model here
    openai_api_key='EMPTY'
)
```
 
### Search Results Limit
 
Control number of URLs fetched per query at line 96:
```python
if len(texts) >= 2:  # Adjust number of pages
    break
```

## Output Format
 
The system outputs timing information in the format:
```
<job_id>: [TIMING] start: <timestamp>s
<job_id>: [TIMING] end: <elapsed_time>s
```

Uncomment lines 258-262 to print full results:
```python
for state in result_states:
    print(f"🧑 » {state['query']}")
    print(f"🤖 » {state['final_response']}\n")
```

## Error Handling
 
- **Missing API Keys**: Raises `RuntimeError` if `GOOGLE_API_KEY` or `GOOGLE_CX` not set
- **Network Errors**: Silently skips failed URL fetches, continues with available content
- **Timeout Protection**: 10-second timeout on HTTP requests

## Development
 
### Extending the Pipeline
 
To add new pipeline stages:
 
1. Define node function with `GraphState` parameter
2. Add NVTX markers and timing instrumentation
3. Register node in graph builder
4. Connect with edges

```python
def new_stage(state: GraphState) -> GraphState:
    nvtx.push_range("new_stage")
    # ... implementation ...
    nvtx.pop_range()
    return {"new_field": result}
 
builder.add_node('new_stage', new_stage)
builder.add_edge('previous_stage', 'new_stage')
```
