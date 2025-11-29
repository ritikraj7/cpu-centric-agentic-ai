# LangChain Orchestrator
 
A high-performance batch LLM orchestrator built with LangGraph that implements a complete web-queried chatbot pipeline with comprehensive performance profiling using NVTX markers.
 
## Overview
 
This orchestrator processes multiple queries in batches through a four-stage pipeline: web search, content fetching, summarization, and LLM inference. Each stage is instrumented with NVTX markers for detailed performance analysis and supports parallel processing for optimal throughput. The system implements a stateful graph workflow with the following stages:
 
```
web_search → fetch_url → summarize → final_answer
```
 
### Pipeline Stages
 
1. **Web Search** - Uses Google Custom Search API to retrieve relevant URLs
2. **Content Fetching** - Downloads and extracts plain text from web pages (parallel processing)
3. **Summarization** - Generates extractive summaries using LexRank algorithm (parallel processing)
4. **LLM Inference** - Produces final answers using a local VLLM-hosted language model
 
 
## Requirements
 
### Python Dependencies
 
If already setup, activate langchain conda environment-

`conda activate langchain`

Or, if you want to setup from scratch-

```bash
pip install langchain==0.3.27 langgraph==0.6.10 langchain-core==0.3.79 langchain-community==0.3.31
pip install requests beautifulsoup4==4.14.2 sumy==0.11.0
pip install nvtx
```
 
### External Services
 
- **Google Custom Search API**: Requires `GOOGLE_API_KEY` and `GOOGLE_CX` environment variables
    - Go to [Google API website](https://developers.google.com/custom-search/v1/introduction) to request an API key.
        - Click on 'Gey a Key' button.
        - Select or Create a new project.
        - Click on 'CONFIRM AND CONTINUE' button.
        - Click on 'SHOW KEY' button.
    - Go to [Google CX website](https://programmablesearchengine.google.com/controlpanel/all) to request Google CX code.
        - Select your search engine or Create one and go into that.
        - You can find the CX id titled as "Search engine ID".
        - Public URL also has the cx id in the Query param as ?cx=**.
- **VLLM Server**: Local LLM server running at `http://localhost:8000/v1`
    - `conda activate main` if already setup, otherwise install vllm from pip - `pip install vllm==0.11.0`
    - `export HF_HOME=/storage/ritikraj/hugging_face` if you want to use the downloaded model.
    - `vllm serve openai/gpt-oss-20b --no-prefix-caching`
    - Prefix caching is diabled by default to mimic fully independent runs that do not share any prompt. 
 
## Environment Setup
 
```bash
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_CX="your-custom-search-engine-id"
```
 
## Usage
 
### Basic Usage
 
```bash
python langchain_orchestrator.py --batch_size 4 --benchmark freshQA
```
 
### Command-Line Arguments
 
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch_size` | int | 1 | Number of queries to process in parallel |
| `--benchmark` | str | freshQA | Benchmark dataset to use (freshQA, musique, QASC) |
 
 
## Detailed Performance Monitoring
 
### NVTX Profiling
 
Each pipeline stage is annotated with NVTX markers in the format:
```
<stage_name>: <query_preview>
```
 
Use NVIDIA Nsight Systems to visualize:
```bash
nsys profile -t nvtx,cuda python langchain_orchestrator.py --batch_size 8
nsys-ui output_profile.nsys-rep
```
 
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
    model="openai/gpt-oss-20b",  # Change model here
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
