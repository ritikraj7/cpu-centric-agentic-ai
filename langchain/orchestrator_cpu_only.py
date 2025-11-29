"""
CPU-only batch web orchestrator using LangGraph batching with per-query NVTX markers.
Accepts multiple queries as CLI args, runs web search, URL fetching, and summarization without LLM inference.
This version removes the final_answer node to make it CPU-only for performance profiling.
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, math
import sys
import timeit
from typing import List, Dict, Optional
import nvtx
import requests
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain_core.runnables.config import RunnableConfig
 
import time
# 1) Shared state schema - removed final_response since no LLM
typedef = TypedDict  # compatibility alias
class GraphState(typedef('GraphState', {})):
    query: str
    urls: List[str]
    page_texts: List[str]
    summaries: List[str]
 
# 2) Tool implementations with dynamic NVTX markers
 
def web_search(state: GraphState) -> GraphState:
    marker = f"web_search: {state['query'][:30]}"
    nvtx.push_range(marker)
    # key, cx = os.getenv('GOOGLE_API_KEY'), os.getenv('GOOGLE_CX')
    # if not key or not cx:
    #     nvtx.pop_range(); raise RuntimeError('Missing GOOGLE_API_KEY or GOOGLE_CX')
    # params = {'key': key, 'cx': cx, 'q': state['query'], 'num': 5}
    # resp = requests.get('https://www.googleapis.com/customsearch/v1', params=params, timeout=10)
    # resp.raise_for_status()
    # items = resp.json().get('items', [])
    # urls = [i['link'] for i in items if 'link' in i]
    urls = ["https://itif.org/publications/2024/08/19/how-innovative-is-china-in-semiconductors/",
    "https://www.csis.org/analysis/too-good-lose-americas-stake-intel",
    "https://www.piie.com/sites/default/files/2025-01/piieb25-1.pdf",
    "https://www.roc-taiwan.org/uploads/sites/86/2024/04/2024_April___May_Issue.pdf"]
    nvtx.pop_range()
    return {'urls': urls}
 
 
# def fetch_url(state: GraphState) -> GraphState:
#     texts = []
#     nvtx.push_range('Fetch Url')
#     for url in state["urls"]:
#         resp = requests.get(url, timeout=5)
#         soup = BeautifulSoup(resp.text, "html.parser")
#         texts.append(soup.get_text(separator="\n"))
#     nvtx.pop_range()
#     return {"page_texts": texts}
 
def _fetch_single(url: str, timeout: float = 10.0) -> Optional[str]:
    """Download one URL and return plain-text, or None on error."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser").get_text(separator="\n")
    except requests.RequestException:
        return None
 
 
# def fetch_url(state: GraphState) -> GraphState:
#     """Download up to *two* pages in parallel and return their text."""
#     marker = f"fetch_url: {state['query'][:30]}"
#     nvtx.push_range(marker)
 
#     texts: List[str] = []
#     urls = state["urls"]
 
#     # choose a sensible worker count (1-per-CPU but never > #urls)
#     max_workers = min(len(urls), os.cpu_count() or 1)
 
#     with ProcessPoolExecutor(max_workers=max_workers) as pool:
#         futures = {pool.submit(_fetch_single, url): url for url in urls}
 
#         # gather results as they complete
#         for fut in as_completed(futures):
#             page_text = fut.result()
#             if page_text:                       # keep only successful fetches
#                 texts.append(page_text)
#             if len(texts) >= 2:                 # reached the quota ➜ stop early
#                 break
 
#         # cancel anything still running
#         for fut in futures:
#             if not fut.done():
#                 fut.cancel()
 
#     nvtx.pop_range()
#     return {"page_texts": texts}
 
 
def _fetch_url_single_state(state: GraphState) -> GraphState:
    """Sequential download of up to two pages for *one* query."""
    marker = f"fetch_url: {state['query'][:30]}"
    nvtx.push_range(marker)
 
    texts: List[str] = []
    for url in state["urls"]:
        if len(texts) >= 2:
            break
        txt = _fetch_single(url)
        if txt:
            texts.append(txt)
 
    nvtx.pop_range()
    return {"page_texts": texts}
 
 
def fetch_url(state_or_states):  # LangGraph will pass list when batching
    """Batched wrapper: handles either a single state or a list of states."""
    if isinstance(state_or_states, list):
        # Parallelise across *queries* with a process pool
        max_workers = min(len(state_or_states), 1)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_fetch_url_single_state, state_or_states))
        return results
    else:
        return _fetch_url_single_state(state_or_states)
 
 
# def fetch_url(state: GraphState) -> GraphState:
#     marker = f"fetch_url: {state['query'][:30]}"
#     nvtx.push_range(marker)
 
#     texts: List[str] = []
#         # print(texts)
 
#     for url in state['urls']:
#         # stop once we've got 5 pages
#         if len(texts) >= 2:
#             break
 
#         try:
#             # this will raise on timeout or bad status
#             r = requests.get(url, timeout=10)
#             r.raise_for_status()
#         except requests.RequestException:
#             # skip timeouts, connection errors, 4xx/5xx, etc.
#             continue
 
#         # parse only if we got a good response
#         page_text = BeautifulSoup(r.text, 'html.parser') \
#                         .get_text(separator='\n')
#         texts.append(page_text)
 
#     nvtx.pop_range()
 
 
#     return {'page_texts': texts}
 
 
 
    # return {'page_texts': texts}
 
# --- helper: picklable worker function ---
def _lexrank_one(text: str) -> str:
    """Run LexRank on a single document and return one-sentence summary."""
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    summarizer = LexRankSummarizer()
    doc = PlaintextParser.from_string(text, Tokenizer("english")).document
    sentences = summarizer(doc, sentences_count=1)
    return " ".join(str(s) for s in sentences)
 
# --- replace your current summarize() node ---
def summarize(state: GraphState) -> GraphState:
    marker = f"summarize: {state['query'][:30]}"
    nvtx.push_range(marker)
 
    # Workers = min(#docs, #cores) for good scaling
 
 
    max_workers = min(len(state["page_texts"]), 1)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        sums = list(pool.map(_lexrank_one, state["page_texts"]))
 
    # sums = _lexrank_one(state["page_texts"])
 
    nvtx.pop_range()
    return {"summaries": sums}
 
 
# def summarize(state: GraphState) -> GraphState:
#     marker = f"summarize: {state['query'][:30]}"
 
#     nvtx.push_range(marker)
#     summarizer = LexRankSummarizer()
#     sums: List[str] = []
#     for t in state['page_texts']:
#         doc = PlaintextParser.from_string(t, Tokenizer('english')).document
#         sents = summarizer(doc, sentences_count=1)
#         sums.append(' '.join(str(s) for s in sents))
#     nvtx.pop_range()
    
  
 
#     return {'summaries': sums}
 
# CPU-only final processing - no LLM inference
def final_processing(state: GraphState) -> GraphState:
    """Final processing step that consolidates results without LLM inference."""
    marker = f"final_processing: {state['query'][:30]}"
    nvtx.push_range(marker)
    
    # Simple concatenation of summaries instead of LLM inference
    combined_summary = "\n".join([f"Summary {i+1}: {summary}" for i, summary in enumerate(state['summaries'])])
    
    # Create a basic response structure
    final_output = f"Query: {state['query']}\n\nCollected {len(state['urls'])} URLs\nProcessed {len(state['page_texts'])} pages\nGenerated {len(state['summaries'])} summaries\n\nCombined summaries:\n{combined_summary}"
    
    nvtx.pop_range()
    return {'final_response': final_output}
 
# 3) Build and compile the graph - removed final_answer node
builder = StateGraph(GraphState)
builder.set_entry_point('web_search')
builder.add_node('web_search', web_search)
builder.add_node('fetch_url', fetch_url)
builder.add_node('summarize', summarize)
# No final_answer node - this makes it CPU-only
builder.add_edge('web_search', 'fetch_url')
builder.add_edge('fetch_url', 'summarize')
builder.set_finish_point('summarize')  # End at summarize instead of final_answer
compiled_graph = builder.compile()
 
# 4) Batch invocation
if __name__ == '__main__':
    batch_size=1
    mini_batch=1
    queries0 = []
    queries1 = []
    queries2 = []
    queries = []
    if len(sys.argv) < 2:
        print("usage: python orchestrator.py <job_id>")
        job_id = 0
    else:
        job_id = int(sys.argv[1])

    for i in range(batch_size):
        queries0.append("What are the projected benefits and challenges of TSMC's 2 nm process based on analyst briefings in 2024?")
 
    for i in range(mini_batch):
        queries1.append("Summarize Apple's vision-OS app ecosystem growth since launch and its impact on developer revenues.")
   
    for i in range(mini_batch):
        queries2.append("What is the outlook for RISC-V adoption based on recent traction among startups and major CPU vendors")
   
 
    queries = queries0
 
 
 
    # "What will be the weather in Santa Clara tomorrow, based on the last 10 days of weather reports?",
    # "What is the outlook for RISC-V adoption based on recent traction among startups and major CPU vendors?",
    # "How is NVIDIA positioning Grace Hopper for data-center AI workloads according to its 2025 GTC keynote?",
    # "Summarize Apple's vision-OS app ecosystem growth since launch and its impact on developer revenues.",
    # "What are the projected benefits and challenges of TSMC's 2 nm process based on analyst briefings in 2024?",
    # "Explain how quantum error-correction schemes evolved in 2023-24 academic literature and what remains unsolved."]
    # "Compare Meta's and Google's approaches to open-sourcing large language models in terms of community adoption."
    # "What trends are shaping battery density improvements in solid-state EV cells, based on 2024 conference papers?",
    # "Assess Microsoft's Copilot integration strategy across Office products and user feedback from early 2025."
# ]
    # queries = ["What is Intel's AI strategy based on 2024 Q1 quarterly earning"]
    initial_states = [
        {'query': q, 'urls': [], 'page_texts': [], 'summaries': []}
        for q in queries[0:batch_size]
    ]
    # time.sleep(50)
    cfg = RunnableConfig(batch_size=batch_size, max_concurrency=mini_batch)
 
    # initial_state: GraphState = {
    # "query": queries,
    # "urls": [],
    # "page_texts": [],
    # "summaries": [],
    # "final_response": ""
    # }
 
    nvtx.push_range('batch_run_all_queries_cpu_only')
 
    # try:
    #     pyRAPL.setup()
    #     s = pyRAPL.sensor.SENSOR
    #     print("Sensor detected:", s)
    # except Exception as e:
    #     print("pyRAPL sensor error →", e)
 
    # with pyRAPL.Measurement('run'):
    start_time = timeit.default_timer()
 
    print(f"{job_id}: [TIMING] start: {start_time:.4f}s")
    result_states = compiled_graph.batch(initial_states, config=cfg)
    elapsed = timeit.default_timer() - start_time
 
    print(f"{job_id}: [TIMING] end: {elapsed:.4f}s")
 
    # result_state = compiled_graph.invoke(initial_state)
    # print(result_state["final_response"])
    nvtx.pop_range()
 
    # # Print results to verify CPU-only execution
    # for i, state in enumerate(result_states):
    #     print(f"\n=== RESULT {i+1} ===")
    #     print(f"🧑 Query: {state['query']}")
    #     print(f"🔗 URLs found: {len(state['urls'])}")
    #     print(f"📄 Pages processed: {len(state['page_texts'])}")
    #     print(f"📝 Summaries generated: {len(state['summaries'])}")
    #     print("📋 Summaries:")
    #     for j, summary in enumerate(state['summaries']):
    #         print(f"  {j+1}. {summary[:100]}...")
    #     print()
 