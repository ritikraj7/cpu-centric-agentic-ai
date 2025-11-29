"""
Batch web-LLM orchestrator using LangGraph batching with per-query NVTX markers.
Accepts multiple queries as CLI args, runs the full tool chain in a single batched graph invocation, and marks each node per query.
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
import sys
import argparse
 
import time
from collections import defaultdict
 
# Global timing storage for statistics
timing_stats = defaultdict(list)
 
# 1) Shared state schema
typedef = TypedDict  # compatibility alias
class GraphState(typedef('GraphState', {})):
    query: str
    urls: List[str]
    page_texts: List[str]
    summaries: List[str]
    final_response: str
    job_id: int
    skip_web_search: bool
 
# 2) Tool implementations with dynamic NVTX markers
 
def web_search(state: GraphState) -> GraphState:
    marker = f"web_search: {state['query'][:30]}"
    nvtx.push_range(marker)
    start_time = timeit.default_timer()

    if state['skip_web_search'] == False:
        key, cx = os.getenv('GOOGLE_API_KEY'), os.getenv('GOOGLE_CX')
        if not key or not cx:
            nvtx.pop_range(); raise RuntimeError('Missing GOOGLE_API_KEY or GOOGLE_CX')
        params = {'key': key, 'cx': cx, 'q': state['query'], 'num': 10}
        resp = requests.get('https://www.googleapis.com/customsearch/v1', params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get('items', [])
        urls = [i['link'] for i in items if 'link' in i]
    else:
        urls = ['https://en.wikipedia.org/wiki/Spiel_des_Jahres', 'https://boardgamegeek.com/wiki/page/Spiel_des_Jahres', 'https://www.reddit.com/r/boardgames/comments/buwap5/are_previous_spiel_des_jahres_winners_now_too/', 'https://boardgamegeek.com/thread/3282083/spiel-des-jahres-winners-1979-to-2023-and-who-do-y', 'https://blog.recommend.games/posts/thoughts-on-spiel-des-jahres/', 'https://www.spiel-des-jahres.de/en/award-winners-2024/', 'https://www.facebook.com/groups/132851767828/posts/10162746926537829/', 'https://www.tabletopgaming.co.uk/news/spiel-des-jahres-2024-winners-announced/', 'https://therewillbe.games/board-game-lists-and-guides/6214-the-ten-greatest-spiel-des-jahres-winners', 'https://www.dicebreaker.com/topics/spiel-des-jahres/best-games/overlooked-spiel-des-jahres-winners']
 

    elapsed = timeit.default_timer() - start_time
    timing_stats['web_search'].append(elapsed)
    # print(f"[TIMING] {marker}: {elapsed:.4f}s")
    nvtx.pop_range()
    return {'urls': urls}
 
 
 
def _fetch_single(url: str, timeout: float = 10.0) -> Optional[str]:
    """Download one URL and return plain-text, or None on error."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser").get_text(separator="\n")
    except requests.RequestException:
        return None
 
 
 
def _fetch_url_single_state(state: GraphState) -> GraphState: 
    """Sequential download of up to two pages for *one* query."""
    marker = f"fetch_url: {state['query'][:30]}"
    nvtx.push_range(marker)
    start_time = timeit.default_timer()
 
    texts: List[str] = []
    for url in state["urls"]:
        if len(texts) >= 2:
            break
        txt = _fetch_single(url)
        if txt:
            texts.append(txt)
 
    elapsed = timeit.default_timer() - start_time
    timing_stats['fetch_url'].append(elapsed)
    # print(f"[TIMING] {marker}: {elapsed:.4f}s")
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
    start_time = timeit.default_timer() 
 
    max_workers = min(len(state["page_texts"]), 1)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        sums = list(pool.map(_lexrank_one, state["page_texts"]))
 
 
    elapsed = timeit.default_timer() - start_time
    timing_stats['summarize'].append(elapsed)
    # print(f"[TIMING] {marker}: {elapsed:.4f}s")
    nvtx.pop_range()
    return {"summaries": sums}

 
 
def final_answer(state: GraphState) -> GraphState:
    marker = f"llm_inference: {state['query'][:30]}"
    nvtx.push_range(marker)
    start_time = timeit.default_timer()
    from langchain_community.llms import VLLMOpenAI
    llm = VLLMOpenAI(
        base_url='http://localhost:5000/v1',
        model="openai/gpt-oss-20b",
        openai_api_key='EMPTY'  # no API key required for local VLLM
 
    )
    prompt = f"Based on these summaries, answer: {state['query']}\n\n" + "\n\n".join(state['summaries'])
    answer = llm.invoke([prompt])
    elapsed = timeit.default_timer() - start_time
    timing_stats['llm_inference'].append(elapsed)
    # print(f"[TIMING] {marker}: {elapsed:.4f}s")
    nvtx.pop_range()
    return {'final_response': answer}
 
def print_timing_statistics():
    """Print average, min, and max time for each stage"""
    print("\n" + "="*70)
    print("TIMING STATISTICS (across all batches)")
    print("="*70)
    print(f"{'Stage':<20} {'Count':<10} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
    print("-"*70)
 
    for stage in ['web_search', 'fetch_url', 'summarize', 'llm_inference']:
        if stage in timing_stats and timing_stats[stage]:
            times = timing_stats[stage]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            count = len(times)
            print(f"{stage:<20} {count:<10} {avg_time:<12.4f} {min_time:<12.4f} {max_time:<12.4f}")
        else:
            print(f"{stage:<20} {'0':<10} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
 
    print("="*70 + "\n")
 
# 3) Build and compile the graph
builder = StateGraph(GraphState)
builder.set_entry_point('web_search')
builder.add_node('web_search', web_search)
builder.add_node('fetch_url', fetch_url)
builder.add_node('summarize', summarize)
builder.add_node('final_answer', final_answer)
builder.add_edge('web_search', 'fetch_url')
builder.add_edge('fetch_url', 'summarize')
builder.add_edge('summarize', 'final_answer')
builder.set_finish_point('final_answer')
compiled_graph = builder.compile()
 
# 4) Batch invocation
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', action='store_true', help='Enable output of per-stage latencies')
    parser.add_argument('--skip-web-search', action='store_true', help='Skip web search stage, only for FreshQA benchmark')
    parser.add_argument('--sequential', action='store_true', help='Run multiple batches sequentially')
    parser.add_argument('--batch-size', type=int, default=1, help="Langchain batch size")
    parser.add_argument('--job-id', type=int, default=1, help="Job id for bash multiprocessing")
    parser.add_argument('--benchmark', choices=["freshQA", "musique", "QASC"], default="freshQA")


    args = parser.parse_args()


    batch_size=args.batch_size
    if args.sequential:
        mini_batch = 1
    else:
        mini_batch = batch_size
    job_id = args.job_id

    if args.benchmark == "freshQA":
        query_single = "Which game won the Spiel des Jahres award most recently?"
    elif args.benchmark == "musique":
        query_single = "When did the people who captured Malakoff come to the region where Philipsburg is located?"
    elif args.benchmark == "QASC":
        query_single = "Differential heating of air can be harnessed for what?"
    else:
        print("Wrong benchmark choice. Choose among 'freshQA', 'musique' and 'QASC'.")
        sys.exit()
    

    queries = []


    for i in range(batch_size):
        queries.append(query_single)
 
 
    initial_states = [
        {'query': q, 'urls': [], 'page_texts': [], 'summaries': [], 'final_response': '', 'job_id': job_id, 'skip_web_search': args.skip_web_search}
        for q in queries[0:batch_size]
    ]

    cfg = RunnableConfig(batch_size=batch_size, max_concurrency=mini_batch)
 
 
    nvtx.push_range('batch_run_all_queries')
    start_time = timeit.default_timer()
 
    print(f"{job_id}: [TIMING] start: {start_time:.4f}s")

    result_states = compiled_graph.batch(initial_states, config=cfg)
    # print(result_state["final_response"])
    elapsed = timeit.default_timer() - start_time
    print(f"{job_id}: [TIMING] end: {elapsed:.4f}s")
    nvtx.pop_range()
 
    # Print timing statistics
    if args.verbose:
        print_timing_statistics()
 
    # for state in result_states:
    #     print(f"🧑 » {state['query']}")
    #     print(f"🤖 » {state['final_response']}\n")
 
 