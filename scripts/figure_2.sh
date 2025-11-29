# This automated script creates vllm servers, runs the benchmarks and plots figure 2 of the paper.
# Note: It excludes DS-1000 benchmark 

conda activate main

# Update root directory as per your file structure
export ROOT=/home/cpu-centric-agentic-ai

# Update HF_HOME env variable as per your hugging face home location
export HF_HOME=/storage/hugging_face/

# Default model, can be updated as needed
MODEL=openai/gpt-oss-20b

# Default GPU ID
GPU=0

echo "Starting vLLM server for gpt-oss-20b ..."

# By default, the script uses port = 5000
CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" --no-enable-prefix-caching --port 5000 > vllm.log 2>&1 &
echo $! > vllm.pid



if ! timeout 60 bash -c 'until curl -sf http://localhost:5000/health > /dev/null; do
  sleep 1
done'; then
  echo "vLLM did not become healthy in time. Check logs: vllm.log"
  exit 1
fi

echo "vLLM server started. Running langchain workload ..."

conda deactivate

conda activate langchain

# Add your Google Search API keys
export GOOGLE_CX=<>
export GOOGLE_API_KEY=<>

python "$ROOT/langchain/orchestrator.py" --benchmark freshQA --verbose > "$ROOT/langchain/latency_figure2_temp4.txt"
python "$ROOT/langchain/orchestrator.py" --benchmark musique --verbose >> "$ROOT/langchain/latency_figure2_temp4.txt"
python "$ROOT/langchain/orchestrator.py" --benchmark QASC --verbose >> "$ROOT/langchain/latency_figure2_temp4.txt"


conda deactivate

conda activate main

echo "Running haystack workload ..."

python "$ROOT/haystack/retrieval.py" query-rag --store-dir /storage/rag_flat_store --question "When was Albert Einstein born?" > "$ROOT/haystack/latency_figure2_temp.txt"
python "$ROOT/haystack/retrieval.py" query-rag --store-dir /storage/rag_flat_store --question "Which year was the scientist who developed E=mc^2 born?" >> "$ROOT/haystack/latency_figure2_temp.txt"
python "$ROOT/haystack/retrieval.py" query-rag --store-dir /storage/rag_flat_store --question "What is Einstein's most famous equation?" >> "$ROOT/haystack/latency_figure2_temp.txt"

kill -TERM "$(cat vllm.pid)"

wait "$(cat vllm.pid)" 2>/dev/null || true

rm -f vllm.pid


MODEL2=EleutherAI/gpt-j-6b

echo "Starting vLLM server for gpt-j-6b model ..."

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL2" --no-enable-prefix-caching --port 5000 >> vllm.log 2>&1 &
echo $! > vllm.pid



if ! timeout 90 bash -c 'until curl -sf http://localhost:5000/health > /dev/null; do
  sleep 1
done'; then
  echo "vLLM did not become healthy in time. Check logs: vllm.log"
  exit 1
fi
echo "vLLM server started. Running toolformer workload ..."

# ADD your Wolfram Alpha API key
export WOLFRAM_ALPHA_APPID=<>
python "$ROOT/toolformer/math_toolformer.py" > "$ROOT/toolformer/latency_log.txt"


kill -TERM "$(cat vllm.pid)"

wait "$(cat vllm.pid)" 2>/dev/null || true

rm -f vllm.pid

sleep 5

MODEL3=Qwen/Qwen2.5-Coder-32B-Instruct

echo "Starting vLLM server for Qwen2.5-Coder-32B-Instruct ..."

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL3" --no-enable-prefix-caching --port 5000 >> vllm.log 2>&1 &
echo $! > vllm.pid



if ! timeout 90 bash -c 'until curl -sf http://localhost:5000/health > /dev/null; do
  sleep 1
done'; then
  echo "vLLM did not become healthy in time. Check logs: vllm.log"
  exit 1
fi

echo "vLLM server started. Running mini-swe-agent workload ..."
conda deactivate
conda activate swe
python "$ROOT/mini-swe-agent/benchmark_latency.py" --output-dir "$ROOT/mini-swe-agent/benchmark_results_temp" --benchmark-type sorting
python "$ROOT/mini-swe-agent/benchmark_latency.py" --output-dir "$ROOT/mini-swe-agent/benchmark_results_temp" --benchmark-type integration

kill -TERM "$(cat vllm.pid)"

wait "$(cat vllm.pid)" 2>/dev/null || true

rm -f vllm.pid
sleep 5     
echo "Experiments done ... Plotting figure now"

python "$ROOT/plot_latency.py" --output "$ROOT/figures/figure_2.png" --base-dir "$ROOT"