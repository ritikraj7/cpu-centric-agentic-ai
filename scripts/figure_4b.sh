conda activate main

# Update root directory as per your file structure
export ROOT=/home/cpu-centric-agentic-ai

# Update HF_HOME env variable as per your hugging face home location
export HF_HOME=/storage/hugging_face/

MODEL=openai/gpt-oss-20b
GPU=0

echo "Starting vLLM server for gpt-oss-20b ..."

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

bash "$ROOT/langchain/run_batch_experiment.sh" -r "$ROOT"

conda deactivate

conda activate main

echo "Running haystack workload ..."

python "$ROOT/haystack/benchmark_batch_parallel_retrieval.py" --query-file "$ROOT/haystack/queries.txt" --output-file "$ROOT/haystack/figure_4b.json" --store-dir /storage/rag_flat_store

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

python "$ROOT/toolformer/batch_toolformer.py" --output "$ROOT/toolformer/benchmark_results_4b.json" > "$ROOT/toolformer/throughput_log.txt"


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
bash "$ROOT/mini-swe-agent/run_batch_experiment.sh"


kill -TERM "$(cat vllm.pid)"

wait "$(cat vllm.pid)" 2>/dev/null || true

rm -f vllm.pid

echo "Experiments done ... Plotting figure now"

python "$ROOT/plot_agentic_throughput.py" --base-dir "$ROOT" --output "$ROOT/figures/figure_4b.png"