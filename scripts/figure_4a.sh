conda activate main

# Update root directory as per your file structure
export ROOT=/home/cpu-centric-agentic-ai

# Update HF_HOME env variable as per your hugging face home location
export HF_HOME=/storage/hugging_face/

MODEL=openai/gpt-oss-20b
GPU=0

echo "Starting vLLM server ..."

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" --no-enable-prefix-caching --port 5000 > vllm.log 2>&1 &
echo $! > vllm.pid



if ! timeout 60 bash -c 'until curl -sf http://localhost:5000/health > /dev/null; do
  sleep 1
done'; then
  echo "vLLM did not become healthy in time. Check logs: vllm.log"
  exit 1
fi

echo "vLLM server started. Running throughput experiment ..."

python "$ROOT/throughput.py"

echo "Throughput experiment completed. Plotting figure ..."

python "$ROOT/plot_throughput.py" --input "$ROOT/benchmark_results.json" --output "$ROOT/figures/figure_4a.png"
kill -TERM "$(cat vllm.pid)"

wait "$(cat vllm.pid)" 2>/dev/null || true

rm -f vllm.pid