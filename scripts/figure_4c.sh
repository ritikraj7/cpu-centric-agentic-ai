conda activate main

# Update root directory as per your file structure
export ROOT=/home/cpu-centric-agentic-ai

# Update HF_HOME env variable as per your hugging face home location
export HF_HOME=/storage/hugging_face/

MODEL=openai/gpt-oss-20b
GPU=0

echo "Starting vLLM server ..."

export HF_HOME=/storage/ritikraj/hugging_face/
CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" --no-enable-prefix-caching --port 5000 > vllm.log 2>&1 &
echo $! > vllm.pid



if ! timeout 60 bash -c 'until curl -sf http://localhost:5000/health > /dev/null; do
  sleep 1
done'; then
  echo "vLLM did not become healthy in time. Check logs: vllm.log"
  exit 1
fi

echo "vLLM server started. Running experiment ..."

conda deactivate

conda activate langchain

bash "$ROOT/langchain/run_batch_experiment_verbose.sh" -r "$ROOT" 

echo "Experiment completed. Plotting figure ..."

python "$ROOT/langchain/plot_error_bar.py" -i "$ROOT/langchain/batch_timing_results_4c.csv" -o "$ROOT/figures/figure_4c.png"
kill -TERM "$(cat vllm.pid)"

wait "$(cat vllm.pid)" 2>/dev/null || true

rm -f vllm.pid

