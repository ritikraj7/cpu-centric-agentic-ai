# Update root directory as per your file structure
export ROOT=/home/cpu-centric-agentic-ai

# Update HF_HOME env variable as per your hugging face home location
export HF_HOME=/storage/hugging_face/

conda activate main

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

echo "vLLM server started. Running baseline ..."

conda deactivate

conda activate langchain

bash "$ROOT/langchain/bash_parallel.sh" -o "$ROOT/langchain/orchestrator.py" > "$ROOT/langchain/baseline_7a.txt"

echo "Running cgam ..."

# Generate jobs.txt with the correct ROOT path
JOBS_FILE="$ROOT/langchain/jobs.txt"
> "$JOBS_FILE"  # Clear the file

for i in {1..64}; do
    echo "python $ROOT/langchain/orchestrator.py --skip-web-search --job-id $i" >> "$JOBS_FILE"
done

cat "$ROOT/langchain/jobs.txt" "$ROOT/langchain/jobs.txt" | xargs -P 64 -n 1 -I{} bash -c "{}" > "$ROOT/langchain/cgam_7a.txt"
echo "Running cgam_overlap ..."

bash "$ROOT/langchain/cgam_overlap.sh" -r "$ROOT"

cat "$ROOT/langchain/cgam_7a_o1.txt" "$ROOT/langchain/cgam_7a_o2.txt" > "$ROOT/langchain/cgam_7a_overlap.txt"

echo "Plotting figures ..."

python "$ROOT/langchain/plot_percentiles.py" --case1 "$ROOT/langchain/baseline_7a.txt" --case2 "$ROOT/langchain/cgam_7a.txt" --output "$ROOT/figures/figure_7a.png"
python "$ROOT/langchain/plot_percentiles_overlap.py" --case1 "$ROOT/langchain/baseline_7a.txt" --case2 "$ROOT/langchain/cgam_7a_overlap.txt" --output "$ROOT/figures/figure_7a_overlap.png" 

kill -TERM "$(cat vllm.pid)"

wait "$(cat vllm.pid)" 2>/dev/null || true

rm -f vllm.pid

