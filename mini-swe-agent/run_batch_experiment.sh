 
# Output file for results
OUTPUT_FILE="/home/cpu-centric-agentic-ai/mini-swe-agent/batch_timing_results_4b.txt"
LOG_DIR="/home/cpu-centric-agentic-ai/mini-swe-agent/batch_logs_4b"
 
# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"
 
# Clear previous results
> "$OUTPUT_FILE"
 
# Array of batch sizes to test
BATCH_SIZES=(1 2 4 8 16 32 64 128)
 
echo "Starting batch size experiments..."
echo "Results will be saved to $OUTPUT_FILE"
echo "Individual logs will be saved to $LOG_DIR/"
echo ""
sed -i 's/timeout: int = 30/timeout: int = 60/' /home/cpu-centric-agentic-ai/mini-swe-agent/src/minisweagent/environments/local.py
 
run_parallel() {
    local num_processes=$1
    
    if [ -z "$num_processes" ] || [ "$num_processes" -lt 1 ]; then
        echo "Usage: run_parallel <number_of_processes>"
        echo "Number of processes must be >= 1"
        return 1
    fi
        
    # Run first (num_processes - 1) in background
    for ((i=1; i<num_processes; i++)); do
        python /home/cpu-centric-agentic-ai/mini-swe-agent/benchmark_latency.py --benchmark-type sorting --output-dir /home/cpu-centric-agentic-ai/mini-swe-agent/temp/tempf_$num_processes_$i --no-print & 
    done
    
    # Run the last one in foreground
    python /home/cpu-centric-agentic-ai/mini-swe-agent/benchmark_latency.py --benchmark-type sorting --output-dir /home/cpu-centric-agentic-ai/mini-swe-agent/temp/tempf_$num_processes_$num_processes --no-print
}



# Loop through each batch size
for batch_size in "${BATCH_SIZES[@]}"; do
    echo "========================================"
    echo "Running with batch_size=$batch_size"
    echo "========================================"
 
    echo "Batch Size: $batch_size" >> "$OUTPUT_FILE"
    echo "-------------------" >> "$OUTPUT_FILE"
 
    run_parallel "$batch_size"  2>&1 >> "$OUTPUT_FILE"

    sleep 4
 
    echo "Completed batch_size=$batch_size"
    echo ""
 
done
sed -i 's/timeout: int = 60/timeout: int = 30/' /home/cpu-centric-agentic-ai/mini-swe-agent/src/minisweagent/environments/local.py

echo "========================================"
echo "All experiments completed!"
echo "Results saved to $OUTPUT_FILE"
echo "========================================"
