#!/bin/bash
# Script to run orchestrator with different batch sizes and collect timing statistics

# Parse command line arguments
ROOT="/home/cpu-centric-agentic-ai"

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--root)
            ROOT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-r|--root ROOT_DIR]"
            echo ""
            echo "Options:"
            echo "  -r, --root ROOT_DIR    Set root directory (default: /home/cpu-centric-agentic-ai)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

export ROOT

# Output file for results
OUTPUT_FILE="$ROOT/langchain/batch_timing_results_4c.txt"
LOG_DIR="$ROOT/langchain/batch_logs_4c"
 
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
 
run_parallel() {
    local num_processes=$1
    
    if [ -z "$num_processes" ] || [ "$num_processes" -lt 1 ]; then
        echo "Usage: run_parallel <number_of_processes>"
        echo "Number of processes must be >= 1"
        return 1
    fi
        
    # Run first (num_processes - 1) in background
    for ((i=1; i<num_processes; i++)); do
        python "$ROOT/langchain/orchestrator.py" --skip-web-search --verbose & 
    done
    
    # Run the last one in foreground
    python "$ROOT/langchain/orchestrator.py" --skip-web-search --verbose
}



# Loop through each batch size
for batch_size in "${BATCH_SIZES[@]}"; do
    echo "========================================"
    echo "Running with batch_size=$batch_size"
    echo "========================================"
 

    # Run the experiment and capture output
    LOG_FILE_FULL="$LOG_DIR/batch_${batch_size}_full_log.log"
    LOG_FILE_AVG="$LOG_DIR/batch_${batch_size}_average_log.log"
    echo "Batch Size: $batch_size" >> "$OUTPUT_FILE"
    echo "-------------------" >> "$OUTPUT_FILE"
 
    run_parallel "$batch_size"  2>&1 > "$LOG_FILE_FULL"

    sleep 4

    python "$ROOT/langchain/parse_stats.py" --input_file "$LOG_FILE_FULL" --batch-size "$batch_size" > "$LOG_FILE_AVG" 
 
    # Extract timing statistics from the log and append to results file
    grep -A 10 "TIMING STATISTICS" "$LOG_FILE_AVG" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
 
    echo "Completed batch_size=$batch_size"
    echo ""
 
done
 
echo "========================================"
echo "All experiments completed!"
echo "Results saved to $OUTPUT_FILE"
echo "Logs saved to $LOG_DIR/"
echo "========================================"
 
# Parse results into CSV format for easy plotting
CSV_FILE="$ROOT/langchain/batch_timing_results_4c.csv"
echo "batch_size,stage,count,avg,min,max" > "$CSV_FILE"
 
echo ""
echo "Parsing results to CSV format: $CSV_FILE"
 
# Parse the results file and create CSV
python3 << 'EOF'
import re
 
results_file = "$ROOT/langchain/batch_timing_results_4c.txt"
csv_file = "$ROOT/langchain/batch_timing_results_4c.csv"
 
with open(results_file, 'r') as f:
    content = f.read()
 
# Split by batch size sections
sections = re.split(r'Batch Size: (\d+)', content)
 
csv_lines = []
 
for i in range(1, len(sections), 2):
    batch_size = sections[i]
    section_content = sections[i+1]
 
    # Find the statistics table
    lines = section_content.split('\n')
    for line in lines:
        # Match lines like: "web_search           128        0.0012       0.0008       0.0025"
        match = re.match(r'(\w+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
        if match:
            stage, count, avg, min_val, max_val = match.groups()
            csv_lines.append(f"{batch_size},{stage},{count},{avg},{min_val},{max_val}")
 
# Append to CSV (header already written in bash)
with open(csv_file, 'a') as f:
    for line in csv_lines:
        f.write(line + '\n')
 
print(f"CSV file created: {csv_file}")
print(f"Total records: {len(csv_lines)}")
EOF
 
echo ""
echo "Done! You can now run the plotting script."
 
 