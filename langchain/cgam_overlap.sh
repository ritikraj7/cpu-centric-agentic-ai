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

# Generate jobs.txt with the correct ROOT path
JOBS_FILE="$ROOT/langchain/jobs.txt"
> "$JOBS_FILE"  # Clear the file

for i in {1..64}; do
    echo "python $ROOT/langchain/orchestrator.py --skip-web-search --job-id $i" >> "$JOBS_FILE"
done

cat "$JOBS_FILE" | xargs -P 64 -n 1 -I{} bash -c "{}" > "$ROOT/langchain/cgam_7a_o1.txt" &
sleep 2.5 # Need to tune based on the system to ensure overlap
cat "$JOBS_FILE" | xargs -P 64 -n 1 -I{} bash -c "{}" > "$ROOT/langchain/cgam_7a_o2.txt"