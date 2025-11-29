OUTPUT_FILE="gpu_power_baseline.csv"
  
# Continuously append metrics every 100 ms (-lms 100)
# You can adjust the interval (in milliseconds) if needed

nvidia-smi -i 0 \
  --query-gpu=timestamp,power.draw \
  --format=csv \
  -lms 100 >> "$OUTPUT_FILE"