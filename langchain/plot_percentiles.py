import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

# Case 1: All jobs start at t=0 and finish at times in test.txt
def parse_case1(filename):
    """Parse timing file - extract completion times from [TIMING] end lines"""
    completion_times = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '[TIMING] end:' in line:
                # Parse format like "1: [TIMING] end: 6.3151s"
                # Use regex to extract the time value
                match = re.search(r'\[TIMING\] end:\s*([\d.]+)s', line)
                if match:
                    time_value = float(match.group(1))
                    completion_times.append(time_value)
    return sorted(completion_times)

 
# Case 2: Jobs are scheduled on 64 cores with queueing
def parse_case2(filename):
    """Parse results.txt - contains start/end timing events"""
    # Parse events with their line numbers to preserve ordering
    events = []
 
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
 
            parts = line.split(': ')
            job_id = int(parts[0])
 
            if '[TIMING] start' in line:
                time_str = parts[2].rstrip('s')
                events.append({
                    'line': line_num,
                    'job_id': job_id,
                    'type': 'start',
                    'time': float(time_str)
                })
            elif '[TIMING] end' in line:
                duration_str = parts[2].rstrip('s')
                events.append({
                    'line': line_num,
                    'job_id': job_id,
                    'type': 'end',
                    'duration': float(duration_str)
                })
 
    # Find the global minimum start time as t=0 reference
    min_start_time = min(e['time'] for e in events if e['type'] == 'start')
 
    # Match each end event with its corresponding start event
    # Track all unmatched starts for each job_id as a queue
    from collections import deque
    active_jobs = {}  # job_id -> deque of start times
    completion_times = []
 
    for event in events:
        job_id = event['job_id']
 
        if event['type'] == 'start':
            # Add this start time to the queue for this job_id
            if job_id not in active_jobs:
                active_jobs[job_id] = deque()
            active_jobs[job_id].append(event['time'])
        elif event['type'] == 'end':
            # Match with the oldest unmatched start for this job_id (FIFO)
            if job_id in active_jobs and len(active_jobs[job_id]) > 0:
                start_time = active_jobs[job_id].popleft()
                duration = event['duration']
                # Completion time relative to t=0
                completion_time = (start_time - min_start_time) + duration
                completion_times.append(completion_time)
 
    return sorted(completion_times)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Plot job completion percentiles comparing baseline and CGAM'
    )
    parser.add_argument(
        '--case1',
        type=str,
        default='/home/cpu-centric-agentic-ai/langchain/baseline_7a.txt',
        help='Input file for case 1 (baseline) (default: /home/cpu-centric-agentic-ai/langchain/baseline_7a.txt)'
    )
    parser.add_argument(
        '--case2',
        type=str,
        default='/home/cpu-centric-agentic-ai/langchain/cgam_7a.txt',
        help='Input file for case 2 (CGAM) (default: /home/cpu-centric-agentic-ai/langchain/cgam_7a.txt)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='/home/cpu-centric-agentic-ai/figures/figure_7a_cgam.png',
        help='Output path for the generated plot (default: /home/cpu-centric-agentic-ai/figures/figure_7a_cgam.png)'
    )

    args = parser.parse_args()

    # Parse both cases
    case1_times = parse_case1(args.case1)
    case2_times = parse_case2(args.case2)

    # Calculate percentiles
    def calculate_percentiles(completion_times):
        """Calculate percentiles for completed jobs"""
        n_jobs = len(completion_times)
        percentiles = [(i + 1) / n_jobs * 100 for i in range(n_jobs)]
        return percentiles, completion_times

    percentiles1, times1 = calculate_percentiles(case1_times)
    percentiles2, times2 = calculate_percentiles(case2_times)

    # Create the plot
    plt.figure(figsize=(14, 8))

    plt.plot(percentiles1, times1, 'b-', linewidth=4, label='Baseline: All 128 Multi-Processing')
    plt.plot(percentiles2, times2, 'r-', linewidth=4, label='$\\mathbf{\ Ours:\ CGAM\ with\ Bcap=64}$')


    # Get P50 values (50th percentile = 64th job out of 128, which is index 63)
    p50_case1 = times1[63]  # 64th job (0-indexed)
    p50_case2 = times2[63]  # 64th job (0-indexed)

    # # Add horizontal lines at P50
    # plt.axhline(y=p50_case1, color='b', linestyle='--', alpha=0.5, linewidth=1.5)
    # plt.axhline(y=p50_case2, color='r', linestyle='--', alpha=0.5, linewidth=1.5)

    # Add vertical line at 50th percentile
    plt.axvline(x=50, color='gray', linestyle=':', alpha=0.4, linewidth=1)

    # Annotate P50 for Case 1
    plt.annotate(f'P50 = {p50_case1:.2f}s',
                 xy=(50, p50_case1),
                 xytext=(25, p50_case1 + 0.3),
                 fontsize=28,
                 color='blue',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.1', facecolor='lightblue', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # Annotate P50 for Case 2
    plt.annotate(f'P50 = {p50_case2:.2f}s',
                 xy=(50, p50_case2),
                 xytext=(70, p50_case2 + 0.5),
                 fontsize=28,
                 color='red',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.1', facecolor='lightcoral', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Calculate and annotate the improvement
    improvement_pct = ((p50_case1 - p50_case2) / p50_case1) * 100
    speedup = p50_case1 / p50_case2

    # Add a double-headed arrow from red P50 to blue P50
    plt.annotate('',
                 xy=(50, p50_case1),
                 fontsize=14,
                 xytext=(50, p50_case2),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=4))

    # Add the improvement text box (original position)
    plt.text(55, (p50_case1 + p50_case2) / 2,
             f'P50 Speedup: {speedup:.2f}x',
             fontsize=30,
             color='green',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
             verticalalignment='center')

    plt.xlabel('Percentile of Jobs Completed (%)', fontsize=32, fontweight='bold')
    plt.ylabel('Latency (s)', fontsize=32, fontweight='bold')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=25, loc='lower right')


    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{args.output}'")

    # Print summary statistics
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")
    print(f"\nCase 1 (All jobs start at t=0):")
    print(f"  Total jobs: {len(case1_times)}")
    print(f"  Min completion time: {min(case1_times):.4f}s")
    print(f"  P50 completion time: {p50_case1:.4f}s")
    print(f"  Max completion time: {max(case1_times):.4f}s")
    print(f"  Median completion time: {np.median(case1_times):.4f}s")

    print(f"\nCase 2 (64 cores with queueing):")
    print(f"  Total jobs: {len(case2_times)}")
    print(f"  Min completion time: {min(case2_times):.4f}s")
    print(f"  P50 completion time: {p50_case2:.4f}s")
    print(f"  Max completion time: {max(case2_times):.4f}s")
    print(f"  Median completion time: {np.median(case2_times):.4f}s")

    print(f"\n{'='*70}")
    print("P50 Latency Comparison:")
    print(f"{'='*70}")
    print(f"  Case 1 P50: {p50_case1:.4f}s")
    print(f"  Case 2 P50: {p50_case2:.4f}s")
    print(f"  Improvement: {improvement_pct:.2f}% faster")
    print(f"  P50 Speedup: {speedup:.2f}x")

    max_speedup = max(case1_times) / max(case2_times)
    print(f"\n{'='*70}")
    print("Overall Comparison:")
    print(f"{'='*70}")
    print(f"  P50 Speedup (Case 2 vs Case 1): {speedup:.2f}x")
    print(f"  Max time Speedup: {max_speedup:.2f}x")
    print(f"{'='*70}")

    plt.show()

if __name__ == '__main__':
    main()