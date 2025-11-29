"""
Simple Timing Statistics Parser
"""

import re
import argparse
import sys
from collections import defaultdict

def parse_timing_file(file_path):
    """Simple parser for timing statistics"""
    stage_data = defaultdict(list)
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}
    
    # Find all timing statistics tables
    # Look for lines that match the pattern: stage_name number number number number
    pattern = r'^(\w+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    
    for line in content.split('\n'):
        line = line.strip()
        match = re.match(pattern, line)
        if match:
            stage = match.group(1)
            count = int(match.group(2))
            avg = float(match.group(3))
            min_val = float(match.group(4))
            max_val = float(match.group(5))
            
            stage_data[stage].append((count, avg, min_val, max_val))
    
    return stage_data

def aggregate_stats(stage_data):
    """Aggregate statistics across all entries"""
    result = {}
    
    for stage, entries in stage_data.items():
        if not entries:
            continue
        
        total_count = sum(e[0] for e in entries)
        avg_time = sum(e[1] for e in entries) / len(entries)
        min_time = min(e[2] for e in entries)
        max_time = max(e[3] for e in entries)
        
        result[stage] = (total_count, avg_time, min_time, max_time)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Parse timing statistics")
    parser.add_argument('--input_file', type=str, help='Input file')
    parser.add_argument('--batch-size', '-b', default=1,type=int, help='Batch size')
    
    args = parser.parse_args()
    
    # Parse the file
    stage_data = parse_timing_file(args.input_file)
    
    if not stage_data:
        print("No timing statistics found in the input file.")
        return
    
    # Aggregate the data
    aggregated = aggregate_stats(stage_data)
    
    # Print results
    print("=" * 70)
    print("TIMING STATISTICS")
    if args.batch_size:
        print(f"(Batch Size: {args.batch_size})")
    print("=" * 70)
    print(f"{'Stage':<20} {'Count':<10} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
    print("-" * 70)
    
    for stage in sorted(aggregated.keys()):
        count, avg, min_val, max_val = aggregated[stage]
        print(f"{stage:<20} {count:<10} {avg:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()