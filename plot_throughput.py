#!/usr/bin/env python3
"""
Plot token throughput benchmark results from JSON data.
Calculates tokens/s using: batch_size * (input_tokens + output_tokens) / batch_time
"""
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import argparse
import numpy as np
def thousands_formatter(x, pos):
    if x >= 1000:
        return f'{int(x/1000)}k'
    else:
        return f'{int(x)}' 
def plot_throughput_from_json(json_file: str, output_file: str = None):
    """Plot throughput results from JSON file"""
 
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
 
    # Extract results array from JSON
    results = data['results'] if 'results' in data else data
 
    # Filter successful results only
    results = [r for r in results if r.get('successful', True)]
 
    # Extract data and calculate tokens/s
    batch_sizes = [r['batch_size'] for r in results]
    input_tokens = [r['input_tokens'] for r in results]
    output_tokens = [r['output_tokens'] for r in results]
    batch_times = [r['batch_time'] for r in results]
 
    # Calculate tokens/s: batch_size * (input_tokens + output_tokens) / batch_time
    tokens_per_sec = [
        bs * (inp + out) / bt
        for bs, inp, out, bt in zip(batch_sizes, input_tokens, output_tokens, batch_times)
    ]
    
    # Create single plot: Tokens/s vs Batch Size
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
 
    # Get unique input and output token values
    unique_input_tokens = sorted(set(input_tokens))
    unique_output_tokens = sorted(set(output_tokens))
 
    # Define colors for input tokens and markers for output tokens
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_input_tokens)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:len(unique_output_tokens)]
 
    # Create color and marker mappings
    color_map = {inp: colors[i] for i, inp in enumerate(unique_input_tokens)}
    marker_map = {out: markers[i] for i, out in enumerate(unique_output_tokens)}
 
    # Get unique combinations of input and output tokens
    token_combinations = sorted(set(zip(input_tokens, output_tokens)))
 
    # Plot a line for each input/output token combination
    for inp, out in token_combinations:
        # Find all results with this input/output combination
        indices = [i for i in range(len(results))
                   if input_tokens[i] == inp and output_tokens[i] == out]
 
        # Get batch sizes and tokens/s for this combination
        x_vals = [batch_sizes[i] for i in indices]
        y_vals = [tokens_per_sec[i] for i in indices]
 
        # Sort by batch size for proper line plotting
        sorted_pairs = sorted(zip(x_vals, y_vals))
        x_vals_sorted = [x for x, y in sorted_pairs]
        y_vals_sorted = [y for x, y in sorted_pairs]
 
        # Plot with consistent color (by input) and marker (by output)
        ax.plot(x_vals_sorted, y_vals_sorted, marker=marker_map[out], linestyle='-',
                linewidth=2, markersize=8, color=color_map[inp], alpha=0.7)
 
    ax.set_xlabel('Batch Size', fontsize=22, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_ylabel('Throughput (tokens/s)', fontsize=22, fontweight='bold')
    # ax.set_title('Token Throughput vs Batch Size', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # Create separate legends for input and output tokens
    from matplotlib.lines import Line2D


    # Apply the formatter
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    # Legend for input tokens (colors)
    input_legend_elements = [Line2D([0], [0], color=color_map[inp], linewidth=2,
                                     label=f'Input: {inp}')
                             for inp in unique_input_tokens]
 
    # Legend for output tokens (markers)
    output_legend_elements = [Line2D([0], [0], marker=marker_map[out], color='gray',
                                      linestyle='None', markersize=8,
                                      label=f'Output: {out}')
                              for out in unique_output_tokens]
 
    # Combine both legend elements with a separator
    combined_legend = input_legend_elements + output_legend_elements
 
    # Add combined legend
    ax.legend(handles=combined_legend, loc='best',
              fontsize=22, ncol=2, framealpha=0.5, columnspacing=0.2)
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig
 
def print_analysis(json_file: str):
    """Print detailed analysis of the results"""
 
    with open(json_file, 'r') as f:
        data = json.load(f)
 
    # Extract results array from JSON
    results = data['results'] if 'results' in data else data
 
    # Filter successful results only
    results = [r for r in results if r.get('successful', True)]
 
    print("=" * 90)
    print("🚀 TOKEN THROUGHPUT BENCHMARK ANALYSIS")
    print("=" * 90)
 
    # Print metadata if available
    if 'timestamp' in data:
        print(f"Timestamp: {data['timestamp']}")
    if 'model_path' in data:
        print(f"Model: {data['model_path'].split('/')[-1]}")
    print()
 
    # Results table
    print(f"{'Batch Size':<12} {'Input':<10} {'Output':<10} {'Batch Time':<12} {'Tokens/s':<15} {'Success':<10}")
    print("-" * 90)
 
    for result in results:
        batch_size = result['batch_size']
        inp_tokens = result['input_tokens']
        out_tokens = result['output_tokens']
        batch_time = result['batch_time']
        tokens_per_sec = batch_size * (inp_tokens + out_tokens) / batch_time
        success = "✓" if result.get('successful', True) else "✗"
 
        print(f"{batch_size:<12} {inp_tokens:<10} {out_tokens:<10} {batch_time:<12.2f} "
              f"{tokens_per_sec:<15.2f} {success:<10}")
 
    # Calculate tokens/s for analysis
    tokens_per_sec_list = [
        r['batch_size'] * (r['input_tokens'] + r['output_tokens']) / r['batch_time']
        for r in results
    ]
 
    max_throughput = max(tokens_per_sec_list)
    max_idx = tokens_per_sec_list.index(max_throughput)
    min_throughput = min(tokens_per_sec_list)
    avg_throughput = sum(tokens_per_sec_list) / len(tokens_per_sec_list)
 
    print(f"\n🎯 Key Insights:")
    print(f"   • Best throughput: {max_throughput:.2f} tokens/s")
    print(f"     - Batch size: {results[max_idx]['batch_size']}")
    print(f"     - Input/Output tokens: {results[max_idx]['input_tokens']}/{results[max_idx]['output_tokens']}")
    print(f"     - Batch time: {results[max_idx]['batch_time']:.2f}s")
    print(f"   • Worst throughput: {min_throughput:.2f} tokens/s")
    print(f"   • Average throughput: {avg_throughput:.2f} tokens/s")
    print(f"   • Range: {max_throughput - min_throughput:.2f} tokens/s ({(max_throughput/min_throughput - 1)*100:.1f}% variation)")
 
    # Group by batch size for analysis
    batch_sizes = sorted(set(r['batch_size'] for r in results))
    print(f"\n📊 Throughput by Batch Size:")
    for bs in batch_sizes:
        bs_results = [r for r in results if r['batch_size'] == bs]
        bs_throughputs = [
            bs * (r['input_tokens'] + r['output_tokens']) / r['batch_time']
            for r in bs_results
        ]
        avg_bs_throughput = sum(bs_throughputs) / len(bs_throughputs)
        print(f"   • Batch size {bs}: {avg_bs_throughput:.2f} tokens/s average "
              f"(min: {min(bs_throughputs):.2f}, max: {max(bs_throughputs):.2f})")
 
    print(f"\n💡 Observations:")
    print(f"   • Throughput varies with output token length")
    print(f"   • Longer sequences generally have lower tokens/s due to autoregressive generation")
    print(f"   • Batch size {results[max_idx]['batch_size']} achieved best throughput for this workload")
 
def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Plot token throughput benchmark results from JSON data'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='/home/cpu-centric-agentic-ai/benchmark_results.json',
        help='Input JSON file with benchmark results (default: /home/cpu-centric-agentic-ai/benchmark_results.json)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='/home/cpu-centric-agentic-ai/figures/figure_4a.png',
        help='Output path for the generated plot (default: /home/cpu-centric-agentic-ai/figures/figure_4a.png)'
    )
    parser.add_argument(
        '--no-analysis',
        action='store_true',
        help='Skip printing detailed analysis'
    )

    args = parser.parse_args()

    try:
        # Generate plot
        fig = plot_throughput_from_json(args.input, args.output)

        # Print detailed analysis unless disabled
        if not args.no_analysis:
            print_analysis(args.input)

        print(f"\n✅ Token throughput analysis completed!")
        print(f"📈 Plot saved to: {args.output}")
        print(f"📊 Data available in: {args.input}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
 