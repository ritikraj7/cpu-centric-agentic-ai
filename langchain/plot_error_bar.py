"""
Plot timing statistics for different batch sizes with error bars showing min/max values.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
 
def load_data(csv_file='/home/cpu-centric-agentic-ai/langchain/batch_timing_results_4c.csv'):
    """Load timing data from CSV file"""
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        print("Please run the bash script first: /home/cpu-centric-agentic-ai/langchain/run_batch_experiments.sh")
        sys.exit(1)
 
    df = pd.read_csv(csv_file)
    return df
 
def plot_timing_statistics(df, output_file='/home/cpu-centric-agentic-ai/figures/figure_4c.png'):
    """
    Create error plot showing average latency vs batch size for three stages
    with min/max values as error bars
    """
    # Filter for the three stages we want to plot
    stages_to_plot = ['fetch_url', 'summarize', 'llm_inference']
    df_filtered = df[df['stage'].isin(stages_to_plot)]
 
    # Create figure with good size
    fig, ax = plt.subplots(figsize=(10, 6))
 
    # Color and marker for each stage
    colors = {'fetch_url': '#2E86AB', 'summarize': '#A23B72', 'llm_inference': '#F18F01'}
    markers = {'fetch_url': 'o', 'summarize': 's', 'llm_inference': '^'}
    labels = {'fetch_url': 'URL Fetch', 'summarize': 'Summarization', 'llm_inference': 'LLM Inference'}
 
    # Plot each stage
    for stage in stages_to_plot:
        stage_data = df_filtered[df_filtered['stage'] == stage].sort_values('batch_size')
 
        batch_sizes = stage_data['batch_size'].values
        avg_times = stage_data['avg'].values
        min_times = stage_data['min'].values
        max_times = stage_data['max'].values
 
        # Calculate error bars (distance from average to min/max)
        lower_error = avg_times - min_times
        upper_error = max_times - avg_times
 
        # Plot with error bars
        ax.errorbar(batch_sizes, avg_times,
                   yerr=[lower_error, upper_error],
                   label=labels[stage],
                   marker=markers[stage],
                   color=colors[stage],
                   linewidth=2,
                   markersize=8,
                   capsize=5,
                   capthick=2,
                   elinewidth=1.5,
                   alpha=0.8)
 
    # Customize plot
    ax.set_xlabel('Batch Size', fontsize=24, fontweight='bold')
    ax.set_ylabel('Latency (s)', fontsize=24, fontweight='bold')
 
    # Set x-axis to log scale for better visualization
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128])
    ax.tick_params(axis='both', which='major', labelsize=18)
 
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
 
    # Legend
    ax.legend(loc='best', fontsize=24, framealpha=0.9)
 
    # Tight layout
    plt.tight_layout()
 
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
 
    return fig, ax
 
def plot_separate_stages(df, output_file='batch_timing_separate.png'):
    """
    Create separate subplots for each stage
    """
    stages_to_plot = ['fetch_url', 'summarize', 'llm_inference']
    labels = {'fetch_url': 'URL Fetch', 'summarize': 'Summarization', 'llm_inference': 'LLM Inference'}
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 
    for idx, stage in enumerate(stages_to_plot):
        ax = axes[idx]
        stage_data = df[df['stage'] == stage].sort_values('batch_size')
 
        batch_sizes = stage_data['batch_size'].values
        avg_times = stage_data['avg'].values
        min_times = stage_data['min'].values
        max_times = stage_data['max'].values
 
        lower_error = avg_times - min_times
        upper_error = max_times - avg_times
 
        ax.errorbar(batch_sizes, avg_times,
                   yerr=[lower_error, upper_error],
                   marker='o',
                   linewidth=2,
                   markersize=8,
                   capsize=5,
                   capthick=2,
                   elinewidth=1.5,
                   color='#2E86AB',
                   alpha=0.8)
 
        ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(labels[stage], fontsize=14, fontweight='bold')
 
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
        ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128])
 
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
 
    plt.suptitle('Pipeline Stage Latency vs Batch Size (Separate Views)',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
 
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Separate plot saved to: {output_file}")
 
    return fig, axes
 
def print_summary_statistics(df):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
 
    stages_to_plot = ['fetch_url', 'summarize', 'llm_inference']
    labels = {'fetch_url': 'URL Fetch', 'summarize': 'Summarization', 'llm_inference': 'LLM Inference'}
 
    for stage in stages_to_plot:
        stage_data = df[df['stage'] == stage].sort_values('batch_size')
        print(f"\n{labels[stage]}:")
        print(f"  Batch sizes tested: {stage_data['batch_size'].tolist()}")
        print(f"  Avg latency range: {stage_data['avg'].min():.4f}s - {stage_data['avg'].max():.4f}s")
        print(f"  Overall min: {stage_data['min'].min():.4f}s")
        print(f"  Overall max: {stage_data['max'].max():.4f}s")
 
    print("\n" + "="*70 + "\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Plot timing statistics with error bars for different batch sizes'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='/home/cpu-centric-agentic-ai/langchain/batch_timing_results_4c.csv',
        help='Input CSV file with timing data (default: /home/cpu-centric-agentic-ai/langchain/batch_timing_results_4c.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='/home/cpu-centric-agentic-ai/figures/figure_4c.png',
        help='Output path for the generated plot (default: /home/cpu-centric-agentic-ai/figures/figure_4c.png)'
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip printing summary statistics'
    )

    args = parser.parse_args()

    print("Loading data...")
    df = load_data(csv_file=args.input)

    print(f"Loaded {len(df)} records")
    print(f"Stages found: {df['stage'].unique().tolist()}")
    print(f"Batch sizes: {sorted(df['batch_size'].unique().tolist())}")

    # Print summary statistics unless disabled
    if not args.no_summary:
        print_summary_statistics(df)

    # Create combined plot
    print("\nGenerating combined plot...")
    plot_timing_statistics(df, output_file=args.output)

    # # Create separate plots
    # print("Generating separate plots...")
    # plot_separate_stages(df)

    print(f"\nDone! Check the generated PNG at {args.output}")

if __name__ == '__main__':
    main()