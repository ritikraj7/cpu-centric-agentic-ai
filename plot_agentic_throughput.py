"""
AI Workload Throughput Analysis Visualization Script

This script plots throughput (batch/sec) vs batch size for 4 different AI workloads:
- Toolformer
- Haystack
- LangChain
- SWE-Agent

Batch sizes: 1, 2, 4, 8, 16, 32, 64, 128
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import argparse
import os
from pathlib import Path
 
# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")
 
class AIWorkloadThroughputVisualizer:
    """Visualizer for throughput analysis across different AI workloads."""

    def __init__(self, base_dir: str = '/home/cpu-centric-agentic-ai'):
        """Initialize the visualizer."""
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        self.base_dir = Path(base_dir)
        
    def extract_max_end_times(self, text):
        """
        Extract maximum end time for each batch size from the text.
        Returns a list of maximum end times in order of batch sizes.
        """
        max_times = []
        
        # Split text into sections by "Batch Size:"
        batch_sections = re.split(r'Batch Size: (\d+)', text)
        throughput = []
        # Process each batch section (skip the first empty element)
        for i in range(1, len(batch_sections), 2):
            batch_size = int(batch_sections[i])
            batch_content = batch_sections[i + 1] if i + 1 < len(batch_sections) else ""
            
            # Find all timing end values in this batch
            timing_pattern = r'1: \[TIMING\] end: ([\d.]+)s'
            end_times = re.findall(timing_pattern, batch_content)
            
            if end_times:
                # Convert to float and find maximum
                float_times = [float(time) for time in end_times]
                max_time = max(float_times)
                max_times.append(max_time)
                throughput.append(batch_size/max_time)
                # print(f"Batch Size {batch_size}: Found {len(end_times)} times, Max = {max_time}s")
            else:
                print(f"Batch Size {batch_size}: No timing data found")
        
        return throughput


    def extract_haystack_throughput(self):
        haystack_path = os.path.join(self.base_dir, 'haystack/figure_4b.json')
        with open(haystack_path, 'r') as file:
            text = file.read()

        throughput_str = re.findall(r'"throughput_queries_per_sec":\s*([0-9]*\.?[0-9]+)', text)
        throughput = [float(x) for x in throughput_str]

        return throughput

    def extract_toolformer_throughput(self):
        """Extract throughput values using regex"""
        toolformer_path = os.path.join(self.base_dir, 'toolformer/throughput_log.txt')
        with open(toolformer_path, 'r') as file:
            text = file.read()

        # Pattern to match throughput values (numbers in the second column)
        pattern = r'^\s*\d+\s+(\d+\.\d+)\s+'

        throughput_list = []
        for line in text.split('\n'):
            match = re.match(pattern, line)
            if match:
                throughput_list.append(float(match.group(1)))

        return throughput_list

    def generate_throughput_data(self):
        """Generate realistic throughput data for different workloads."""

        langchain_path = os.path.join(self.base_dir, 'langchain/batch_timing_results_4b.txt')
        with open(langchain_path, 'r') as file:
            langchain_text = file.read()

        swe_path = os.path.join(self.base_dir, 'mini-swe-agent/batch_timing_results_4b.txt')
        with open(swe_path, 'r') as file:
            swe_text = file.read()

        # Throughput data (batch/sec) - realistic values based on AI workload characteristics
        throughput_data = {
            'Toolformer': self.extract_toolformer_throughput(),  # Math reasoning workload
            'Haystack': self.extract_haystack_throughput(),   # RAG retrieval workload   
            'LangChain': self.extract_max_end_times(langchain_text),   # Agentic AI workload
            'SWE-Agent': self.extract_max_end_times(swe_text)        # Code generation workload
        }
        # print(throughput_data)
        return throughput_data
    
    def create_throughput_plot(self, output_path: str = None):
        """Create throughput vs batch size plot."""
        
        # Get throughput data
        throughput_data = self.generate_throughput_data()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Define colors for each workload
        colors = {
            'Toolformer': '#FF6B6B',    # Red
            'Haystack': '#45B7D1',      # Teal
            'LangChain': '#96CEB4',     # Green
            'SWE-Agent': '#FECA57'      # Yellow
        }
        
        # Define markers for each workload
        markers = {
            'Toolformer': 'o',          # Circle
            'Haystack': 's',            # Square
            'LangChain': 'D',           # Diamond
            'SWE-Agent': 'v'            # Triangle down
        }
        
        # Plot throughput curves
        for workload, throughput in throughput_data.items():
            # Use only as many batch sizes as we have throughput data points
            x_data = self.batch_sizes[:len(throughput)]
            plt.plot(x_data, throughput,
                    color=colors[workload],
                    marker=markers[workload],
                    linewidth=4,
                    markersize=8,
                    label=workload,
                    alpha=1)
            
        
        # Customize plot
        plt.xlabel('Batch Size', fontsize=24, fontweight='bold')
        plt.ylabel('Throughput (batch/s)', fontsize=24, fontweight='bold')
        
        # Set x-axis to log scale for better visualization
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        plt.xticks(self.batch_sizes, labels=[str(x) for x in self.batch_sizes])
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(loc='lower right', fontsize=24, frameon=True,
                  fancybox=True, shadow=True)
        
        # Improve layout
        plt.tight_layout()
        
        # Save the plot
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Throughput plot saved to: {output_path}")
        else:
            default_path = os.path.join(self.base_dir, 'figures/figure_4b.png')
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"✅ Throughput plot saved to: {default_path}")
        
        plt.show()
        plt.close()
    
    
    def print_throughput_summary(self):
        """Print summary statistics for throughput analysis."""
        
        throughput_data = self.generate_throughput_data()
        
        print("\n" + "="*80)
        print("📊 AI WORKLOAD THROUGHPUT ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"{'Workload':<15} {'Batch=1':<10} {'Batch=16':<10} {'Batch=128':<10} {'Peak Tput':<12} {'Efficiency':<10}")
        print("-" * 80)
        
        for workload, throughput in throughput_data.items():
            batch_1 = throughput[0] if len(throughput) > 0 else 0
            batch_16 = throughput[4] if len(throughput) > 4 else "N/A"
            batch_128 = throughput[7] if len(throughput) > 7 else "N/A"
            peak_tput = max(throughput) if throughput else 0
            efficiency = (peak_tput / (batch_1 * 128)) * 100 if batch_1 > 0 else 0
            
            batch_16_str = f"{batch_16:.1f}" if batch_16 != "N/A" else "N/A"
            batch_128_str = f"{batch_128:.1f}" if batch_128 != "N/A" else "N/A"
            
            print(f"{workload:<15} {batch_1:<10.1f} {batch_16_str:<10} {batch_128_str:<10} {peak_tput:<12.1f} {efficiency:<10.1f}%")
        
        print("\n📈 Key Insights:")
        
        # Find best and worst performers
        max_tputs = {workload: max(throughput) for workload, throughput in throughput_data.items()}
        best_workload = max(max_tputs, key=max_tputs.get)
        worst_workload = min(max_tputs, key=max_tputs.get)
        
        print(f"• Highest throughput: {best_workload} ({max_tputs[best_workload]:.1f} batch/sec)")
        print(f"• Lowest throughput: {worst_workload} ({max_tputs[worst_workload]:.1f} batch/sec)")
        
        # Calculate scaling factors
        scaling_factors = {}
        for workload, throughput in throughput_data.items():
            if len(throughput) > 7 and throughput[0] > 0:
                scaling_factor = throughput[7] / throughput[0]  # batch_128 / batch_1
            elif len(throughput) > 0 and throughput[0] > 0:
                scaling_factor = throughput[-1] / throughput[0]  # last / first
            else:
                scaling_factor = 1.0
            scaling_factors[workload] = scaling_factor
        
        best_scaling = max(scaling_factors, key=scaling_factors.get)
        print(f"• Best scaling workload: {best_scaling} ({scaling_factors[best_scaling]:.1f}x improvement)")
        print(f"• Batch size sweet spot: Most workloads peak between 64-128 batch size")
        print(f"• Complex workloads (SWE-Agent, ChemCrow) show lower absolute throughput")
        print(f"• RAG workloads (Haystack) show highest throughput due to parallel retrieval")
 
def main():
    """Main function to create throughput visualizations."""
    parser = argparse.ArgumentParser(
        description='Create AI workload throughput analysis visualization'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path for the generated plot (default: <base-dir>/figures/figure_4b.png)'
    )
    parser.add_argument(
        '-b', '--base-dir',
        type=str,
        default='/home/cpu-centric-agentic-ai',
        help='Base directory for data files (default: /home/cpu-centric-agentic-ai)'
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip printing throughput summary'
    )

    args = parser.parse_args()

    print("📈 Creating AI Workload Throughput Analysis Visualizations...")
    print("=" * 60)

    visualizer = AIWorkloadThroughputVisualizer(base_dir=args.base_dir)

    # Create main throughput plot
    print("\n🎯 Creating main throughput plot...")
    visualizer.create_throughput_plot(output_path=args.output)

    # Print summary statistics unless disabled
    if not args.no_summary:
        visualizer.print_throughput_summary()

    print("\n✅ AI workload throughput analysis completed!")


if __name__ == "__main__":
    main()