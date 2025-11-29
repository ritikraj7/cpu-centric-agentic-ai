#!/usr/bin/env python3
"""
Merged Performance Visualization Script
 
This script combines 4 different performance plots into one figure with shared y-axis:
1. Haystack RAG Performance (ax1 plot from visualize_haystack_performance.py)
2. Toolformer Performance (ax1 plot from create_toolformer_performance_plots)
3. LangChain Agentic AI Performance (ax1 plot from create_langchain_performance_plots)
4. Mini-SWE-Agent End-to-End Latency (main plot from end_to_end_latency_viz.py)
"""
 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List
from matplotlib.font_manager import FontProperties
import re
import argparse
import os
 
# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")
 
class MergedPerformanceVisualizer:
    """Visualizer that merges 5 different RAG/AI performance plots with shared y-axis."""
    
    def __init__(self, base_dir: str = '/home/cpu-centric-agentic-ai'):
        """Initialize the visualizer."""
        self.base_dir = Path(base_dir)
        
    def load_haystack_data(self):
        """Load Haystack RAG performance data."""
 
        # Data from actual Haystack plot
        datasets = ['NQ', 'HotpotQA', 'TriviaQA']

        haystack_path = os.path.join(self.base_dir, 'haystack/latency_figure2.txt')
        with open(haystack_path, 'r') as file:
            text = file.read()


        
        # Find all initial inference times
        retrieval_times_str = re.findall(r'RETRIEVAL PHASE:\s*([\d.]+)', text)
        retrieval_times = [float(x)/1000.0 for x in retrieval_times_str]
        generation_times_str = re.findall(r'GENERATION PHASE:\s*([\d.]+)', text)
        generation_times = [float(x)/1000.0 for x in generation_times_str]
      
        return datasets, retrieval_times, generation_times
    
    def load_toolformer_data(self):
        """Load Toolformer performance data - using ax2 (Math datasets) from the script."""
        # Data from ax2 (Math Dataset Performance) in create_toolformer_performance_plots()
        datasets = ['ASDiv', 'SVAMP', 'MAWPS']  

        toolformer_path = os.path.join(self.base_dir, 'toolformer/latency_log.txt')
        with open(toolformer_path, 'r') as file:
            text = file.read()


        
        # Find all initial inference times
        initial_inference_str = re.findall(r'Initial inference:\s*([\d.]+)', text)
        tool_usage_str = re.findall(r'Calculations:\s*([\d.]+s avg)', text)
        final_inference_str = re.findall(r'Final inference:\s*([\d.]+)', text)

        initial_inference = [float(x) for x in initial_inference_str]
        tool_usage = [float(x.split('s')[0]) for x in tool_usage_str]
        final_inference = [float(x) for x in final_inference_str]

 
        return datasets, initial_inference[0:3], tool_usage[0:3], final_inference[0:3]
    
    
    def load_langchain_data(self):
        """Load LangChain agentic AI performance data."""
        # Data from the langchain visualization script
        datasets = ['FreshQA', 'MusiQue', 'QASC']
        google_search_times = []
        
        langchain_path = os.path.join(self.base_dir, 'langchain/latency_figure2.txt')
        with open(langchain_path, 'r') as file:
            text = file.read()

        web_search_times = [float(x) for x in re.findall(r'web_search\s+\d+\s+([\d.]+)', text)]
        fetch_url_times = [float(x) for x in re.findall(r'fetch_url\s+\d+\s+([\d.]+)', text)]
        summarization_times = [float(x) for x in re.findall(r'summarize\s+\d+\s+([\d.]+)', text)]
        llm_inference_times = [float(x) for x in re.findall(r'llm_inference\s+\d+\s+([\d.]+)', text)]
        

        # Calculate search times (sum of web_search and fetch_url for each batch)
        for i in range(len(web_search_times)):
            google_search_time = web_search_times[i] + fetch_url_times[i]
            google_search_times.append(google_search_time)
 
        return datasets, google_search_times, summarization_times, llm_inference_times
 
    def extract_timing_sequence_swe_agent(self, text, dataset_name='APPS'):
        """
        Extract LLM and Bash timing data and create the required format
        
        Args:
            text: Input text containing timing data
            dataset_name: Name of the dataset (default: 'APPS')
        
        Returns:
            Dictionary with name, sequence, and labels
        """
        
        # Extract LLM times (duration_total_seconds)
        llm_pattern = r'"duration_total_seconds":\s*([\d.]+)'
        llm_times = [float(x) for x in re.findall(llm_pattern, text)]
        
        # Extract Bash times (duration_seconds)
        bash_pattern = r'"duration_seconds":\s*([\d.]+)'
        bash_times = [float(x) for x in re.findall(bash_pattern, text)]
        
        # Verify both have same number of occurrences
        if len(llm_times) != len(bash_times):
            print(f"Warning: LLM times ({len(llm_times)}) != Bash times ({len(bash_times)})")
        
        # Create alternating sequence: LLM, Bash, LLM, Bash, ...
        sequence = []
        labels = []
        
        max_length = max(len(llm_times), len(bash_times))
        
        for i in range(max_length):
            # Add LLM time if available
            if i < len(llm_times):
                sequence.append(llm_times[i])
                labels.append('LLM')
            
            # Add Bash time if available
            if i < len(bash_times):
                sequence.append(bash_times[i])
                labels.append('Bash')
        
        return {
            'name': dataset_name,
            'sequence': sequence,
            'labels': labels
        }


    def load_swe_agent_data(self):
        """Load Mini-SWE-Agent latency data with multi-stacked sequential segments."""

        swe_agent_sorting_path = os.path.join(self.base_dir, 'mini-swe-agent/benchmark_results_temp/sorting_benchmark.json')
        with open(swe_agent_sorting_path, 'r') as file:
            text_sorting = file.read() 
        swe_agent_integration_path = os.path.join(self.base_dir, 'mini-swe-agent/benchmark_results_temp/integration_benchmark.json')
        with open(swe_agent_integration_path, 'r') as file:
            text_integration = file.read()         
        workload_data = []

        workload_data.append(self.extract_timing_sequence_swe_agent(text_sorting, 'APPS'))
        workload_data.append(self.extract_timing_sequence_swe_agent(text_integration, 'BigCodeBench'))
        
        return workload_data
    
    def create_merged_plot(self, output_path: str = None):
        """Create merged plot with all 5 performance visualizations."""
        
        # Load data from all sources
        hay_datasets, hay_search, hay_llm = self.load_haystack_data()
        tf_datasets, tf_init, tf_tool, tf_final = self.load_toolformer_data()
        # chem_datasets, chem_init, chem_tool, chem_final = self.load_chemcrow_data()
        lc_datasets, lc_google, lc_summ, lc_llm = self.load_langchain_data()
        swe_workload_data = self.load_swe_agent_data()
        
        # Create figure with 5 subplots with separate y-axes
        fig, axes = plt.subplots(1, 4, figsize=(24, 7))
        
        # Define distinct colors for LLM models
        llama2_color = '#FF9999'     # Coral pink - Llama2-7B (Haystack, LangChain)
        gptj_color = '#99FF99'       # Light green - GPT-J-6B (Toolformer)  
        gpt4_color = '#99CCFF'       # Light blue - GPT-4 (ChemCrow)
        codellama_color = '#FFFF99'  # Light yellow - CodeLlama-13B (SWE-Agent)
        
        # Define distinct colors for different tools
        enns_color = '#DDA0DD'       # Plum - ENNS Retrieval
        wolfram_color = '#FFB347'    # Peach - WolframAlpha API
        literature_color = '#F0E68C' # Khaki - Literature Search
        google_color = '#87CEEB'     # Sky blue - Google Search
        summarization_color = '#F4A460'  # Sandy brown - LexRank Summarization
        bash_color = '#D3D3D3'       # Light gray - Bash/Python
        bar_factor = 0.5
        # Plot 1: Haystack RAG Performance
        ax1 = axes[0]
        x1 = np.arange(len(hay_datasets))*bar_factor
        width = 0.4
        
        bars1_1 = ax1.bar(x1, hay_search, width, label='ENNS Retrieval', color=enns_color,
                         hatch='///', edgecolor='gray', alpha=0.8)
        bars1_2 = ax1.bar(x1, hay_llm, width, bottom=np.array(hay_search),
                         label='GPT-OSS-20B', color=llama2_color, alpha=0.8)
        
        ax1.set_title('Haystack RAG', fontsize=24, fontweight='bold', pad=15)
        ax1.set_xlabel('QA Benchmarks', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Runtime (s)', fontsize=20, fontweight='bold')
        ax1.set_xticks(x1)
        ax1.set_xticklabels(hay_datasets, rotation=15, ha='center', fontsize=16)
        ax1.grid(True, alpha=0.2, axis='y')
        ax1.tick_params(axis='both', labelsize=16)
        
        # Add numbers inside bars
        for i, (search, llm) in enumerate(zip(hay_search, hay_llm)):
            # Numbers inside bars - show for most segments
            if search > 0.8:
                ax1.text(i*bar_factor, search/2, f'{search:.1f}', ha='center', va='center',
                        fontweight='bold', fontsize=16, color='black')
            if llm > 0.8:
                ax1.text(i*bar_factor, search + llm/2, f'{llm:.1f}', ha='center', va='center',
                        fontweight='bold', fontsize=16, color='black')
 
            # Total on top
            total = search + llm
            ax1.text(i*bar_factor, total, f'{total:.1f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=18)
        
        # Plot 2: Toolformer Performance  
        ax2 = axes[1]
        x2 = np.arange(len(tf_datasets))*bar_factor
        
        bars2_1 = ax2.bar(x2, tf_init, width, label='GPT-J-6B (Init)', color=gptj_color, alpha=0.8)
        bars2_2 = ax2.bar(x2, tf_tool, width, bottom=tf_init, label='WolframAlpha API',
                         color=wolfram_color, hatch='...', edgecolor='gray', alpha=0.8)
        bars2_3 = ax2.bar(x2, tf_final, width, bottom=np.array(tf_init) + np.array(tf_tool),
                         label='GPT-J-6B (Final)', color=gptj_color, alpha=0.8)
        
        ax2.set_title('Toolformer', fontsize=24, fontweight='bold', pad=15)
        ax2.set_xlabel('Math Benchmarks', fontsize=20, fontweight='bold')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(tf_datasets, rotation=15, ha='center', fontsize=16)
        ax2.grid(True, alpha=0.2, axis='y')
        ax2.tick_params(axis='both', labelsize=16)
        
        # Add numbers inside bars
        for i, (init, tool, final) in enumerate(zip(tf_init, tf_tool, tf_final)):
            # Numbers inside bars - show for all visible segments
            if init > 0.5:
                ax2.text(i*bar_factor, init/2, f'{init:.1f}', ha='center', va='center',
                        fontweight='bold', fontsize=16, color='black')
            if tool > 0.5:
                ax2.text(i*bar_factor, init + tool/2, f'{tool:.1f}', ha='center', va='center',
                        fontweight='bold', fontsize=16, color='black')
            if final > 0.5:
                ax2.text(i*bar_factor, init + tool + final/2, f'{final:.1f}', ha='center', va='center',
                        fontweight='bold', fontsize=16, color='black')
 
            # Total on top
            total = init + tool + final
            ax2.text(i*bar_factor, total, f'{total:.1f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=18)
        
        
        # Plot 4: LangChain Agentic AI Performance
        ax4 = axes[2]
        x4 = np.arange(len(lc_datasets))*bar_factor
        
        bars4_1 = ax4.bar(x4, lc_google, width, label='Google Search',
                         color=google_color, hatch='|||', edgecolor='gray', alpha=0.8)
        # bars4_2 = ax4.bar(x4, lc_url, width, bottom=lc_google, label='URL Fetching',
        #                  color=tool_color, hatch='---', edgecolor='gray', alpha=0.8)
        bars4_2 = ax4.bar(x4, lc_summ, width, bottom=np.array(lc_google),
                         label='Summarization', color=summarization_color, hatch='---', edgecolor='gray', alpha=0.8)
        bars4_3 = ax4.bar(x4, lc_llm, width,
                         bottom=np.array(lc_google) + np.array(lc_summ),
                         label='Llama2-7B', color=llama2_color, alpha=0.8)
        
        ax4.set_title('LangChain', fontsize=24, fontweight='bold', pad=15)
        ax4.set_xlabel('QA Benchmarks', fontsize=20, fontweight='bold')
        ax4.set_xticks(x4)
        ax4.set_xticklabels(lc_datasets, rotation=15, ha='center', fontsize=16)
        ax4.grid(True, alpha=0.2, axis='y')
        ax4.tick_params(axis='both', labelsize=16)
        
        # Add numbers inside bars
        for i, (google, summ, llm) in enumerate(zip(lc_google, lc_summ, lc_llm)):
            # Numbers inside bars (show for most segments)
            if google > 0.5:
                ax4.text(i*bar_factor, google/2, f'{google:.1f}', ha='center', va='center',
                        fontweight='bold', fontsize=16, color='black')
            if summ > 0.5:
                ax4.text(i*bar_factor, google + summ/2, f'{summ:.1f}', ha='center', va='center',
                        fontweight='bold', fontsize=16, color='black')
            if llm > 0.8:
                ax4.text(i*bar_factor, google + summ + llm/2, f'{llm:.1f}', ha='center', va='center',
                        fontweight='bold', fontsize=16, color='black')
 
            # Total on top
            total = google + summ + llm
            ax4.text(i*bar_factor, total, f'{total:.0f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=18)
        
        # Plot 5: Mini-SWE-Agent End-to-End Latency (Multi-stacked bars)
        ax5 = axes[3]
        x5 = np.arange(len(swe_workload_data))
        
        # Plot multi-stacked bars for each workload
        for i, workload in enumerate(swe_workload_data):
            bottom = 0
            sequence = workload['sequence']
            labels = workload['labels']
            
            for j, (time_val, label) in enumerate(zip(sequence, labels)):
                if label == 'LLM':
                    color = codellama_color
                    hatch_pattern = None
                    legend_label = 'Qwen2.5-Coder-32B' if i == 0 and j == 0 else ""
                else:  # Bash
                    color = bash_color
                    hatch_pattern = '+++'
                    legend_label = 'Bash/Python' if i == 0 and label not in labels[:j] else ""
                
                ax5.bar(i, time_val, 2*width, bottom=bottom,
                       color=color, alpha=0.8, hatch=hatch_pattern, edgecolor='gray', linewidth=0.5,
                       label=legend_label if legend_label else "")
                
                # Add segment numbers for significant segments
                if time_val > 3.0:  # Only label segments > 3 seconds for better fit
                    ax5.text(i, bottom + time_val/2, f'{time_val:.0f}',
                           ha='center', va='center', fontsize=16,
                           fontweight='bold', color='black')
                elif time_val > 1.0:  # Show 1 decimal for medium segments
                    ax5.text(i, bottom + time_val/2, f'{time_val:.1f}',
                           ha='center', va='center', fontsize=16,
                           fontweight='bold', color='black')
 
                bottom += time_val
 
            # Add total label
            total = sum(sequence)
            ax5.text(i, total, f'{total:.0f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=18, color='black')
        
        # Extract workload names and shorten them
        swe_names = ['APPS', 'BigCodeBench']
        
        ax5.set_title('SWE-Agent', fontsize=24, fontweight='bold', pad=15)
        ax5.set_xlabel('Coding Benchmarks', fontsize=20, fontweight='bold')
        ax5.set_xticks(x5)
        ax5.set_xticklabels(swe_names, rotation=15, ha='center', fontsize=16)
        ax5.grid(True, alpha=0.2, axis='y')
        ax5.tick_params(axis='both', labelsize=16)
        
        # Create unified legend with two main categories
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        # LLM Inference category (different colors for different models)
        llm_elements = [
            Patch(facecolor=llama2_color, alpha=0.8, label='GPT-OSS-20B (vLLM)'),
            Patch(facecolor=gptj_color, alpha=0.8, label='GPT-J-6B (vLLM)'),
            Patch(facecolor=gpt4_color, alpha=0.8, label='GPT-4-0613 (OpenAI API)'),
            Patch(facecolor=codellama_color, alpha=0.8, label='Qwen2.5-Coder-32B (vLLM)')
        ]
        
        # Tool category (distinct colors with different patterns)
        tool_elements = [
            Patch(facecolor=enns_color, hatch='///', edgecolor='gray', alpha=0.8, label='ENNS Retrieval'),
            Patch(facecolor=wolfram_color, hatch='...', edgecolor='gray', alpha=0.8, label='WolframAlpha API'),
            Patch(facecolor=literature_color, hatch='xxx', edgecolor='gray', alpha=0.8, label='Literature Search'),
            Patch(facecolor=google_color, hatch='|||', edgecolor='gray', alpha=0.8, label='Google Search'),
            # Patch(facecolor=tool_color, hatch='---', edgecolor='gray', alpha=0.8, label='URL Fetching'),
            Patch(facecolor=summarization_color, hatch='---', edgecolor='gray', alpha=0.8, label='LexRank Summarization'),
            Patch(facecolor=bash_color, hatch='+++', edgecolor='gray', alpha=0.8, label='Bash/Python')
        ]
        
        # Create two separate legends on the right side
        # LLM Models legend (upper right)
        legend1 = fig.legend(llm_elements, [elem.get_label() for elem in llm_elements],
                            loc='upper right', bbox_to_anchor=(0.99, 0.95),
                            fontsize=20, frameon=True, fancybox=True, shadow=True,
                            title='LLM Inference (GPU)', title_fontproperties=FontProperties(weight='bold', size=20))
        
        # Tools legend (lower right)  
        legend2 = fig.legend(tool_elements, [elem.get_label() for elem in tool_elements],
                            loc='lower right', bbox_to_anchor=(0.98, 0.15),
                            fontsize=20, frameon=True, fancybox=True, shadow=True,
                            title='Tools (CPU)', title_fontproperties=FontProperties(weight='bold', size=20))
 
        # Adjust layout with increased spacing between subplots
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.1, wspace=0.2, right=0.78)
        
        # Save the plot
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Merged plot saved to: {output_path}")
        else:
            plt.savefig('/home/cpu-centric-agentic-ai/figures/figure_2.png', dpi=300, bbox_inches='tight')
            print("✅ Merged plot saved to: /home/cpu-centric-agentic-ai/figures/figure_2.png")
 
        plt.show()
        plt.close()
    
    def print_summary_statistics(self):
        """Print summary statistics for all systems."""
        print("\\n" + "="*100)
        print("📊 COMPREHENSIVE AI/RAG PERFORMANCE SUMMARY")
        print("="*100)
        
        # Load all data
        hay_datasets, hay_search, hay_llm = self.load_haystack_data()
        tf_datasets, tf_init, tf_tool, tf_final = self.load_toolformer_data()
        lc_datasets, lc_google, lc_summ, lc_llm = self.load_langchain_data()
        swe_workload_data = self.load_swe_agent_data()
        
        # Calculate totals
        hay_totals = [s + l for s, l in zip(hay_search, hay_llm)]
        tf_totals = [i + t + f for i, t, f in zip(tf_init, tf_tool, tf_final)]
        lc_totals = [g + s + l for g, s, l in zip(lc_google, lc_summ, lc_llm)]
        swe_totals = [sum(w['sequence']) for w in swe_workload_data]
        
        systems = [
            ("Haystack RAG", np.mean(hay_totals)),
            ("Toolformer", np.mean(tf_totals)),
            ("LangChain", np.mean(lc_totals)),
            ("Mini-SWE-Agent", np.mean(swe_totals))
        ]
        
        print(f"{'System':<20} {'Avg Runtime (s)':<15} {'Primary Components':<40}")
        print("-" * 100)
        
        components_info = [
            "Embedding + Retrieval + LLM Generation",
            "Initial Inference + Tool Usage + Final Inference",
            "Google Search + Summarization + LLM",
            "LLM Inference + Bash Processing"
        ]
        
        for (system, avg_time), components in zip(systems, components_info):
            print(f"{system:<20} {avg_time:<15.2f} {components:<40}")
        
        print("\\n📈 Key Insights:")
        fastest = min(systems, key=lambda x: x[1])
        slowest = max(systems, key=lambda x: x[1])
        print(f"• Fastest System: {fastest[0]} ({fastest[1]:.2f}s average)")
        print(f"• Slowest System: {slowest[0]} ({slowest[1]:.2f}s average)")
        print(f"• Performance Range: {fastest[1]:.2f}s - {slowest[1]:.2f}s")
        print(f"• Systems analyzed: {len(systems)} different AI/RAG architectures")
        print(f"• All systems show multi-component processing pipelines")
        print(f"• Tool usage and external API calls are major latency contributors")
 
def main():
    """Main function to create the merged visualization."""
    parser = argparse.ArgumentParser(
        description='Create merged performance visualization for AI/RAG systems'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='/home/cpu-centric-agentic-ai/figures/figure_2.png',
        help='Output path for the generated plot (default: /home/cpu-centric-agentic-ai/figures/figure_2.png)'
    )
    parser.add_argument(
        '-b', '--base-dir',
        type=str,
        default='/home/cpu-centric-agentic-ai',
        help='Base directory for data files (default: /home/cpu-centric-agentic-ai)'
    )

    args = parser.parse_args()

    print("🎨 Creating Merged Performance Visualization...")
    print("=" * 60)

    visualizer = MergedPerformanceVisualizer(base_dir=args.base_dir)

    # Create the merged plot
    visualizer.create_merged_plot(output_path=args.output)

    # Optional: Print summary statistics
    # visualizer.print_summary_statistics()

    print("✅ Merged visualization completed!")
 
if __name__ == "__main__":
    main()