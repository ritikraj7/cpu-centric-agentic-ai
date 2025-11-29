import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

def extract_multiprocessing_groups(file_path):
    """
    Extract multiprocessing end values and return the maximum for each group.
    Groups are determined by counting consecutive start/end pairs.
    Expected groups: 1, 2, 4, 8, 16, 32, 64, 128 processes.
    
    Returns:
        list: List of maximum end values for each group
    """
    group_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    max_values = []
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Find multiprocessing section
        lines = content.split('\n')
        in_multiprocessing = False
        current_group_ends = []
        group_index = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            if 'Running Multiprocessing' in line_stripped:
                in_multiprocessing = True
                continue
            elif line_stripped.startswith('Running ') and in_multiprocessing:
                # End of multiprocessing section
                break
            
            if in_multiprocessing and '[TIMING] end:' in line:
                match = re.search(r'end:\s*(\d+\.?\d*)s', line)
                if match:
                    current_group_ends.append(float(match.group(1)))
                    
                    # Check if we've collected enough end values for current group
                    if len(current_group_ends) == group_sizes[group_index]:
                        max_values.append(max(current_group_ends))
                        current_group_ends = []
                        group_index += 1
                        
                        # Stop if we've processed all expected groups
                        if group_index >= len(group_sizes):
                            break
        
        # Handle any remaining group
        if current_group_ends and group_index < len(group_sizes):
            max_values.append(max(current_group_ends))
    
    except Exception as e:
        print(f"Error processing multiprocessing section: {e}")
        return []
    
    return max_values




def extract_end_values_by_section(file_path):
    """
    Extract end timing values from different sections of the timing file.
    For multiprocessing, groups end values and returns the max for each group.
    
    Args:
        file_path (str): Path to the file containing timing data
        
    Returns:
        dict: Dictionary with section names as keys and lists of end values as values
    """
    sections = {
        'sequential': [],
        'multithreading': [],
        'multiprocessing': []
    }
    
    current_section = None
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Split content by sections
        lines = content.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            
            # Identify section markers
            if 'Running Sequential' in line_stripped:
                current_section = 'sequential'
                continue
            elif 'Running Multithreading' in line_stripped:
                current_section = 'multithreading'
                continue
            elif 'Running Multiprocessing' in line_stripped:
                current_section = 'multiprocessing'
                continue
            
            # Extract end values for sequential and multithreading (simple case)
            if current_section in ['sequential', 'multithreading'] and '[TIMING] end:' in line:
                match = re.search(r'end:\s*(\d+\.?\d*)s', line)
                if match:
                    sections[current_section].append(float(match.group(1)))
        
        # Special handling for multiprocessing section
        if current_section == 'multiprocessing' or 'multiprocessing' in [s for s in sections.keys()]:
            sections['multiprocessing'] = extract_multiprocessing_groups(file_path)
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return {}
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}
    
    return sections

def main():
    """Main function to create the multiprocessing visualization."""
    parser = argparse.ArgumentParser(
        description='Create multiprocessing performance visualization for LangChain'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='/home/cpu-centric-agentic-ai/figures/figure_3.png',
        help='Output path for the generated plot (default: /home/cpu-centric-agentic-ai/figures/figure_3.png)'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='/home/cpu-centric-agentic-ai/langchain/latency_figure3.txt',
        help='Input file path for timing data (default: /home/cpu-centric-agentic-ai/langchain/latency_figure3.txt)'
    )

    args = parser.parse_args()

    # Extract all sections
    all_sections = extract_end_values_by_section(args.input)

    print("Extracted sections:")
    print(f"Sequential: {all_sections.get('sequential', [])}")
    print(f"Multithreading: {all_sections.get('multithreading', [])}")
    print(f"Multiprocessing (max per group): {all_sections.get('multiprocessing', [])}")

    # Data
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    method1_latency = all_sections.get('sequential', [])
    method2_latency = all_sections.get('multithreading', [])
    method3_latency = all_sections.get('multiprocessing', [])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    bar_width = 0.25
    x_pos = np.arange(len(batch_sizes))
    bars1 = ax.bar(x_pos - bar_width, method1_latency, bar_width,
                   label='Sequential Baseline', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x_pos, method2_latency, bar_width,
                   label='Intra-process Concurrent Batching', alpha=0.8, color='lightcoral')
    bars3 = ax.bar(x_pos + bar_width, method3_latency, bar_width,
                   label='Inter-process Parallelism', alpha=0.8, color='lightgreen')

    # Set logarithmic scale for y-axis
    ax.set_yscale('log')

    # Customize the plot with larger font sizes
    ax.set_xlabel('Batch Size', fontsize=24, fontweight='bold')
    ax.set_ylabel('Latency (s)', fontsize=24, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(batch_sizes, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(fontsize=24, loc='upper left')

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on top of bars (optional) - with larger font
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {args.output}")

if __name__ == "__main__":
    main()