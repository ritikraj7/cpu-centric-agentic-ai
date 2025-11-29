# ChemCrow Benchmark Suite
 
A comprehensive benchmarking framework for evaluating ChemCrow's performance on scientific literature search tasks with detailed timing analysis and tool usage metrics.
 
## Overview
 
This benchmark suite evaluates ChemCrow's capabilities in scientific literature retrieval and analysis tasks. It provides granular performance metrics including LLM inference times, tool execution times, and framework overhead, with automatic result persistence and comprehensive statistical analysis.
 

## Architecture
 
### Task Pipeline
 
```
Initialize ChemCrow → Execute Task → Measure Timing → Validate Tools → Save Results
```

## Requirements

If already setup, activate chemcrow conda environment-

`conda activate chemcrow`

Or, if you want to setup from scratch-

### Python Dependencies
 
```bash
pip install chemcrow
pip install numpy
```
 
### ChemCrow Dependencies
 
ChemCrow requires:
- OpenAI API access (for GPT-4)
- Literature search tools (paperscraper)
- Optional: WolframAlpha API (for computational tasks)
 
### Environment Setup
 
```bash
# OpenAI API Key (required)
export OPENAI_API_KEY="your-openai-api-key"
```
 
## Usage
 
### Basic Usage
 
```bash
python benchmark_chemcrow_tasks.py
```
 
The script automatically runs all curated tasks in sequence.

### Interrupting and Resuming
 
Press `Ctrl+C` to interrupt the benchmark. Progress is automatically saved after each task. Simply run the script again to resume from where you left off.
 
## Task Configuration
 
### Curated Tasks
 
The benchmark includes literature search tasks. For example,
 
**Aspirin Literature Search**
   - Query: "Find papers on aspirin mechanism of action"
   - Expected Tool: LiteratureSearch

### (Optional) Additional Chemcrow tasks

Other ChemCrow Tools can also be used by proving API keys. Please follow original Chemcrow code and documentation.
 
 
## ChemCrow Configuration
 
The benchmark uses the following ChemCrow configuration:
 
```python
ChemCrow(
    model='gpt-4-0613',         # Primary reasoning model
    tools_model='gpt-4-0613',   # Tool selection model
    temp=0.1,                    # Low temperature for consistency
    max_iterations=10            # Maximum agent iterations
)
```
 
### Customizing Configuration
 
To modify ChemCrow settings, edit line 58 in `benchmark_chemcrow_tasks.py`:
 
```python
chem_model = ChemCrow(
    model='gpt-4-turbo',        # Use different model
    temp=0.0,                    # Adjust temperature
    max_iterations=20            # Increase iteration limit
)
```

 
## Extending the Benchmark
 
### Adding New Tool Categories
 
To benchmark tasks using different ChemCrow tools:
 
1. Create a new task category function (e.g., `get_reaction_tasks()`)
2. Specify the expected tools for validation
3. Add tasks with appropriate descriptions
4. Update the main function to include the new task category
 
Example:
 
```python
def get_reaction_tasks():
    """Tasks that use reaction prediction tools"""
    return [
        {
            "name": "predict_reaction",
            "description": "Predict the product of benzene + Cl2",
            "expected_tools": ["RXNPredict"]
        }
    ]
```
 
### Custom Metrics
 
Add custom metrics by modifying the `run_chemcrow_benchmark()` function to track additional data points:
 
```python
# Add custom metric
benchmark_result['custom_metric'] = calculate_custom_metric(result)
```
 
## Data Analysis
 
The JSON output can be easily analyzed with Python:
 
```python
import json
import pandas as pd
 
# Load results
with open('literature_search_benchmark_results.json', 'r') as f:
    results = json.load(f)
 
# Convert to DataFrame for analysis
df = pd.json_normalize(results)
 
# Analyze timing patterns
print(df[['task_name', 'timing_metrics.total_time',
          'timing_metrics.llm_total_time',
          'timing_metrics.tool_total_time']].describe())
 
# Plot time distribution
df.plot(x='task_name', y=['timing_metrics.llm_total_time',
                           'timing_metrics.tool_total_time'],
        kind='bar', stacked=True)
```
 
## Acknowledgments
 
- Built on top of the ChemCrow agent framework
- Uses OpenAI's GPT-4 for reasoning and tool selection
- Inspired by the need for reproducible chemistry AI benchmarks