"""
LLM Inference Throughput Benchmarking Script
 
This script benchmarks LLM inference throughput (requests/second) vs batch size
for different combinations of input and output tokens using vLLM server.
 
Configuration:
- Batch sizes: [1, 2, 4, 8, 16, 32, 64, 128]
- Input tokens: [500, 1000, 1500, 2000]
- Output tokens: [500, 1000, 1500, 2000]
- Total runs: 4 * 4 * 8 = 128 combinations
"""
 
import json
import time
import requests
import asyncio
import aiohttp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Tuple
import argparse
import logging
 
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 
class LLMBenchmark:
    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url
        self.results = []
        
    def generate_prompt(self, target_tokens: int) -> str:
        """Generate a prompt with approximately target_tokens length."""
        # Approximate 4 characters per token
        base_text = "Write a detailed analysis about artificial intelligence and machine learning technologies. "
        repeat_count = max(1, target_tokens * 4 // len(base_text))
        return base_text * repeat_count
    
    async def make_request(self, session: aiohttp.ClientSession, prompt: str, max_tokens: int) -> Tuple[bool, float]:
        """Make a single request to the vLLM server."""
        payload = {
            "model": "openai/gpt-oss-20b",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        start_time = time.time()
        try:
            async with session.post(f"{self.server_url}/v1/completions",
                                  json=payload,
                                  timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status == 200:
                    await response.json()
                    end_time = time.time()
                    return True, end_time - start_time
                else:
                    logger.error(f"Request failed with status {response.status}")
                    return False, 0
        except Exception as e:
            logger.error(f"Request failed with error: {e}")
            return False, 0
    
    async def benchmark_batch(self, batch_size: int, input_tokens: int, output_tokens: int) -> Dict:
        """Benchmark a specific combination of batch_size, input_tokens, and output_tokens."""
        logger.info(f"Benchmarking: batch_size={batch_size}, input_tokens={input_tokens}, output_tokens={output_tokens}")
        
        prompt = self.generate_prompt(input_tokens)
        
        # Single benchmark run
        logger.info(f"Running single benchmark batch...")
        
        async with aiohttp.ClientSession() as session:
            # Create batch of requests
            batch_start = time.time()
            tasks = []
            for _ in range(batch_size):
                task = self.make_request(session, prompt, output_tokens)
                tasks.append(task)
            
            # Execute batch concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_end = time.time()
            
            # Check if all requests in batch succeeded
            batch_successful = all(
                isinstance(result, tuple) and result[0]
                for result in results if not isinstance(result, Exception)
            )
            
            if not batch_successful:
                logger.error("Batch failed!")
                return {
                    'batch_size': batch_size,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'throughput_req_per_sec': 0,
                    'batch_time': 0,
                    'successful': False
                }
            
            batch_time = batch_end - batch_start
            throughput = batch_size / batch_time  # requests per second
            logger.info(f"Completed in {batch_time:.2f}s - {throughput:.2f} req/s")
        
        result = {
            'batch_size': batch_size,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'throughput_req_per_sec': throughput,
            'batch_time': batch_time,
            'successful': True
        }
        
        return result
    
    async def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        input_tokens = [500, 1000, 1500, 2000]
        output_tokens = [500, 1000, 1500, 2000]
        
        total_combinations = len(batch_sizes) * len(input_tokens) * len(output_tokens)
        logger.info(f"Starting benchmark with {total_combinations} combinations...")
        
        current_combination = 0
        
        for batch_size in batch_sizes:
            for input_tok in input_tokens:
                for output_tok in output_tokens:
                    current_combination += 1
                    logger.info(f"Progress: {current_combination}/{total_combinations}")
                    
                    result = await self.benchmark_batch(batch_size, input_tok, output_tok)
                    self.results.append(result)
                    
                    # Add small delay between combinations
                    await asyncio.sleep(1)
        
        logger.info("Benchmark completed!")
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON file."""
        timestamp = datetime.now().isoformat()
        data = {
            'timestamp': timestamp,
            'server_url': self.server_url,
            'model_path': 'openai/gpt-oss-20b',
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    
    def load_results(self, filename: str = "benchmark_results.json"):
        """Load results from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.results = data['results']
        logger.info(f"Results loaded from {filename}")
    
    def plot_results(self):
        """Create comprehensive plots of the benchmark results."""
        if not self.results:
            logger.error("No results to plot!")
            return
        
        # Convert results to structured format for plotting
        batch_sizes = sorted(list(set(r['batch_size'] for r in self.results)))
        input_tokens = sorted(list(set(r['input_tokens'] for r in self.results)))
        output_tokens = sorted(list(set(r['output_tokens'] for r in self.results)))
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Throughput vs Batch Size for different input/output combinations
        plt.subplot(2, 3, 1)
        for input_tok in input_tokens:
            for output_tok in output_tokens:
                x_vals = []
                y_vals = []
                for result in self.results:
                    if result['input_tokens'] == input_tok and result['output_tokens'] == output_tok:
                        x_vals.append(result['batch_size'])
                        y_vals.append(result['throughput_req_per_sec'])
                
                if x_vals:
                    plt.plot(x_vals, y_vals, marker='o',
                           label=f'In:{input_tok}, Out:{output_tok}')
        
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (req/s)')
        plt.title('Throughput vs Batch Size')
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. Heatmap: Batch Size vs Input Tokens (averaged over output tokens)
        plt.subplot(2, 3, 2)
        heatmap_data = np.zeros((len(batch_sizes), len(input_tokens)))
        
        for i, batch_size in enumerate(batch_sizes):
            for j, input_tok in enumerate(input_tokens):
                throughputs = [r['throughput_req_per_sec'] for r in self.results
                             if r['batch_size'] == batch_size and r['input_tokens'] == input_tok]
                heatmap_data[i, j] = np.mean(throughputs) if throughputs else 0
        
        sns.heatmap(heatmap_data,
                   xticklabels=input_tokens,
                   yticklabels=batch_sizes,
                   annot=True, fmt='.1f', cmap='viridis')
        plt.xlabel('Input Tokens')
        plt.ylabel('Batch Size')
        plt.title('Avg Throughput Heatmap\n(Batch Size vs Input Tokens)')
        
        # 3. Heatmap: Batch Size vs Output Tokens (averaged over input tokens)
        plt.subplot(2, 3, 3)
        heatmap_data = np.zeros((len(batch_sizes), len(output_tokens)))
        
        for i, batch_size in enumerate(batch_sizes):
            for j, output_tok in enumerate(output_tokens):
                throughputs = [r['throughput_req_per_sec'] for r in self.results
                             if r['batch_size'] == batch_size and r['output_tokens'] == output_tok]
                heatmap_data[i, j] = np.mean(throughputs) if throughputs else 0
        
        sns.heatmap(heatmap_data,
                   xticklabels=output_tokens,
                   yticklabels=batch_sizes,
                   annot=True, fmt='.1f', cmap='viridis')
        plt.xlabel('Output Tokens')
        plt.ylabel('Batch Size')
        plt.title('Avg Throughput Heatmap\n(Batch Size vs Output Tokens)')
        
        # 4. Box plot: Throughput distribution by batch size
        plt.subplot(2, 3, 4)
        batch_throughputs = {}
        for batch_size in batch_sizes:
            batch_throughputs[batch_size] = [r['throughput_req_per_sec'] for r in self.results
                                           if r['batch_size'] == batch_size]
        
        plt.boxplot([batch_throughputs[bs] for bs in batch_sizes],
                   labels=batch_sizes)
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (req/s)')
        plt.title('Throughput Distribution by Batch Size')
        plt.yscale('log')
        
        # 5. Average batch time vs batch size
        plt.subplot(2, 3, 5)
        avg_times_by_batch = {}
        for batch_size in batch_sizes:
            times = [r['batch_time'] for r in self.results if r['batch_size'] == batch_size and r.get('successful', True)]
            avg_times_by_batch[batch_size] = np.mean(times) if times else 0
        
        plt.plot(list(avg_times_by_batch.keys()), list(avg_times_by_batch.values()),
                'bo-', linewidth=2, markersize=8)
        plt.xlabel('Batch Size')
        plt.ylabel('Average Batch Time (s)')
        plt.title('Average Batch Time vs Batch Size')
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 6. Success rate by batch size
        plt.subplot(2, 3, 6)
        success_rates = {}
        for batch_size in batch_sizes:
            batch_results = [r for r in self.results if r['batch_size'] == batch_size]
            if batch_results:
                successful = sum(1 for r in batch_results if r.get('successful', True))
                total = len(batch_results)
                success_rates[batch_size] = (successful / total) * 100 if total > 0 else 0
        
        plt.bar(list(success_rates.keys()), list(success_rates.values()),
               color='lightgreen', alpha=0.7)
        plt.xlabel('Batch Size')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate by Batch Size')
        plt.ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig('llm_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Plots saved to llm_benchmark_results.png")
 
async def main():
    parser = argparse.ArgumentParser(description='LLM Inference Benchmarking Tool')
    parser.add_argument('--server-url', default='http://localhost:5000',
                      help='vLLM server URL (default: http://localhost:5000)')
    parser.add_argument('--output', default='/home/ritikraj/benchmark_results.json',
                      help='Output JSON file (default: /home/ritikraj/benchmark_results.json)')
    parser.add_argument('--load-results', type=str,
                      help='Load results from existing JSON file instead of running benchmark')
    parser.add_argument('--plot-only', action='store_true',
                      help='Only generate plots from existing results file')
    
    args = parser.parse_args()
    
    benchmark = LLMBenchmark(args.server_url)
    
    if args.load_results:
        benchmark.load_results(args.load_results)
    elif not args.plot_only:
        # Check if server is accessible
        try:
            response = requests.get(f"{args.server_url}/health", timeout=10)
            if response.status_code != 200:
                logger.error(f"vLLM server not accessible at {args.server_url}")
                return
        except Exception as e:
            logger.error(f"Failed to connect to vLLM server: {e}")
            return
        
        await benchmark.run_full_benchmark()
        benchmark.save_results(args.output)
    else:
        benchmark.load_results(args.output)
    
    # benchmark.plot_results()
 
if __name__ == "__main__":
    asyncio.run(main())
 