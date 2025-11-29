#!/usr/bin/env python3
"""
Math Toolformer Throughput Benchmark Script

Benchmarks throughput (requests/second) vs batch size for math problem solving.
Uses the exact same workload as math_toolformer.py including Wolfram Calculator.
Implements multiprocessing where each request in a batch runs in parallel.
Supports graceful interruption and JSON output for results.

Based on: math_toolformer.py (using exact same MathToolformer.process_math_problem)
"""

import json
import time
import signal
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
import os
import sys
from dataclasses import dataclass
from datetime import datetime
import atexit

# Import the exact math toolformer components
from math_toolformer import MathToolformer, load_math_datasets


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark"""
    batch_sizes: List[int] = None
    dataset: str = "MAWPS"  # Default to MAWPS
    timeout_per_request: float = 60.0  # Timeout per request in seconds (higher for Wolfram API)
    output_file: str = "benchmark_results.json"
    model_base_url: str = "http://localhost:5000/v1"
    model_path: str = "EleutherAI/gpt-j-6B"
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]


class BenchmarkResults:
    """Thread-safe results collector with graceful shutdown support"""
    
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.results = {
            "benchmark_info": {
                "start_time": datetime.now().isoformat(),
                "script_version": "1.0",
                "dataset": None,
                "total_batch_sizes": None,
                "uses_wolfram_calculator": True,
                "workload": "exact_math_toolformer_process_math_problem"
            },
            "batch_results": [],
            "summary": {},
            "completed": False
        }
        
        # Register cleanup handlers
        atexit.register(self.save_results)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        print(f"\n🛑 Received signal {signum}. Saving results and exiting...")
        self.save_results()
        sys.exit(0)
    
    def add_batch_result(self, batch_size: int, result: Dict[str, Any]):
        """Add results for a specific batch size"""
        result["batch_size"] = batch_size
        result["timestamp"] = datetime.now().isoformat()
        self.results["batch_results"].append(result)
        self.save_results()  # Save after each batch
    
    def set_benchmark_info(self, dataset: str, batch_sizes: List[int]):
        """Set benchmark metadata"""
        self.results["benchmark_info"]["dataset"] = dataset
        self.results["benchmark_info"]["total_batch_sizes"] = len(batch_sizes)
        self.results["benchmark_info"]["planned_batch_sizes"] = batch_sizes
    
    def finalize(self):
        """Mark benchmark as completed and generate summary"""
        self.results["completed"] = True
        self.results["benchmark_info"]["end_time"] = datetime.now().isoformat()
        
        # Generate summary statistics
        if self.results["batch_results"]:
            throughputs = [r["throughput_req_per_sec"] for r in self.results["batch_results"] if "throughput_req_per_sec" in r]
            batch_sizes = [r["batch_size"] for r in self.results["batch_results"] if "throughput_req_per_sec" in r]
            
            if throughputs:
                self.results["summary"] = {
                "total_batches_completed": len(self.results["batch_results"]),
                "max_throughput": max(throughputs),
                "min_throughput": min(throughputs),
                "avg_throughput": np.mean(throughputs),
                "optimal_batch_size": batch_sizes[throughputs.index(max(throughputs))],
                "throughput_vs_batch_size": list(zip(batch_sizes, throughputs))
                }
        
        self.save_results()
    
    def save_results(self):
        """Save current results to JSON file"""
        try:
            with open(self.output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"💾 Results saved to {self.output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def process_single_math_problem_exact(problem: str, model_config: Dict[str, str]) -> Tuple[Dict[str, Any], bool]:
    """
    Process a single math problem using the EXACT same workload as math_toolformer.py
    This uses MathToolformer.process_math_problem which includes Wolfram Calculator
    
    Args:
        problem: Math problem to solve
        model_config: Model configuration dictionary
        
    Returns:
        Tuple of (result_dict, success_flag)
    """
    try:
        start_time = time.time()
        
        # Initialize Math Toolformer for this process (includes Wolfram Calculator)
        toolformer = MathToolformer(
            base_url=model_config['base_url'],
            model_path=model_config['model_path']
        )
        
        # Use the EXACT same method as in math_toolformer.py
        result = toolformer.process_math_problem(problem)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Add timing metadata
        result['processing_time'] = processing_time
        result['success'] = True
        result['timeout'] = False
        
        return result, True
        
    except Exception as e:
        processing_time = time.time() - start_time if 'start_time' in locals() else 0
        return {
            'problem': problem,
            'error': str(e),
            'success': False,
            'timeout': False,
            'processing_time': processing_time,
            'used_calculator': False,
            'latencies': {'end_to_end': processing_time}
        }, False


def benchmark_batch_throughput(problems: List[str], batch_size: int, 
                             model_config: Dict[str, str], 
                             timeout_per_request: float = 60.0) -> Dict[str, Any]:
    """
    Benchmark throughput for a specific batch size using multiprocessing
    Each request in the batch runs in parallel as specified.
    
    Args:
        problems: List of math problems to process
        batch_size: Number of parallel requests
        model_config: Model configuration
        timeout_per_request: Timeout per individual request
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n🚀 Benchmarking batch size: {batch_size}")
    
    # Select exactly batch_size number of problems
    if len(problems) < batch_size:
        # Repeat problems if we don't have enough
        repeated_problems = (problems * ((batch_size // len(problems)) + 1))[:batch_size]
        selected_problems = repeated_problems
        print(f"📝 Repeated {len(problems)} problems to get {batch_size} problems for this batch")
    else:
        selected_problems = problems[:batch_size]
    
    print(f"📊 Processing {len(selected_problems)} problems in parallel")
    
    start_time = time.time()
    results = []
    successful_requests = 0
    failed_requests = 0
    timeout_requests = 0
    
    # Use ProcessPoolExecutor for parallel processing
    # Each request runs in its own process
    with ProcessPoolExecutor(max_workers=batch_size) as executor:
        # Submit all requests in the batch
        future_to_problem = {
            executor.submit(process_single_math_problem_exact, problem, model_config): problem
            for problem in selected_problems
        }
        
        # Collect results with progress tracking
        # with tqdm(total=len(selected_problems), desc=f"Batch {batch_size}", leave=False) as pbar:
        for future in as_completed(future_to_problem, timeout=timeout_per_request * 2):
            try:
                result, success = future.result(timeout=1.0)  # Quick timeout for result collection
                results.append(result)
                
                if success:
                    successful_requests += 1
                else:
                    failed_requests += 1
                    
            except TimeoutError:
                timeout_requests += 1
                problem = future_to_problem[future]
                results.append({
                    'problem': problem,
                    'error': 'Timeout',
                    'success': False,
                    'timeout': True,
                    'processing_time': timeout_per_request,
                    'used_calculator': False,
                    'latencies': {'end_to_end': timeout_per_request}
                })
            except Exception as e:
                failed_requests += 1
                problem = future_to_problem[future]
                results.append({
                    'problem': problem,
                    'error': str(e),
                    'success': False,
                    'timeout': False,
                    'processing_time': 0,
                    'used_calculator': False,
                    'latencies': {'end_to_end': 0}
                })
                
                # pbar.update(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate throughput metrics
    total_requests = len(selected_problems)
    throughput_req_per_sec = total_requests / total_time if total_time > 0 else 0
    
    # Calculate latency statistics from successful requests
    successful_results = [r for r in results if r.get('success', False)]
    if successful_results:
        latencies = [r['processing_time'] for r in successful_results]
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
    else:
        avg_latency = p95_latency = p99_latency = 0
    
    # Count calculator usage (exact same as math_toolformer.py)
    calculator_used_count = sum(1 for r in successful_results if r.get('used_calculator', False))
    
    # Calculate Wolfram API statistics
    wolfram_calls = []
    wolfram_latencies = []
    for r in successful_results:
        if r.get('used_calculator', False) and 'latencies' in r:
            calc_latencies = r['latencies'].get('calculations', {})
            if calc_latencies:
                wolfram_calls.extend(calc_latencies.keys())
                wolfram_latencies.extend(calc_latencies.values())
    
    batch_result = {
        "batch_size": batch_size,
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "timeout_requests": timeout_requests,
        "total_time_seconds": total_time,
        "throughput_req_per_sec": throughput_req_per_sec,
        "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
        "avg_latency_seconds": avg_latency,
        "p95_latency_seconds": p95_latency,
        "p99_latency_seconds": p99_latency,
        "calculator_usage": {
            "problems_using_calculator": calculator_used_count,
            "calculator_usage_rate": calculator_used_count / successful_requests if successful_requests > 0 else 0,
            "total_wolfram_calls": len(wolfram_calls),
            "avg_wolfram_latency": np.mean(wolfram_latencies) if wolfram_latencies else 0
        },
        "detailed_results": results
    }
    
    print(f"✅ Batch {batch_size}: {throughput_req_per_sec:.2f} req/s | "
          f"Success: {successful_requests}/{total_requests} | "
          f"Calculator: {calculator_used_count}/{successful_requests} | "
          f"Avg latency: {avg_latency:.2f}s")
    
    return batch_result


def run_throughput_benchmark(config: BenchmarkConfig) -> BenchmarkResults:
    """
    Run the complete throughput benchmark across all batch sizes
    
    Args:
        config: Benchmark configuration
        
    Returns:
        BenchmarkResults object with all results
    """
    print("=" * 80)
    print("🧮 MATH TOOLFORMER THROUGHPUT BENCHMARK")
    print("Using EXACT same workload as math_toolformer.py")
    print("=" * 80)
    print(f"📊 Dataset: {config.dataset}")
    print(f"📈 Batch sizes: {config.batch_sizes}")
    print(f"💾 Output file: {config.output_file}")
    print(f"⏱️  Timeout per request: {config.timeout_per_request}s")
    print(f"🧮 Wolfram Calculator: Enabled")
    
    # Check for Wolfram Alpha API key
    if not os.getenv("WOLFRAM_ALPHA_APPID"):
        print("⚠️  WARNING: WOLFRAM_ALPHA_APPID environment variable not set!")
        print("   Calculator calls will fail, affecting accuracy of benchmark.")
    else:
        print("✅ Wolfram Alpha API key found")
    
    # Initialize results collector
    results_collector = BenchmarkResults(config.output_file)
    
    try:
        # Load dataset
        print(f"\n📚 Loading {config.dataset} dataset...")
        datasets = load_math_datasets()
        
        if config.dataset not in datasets:
            raise ValueError(f"Dataset {config.dataset} not found. Available: {list(datasets.keys())}")
        
        problems = datasets[config.dataset]
        print(f"✅ Loaded {len(problems)} problems from {config.dataset}")
        
        # Set benchmark info
        results_collector.set_benchmark_info(config.dataset, config.batch_sizes)
        
        # Model configuration
        model_config = {
            'base_url': config.model_base_url,
            'model_path': config.model_path
        }
        
        print(f"\n🔧 Model configuration:")
        print(f"   Base URL: {model_config['base_url']}")
        print(f"   Model path: {model_config['model_path']}")
        
        # Run benchmarks for each batch size
        print(f"\n🚀 Starting throughput benchmarks...")
        for i, batch_size in enumerate(config.batch_sizes):
            print(f"\n{'='*20} Batch Size {batch_size} ({i+1}/{len(config.batch_sizes)}) {'='*20}")
            
            try:
                batch_result = benchmark_batch_throughput(
                    problems=problems,
                    batch_size=batch_size,
                    model_config=model_config,
                    timeout_per_request=config.timeout_per_request
                )
                
                results_collector.add_batch_result(batch_size, batch_result)
                
            except Exception as e:
                print(f"❌ Error in batch size {batch_size}: {e}")
                error_result = {
                    "batch_size": batch_size,
                    "error": str(e),
                    "total_requests": 0,
                    "successful_requests": 0,
                    "throughput_req_per_sec": 0
                }
                results_collector.add_batch_result(batch_size, error_result)
        
        # Finalize results
        results_collector.finalize()
        
        # Print summary
        print_benchmark_summary(results_collector.results)
        
        return results_collector
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        results_collector.save_results()
        raise


def print_benchmark_summary(results: Dict[str, Any]):
    """Print a formatted summary of benchmark results"""
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 80)
    
    if not results["batch_results"]:
        print("❌ No results to display")
        return
    
    print(f"📈 Dataset: {results['benchmark_info']['dataset']}")
    print(f"⏱️  Total batches: {len(results['batch_results'])}")
    print(f"🧮 Workload: Exact math_toolformer.py (with Wolfram Calculator)")
    
    if results.get("summary"):
        summary = results["summary"]
        print(f"🏆 Max throughput: {summary['max_throughput']:.2f} req/s")
        print(f"🎯 Optimal batch size: {summary['optimal_batch_size']}")
        print(f"📊 Average throughput: {summary['avg_throughput']:.2f} req/s")
    
    print(f"\n{'Batch':<8} {'Throughput':<12} {'Success':<10} {'Calculator':<12} {'Latency':<10}")
    print(f"{'Size':<8} {'(req/s)':<12} {'Rate':<10} {'Usage':<12} {'(avg)':<10}")
    print("-" * 62)
    
    for result in results["batch_results"]:
        if "error" not in result:
            calc_usage = result.get('calculator_usage', {}).get('calculator_usage_rate', 0)
            print(f"{result['batch_size']:<8} "
                  f"{result['throughput_req_per_sec']:<12.2f} "
                  f"{result['success_rate']:<10.1%} "
                  f"{calc_usage:<12.1%} "
                  f"{result['avg_latency_seconds']:<10.2f}")
        else:
            print(f"{result['batch_size']:<8} {'ERROR':<12} {'N/A':<10} {'N/A':<12} {'N/A':<10}")


def main():
    """Main entry point for the benchmark script"""
    parser = argparse.ArgumentParser(description="Math Toolformer Throughput Benchmark - Exact Workload")
    parser.add_argument(
        "--dataset",
        choices=["ASDiv", "SVAMP", "MAWPS"],
        default="MAWPS",
        help="Dataset to use for benchmarking (default: MAWPS)"
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="Batch sizes to test (default: [1, 2, 4, 8, 16, 32, 64, 128])"
    )
    parser.add_argument(
        "--output",
        default="/home/agentic/toolformer/benchmark_results_4b.json",
        help="Output JSON file for results (default: benchmark_results.json)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout per request in seconds (default: 60.0)"
    )
    parser.add_argument(
        "--model-url",
        default="http://localhost:5000/v1",
        help="VLLM base URL (default: http://localhost:5000/v1)"
    )
    parser.add_argument(
        "--model-path",
        default="EleutherAI/gpt-j-6b",
        help="Path to the model"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = BenchmarkConfig(
        batch_sizes=sorted(args.batch_sizes),
        dataset=args.dataset,
        timeout_per_request=args.timeout,
        output_file=args.output,
        model_base_url=args.model_url,
        model_path=args.model_path
    )
    
    print(f"🧮 Math Toolformer Throughput Benchmark")
    print(f"📊 Configuration:")
    print(f"   Dataset: {config.dataset}")
    print(f"   Batch sizes: {config.batch_sizes}")
    print(f"   Output: {config.output_file}")
    print(f"   Timeout: {config.timeout_per_request}s")
    print(f"   Workload: Exact math_toolformer.py")
    
    try:
        results = run_throughput_benchmark(config)
        print(f"\n✅ Benchmark completed successfully!")
        print(f"📁 Results saved to: {config.output_file}")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Benchmark interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()