#!/usr/bin/env python3
"""
Benchmarking script for mini-swe-agent with vLLM local server.
This script runs different test datasets and measures latency for various stages.
"""

import argparse
import json
import time
import sys
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Any
import timeit

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import os
from datasets import load_dataset
from typing import List
# Use direct vLLM model to avoid LiteLLM complexities
from vllm_model import VLLMModel
from minisweagent.agents.default import DefaultAgent, AgentConfig  
from minisweagent.environments.local import LocalEnvironment
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
import tempfile


class LatencyBenchmarker:
    """Main benchmarking class for testing different datasets with latency measurement."""
    
    def __init__(self, model_config: Dict[str, Any], output_dir: str = "benchmark_results"):
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_swebench_config(self):
        """Load the proper SWEBench agent configuration."""
        config_path = get_config_path(builtin_config_dir / "default.yaml")
        config = yaml.safe_load(config_path.read_text())
        return config
    
    def save_incremental_results(self, results_file: Path, all_results: List[Dict], 
                                benchmark_info: Dict = None):
        """Save results incrementally to prevent data loss."""
        total_time = sum(r.get("total_runtime", 0) for r in all_results)
        successful_completions = sum(1 for r in all_results 
                                   if r.get("exit_status") == "Submitted" or 
                                      "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in str(r.get("result", "")))
        total_instances = len(all_results)
        success_rate = (successful_completions / total_instances) * 100 if total_instances > 0 else 0
        
        partial_summary = {
            "benchmark_type": "swebench_full",
            "model": self.model_config.get('model_name', 'Unknown'),
            "total_instances": total_instances,
            "successful_completions": successful_completions,
            "success_rate_percent": success_rate,
            "total_benchmark_time": total_time,
            "average_time_per_instance": total_time / total_instances if total_instances > 0 else 0,
            "individual_results": all_results,
            "timestamp": time.time(),
            "status": "in_progress"
        }
        
        if benchmark_info:
            partial_summary.update(benchmark_info)
        
        with open(results_file, 'w') as f:
            json.dump(partial_summary, f, indent=2)
        
        print(f"📊 Progress saved: {successful_completions}/{total_instances} completed ({success_rate:.1f}%)")
        
    def setup_model(self) -> VLLMModel:
        """Initialize direct vLLM model connection."""
        return VLLMModel(
            base_url=self.model_config.get("base_url", "http://localhost:5000/v1"),
            api_key=self.model_config.get("api_key", "token-abc123"),
            model_name=self.model_config.get("model_name", "/usr/scratch/ritik/hugging_face/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7"),
            max_tokens=self.model_config.get("max_tokens", 4096),
            temperature=self.model_config.get("temperature", 0.0),
            **self.model_config.get("model_kwargs", {})
        )
    
    def run_sorting_benchmark(self) -> Dict[str, Any]:
        """Run CPU-intensive sorting algorithms benchmark."""
        # print("Running Sorting Algorithms benchmark")

        model = self.setup_model()

        # Load proper SWEBench config
        swe_config = self.load_swebench_config()
        agent_config = swe_config['agent']

        env = LocalEnvironment()

        # Create agent config with proper templates
        config = AgentConfig(
            system_template=agent_config.get('system_template', AgentConfig.system_template),
            instance_template=agent_config.get('instance_template', AgentConfig.instance_template),
            action_observation_template=agent_config.get('action_observation_template', AgentConfig.action_observation_template),
            format_error_template=agent_config.get('format_error_template', AgentConfig.format_error_template),
            timeout_template=agent_config.get('timeout_template', AgentConfig.timeout_template),
            step_limit=15,
            cost_limit=4.0
        )

        agent = DefaultAgent(model, env, config_class=lambda **kwargs: config)

        task_description = """
Sorting Algorithms Benchmark

Problem Description:
Write Python code to implement and benchmark bubble sort on arrays of sizes 10000 and 20000 elements.

Instructions:
1. Create a Python script with bubble sort implementation
2. Benchmark the sort on the specified array sizes
3. Print timing results for each array size
4. Test your implementation to verify correctness

Please implement a complete solution and finish by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""

        start_time = time.time()
        try:
            exit_status, result = agent.run(task_description)
            print(f"Exit status: {exit_status}")
        except Exception as e:
            print(f"Error during Sorting benchmark: {e}")
            exit_status, result = "Error", str(e)
        finally:
            total_time = time.time() - start_time

        # Collect detailed interaction logs
        model_call_log = []
        if hasattr(model, 'call_log'):
            model_call_log = model.call_log

        bash_execution_log = []
        if hasattr(env, 'execution_log'):
            bash_execution_log = env.execution_log

        # Get agent conversation messages
        agent_messages = agent.messages if hasattr(agent, 'messages') else []

        # Calculate timing summary
        total_llm_time = 0.0
        total_bash_time = 0.0
        successful_llm_calls = 0
        successful_bash_calls = 0

        # Sum up LLM call times
        for call in model_call_log:
            if 'duration_total_seconds' in call:
                total_llm_time += call['duration_total_seconds']
                successful_llm_calls += 1

        # Sum up bash execution times
        for execution in bash_execution_log:
            if 'duration_seconds' in execution:
                total_bash_time += execution['duration_seconds']
                successful_bash_calls += 1

        # Calculate averages
        avg_llm_time = total_llm_time / successful_llm_calls if successful_llm_calls > 0 else 0.0
        avg_bash_time = total_bash_time / successful_bash_calls if successful_bash_calls > 0 else 0.0

        timing_summary = {
            "total_llm_time_seconds": total_llm_time,
            "total_bash_time_seconds": total_bash_time,
            "average_llm_time_seconds": avg_llm_time,
            "average_bash_time_seconds": avg_bash_time,
            "total_llm_calls": successful_llm_calls,
            "total_bash_calls": successful_bash_calls,
            "llm_time_percentage": (total_llm_time / total_time * 100) if total_time > 0 else 0.0,
            "bash_time_percentage": (total_bash_time / total_time * 100) if total_time > 0 else 0.0,
            "other_time_seconds": max(0, total_time - total_llm_time - total_bash_time)
        }

        benchmark_summary = {
            "dataset": "sorting_algorithms",
            "exit_status": exit_status,
            "result": result,
            "total_runtime": total_time,
            "total_wall_time": total_time,
            "task_preview": task_description[:200] + "..." if len(task_description) > 200 else task_description,
            "model_calls": model.n_calls,
            "model_cost": model.cost,
            "timing_summary": timing_summary,
            "detailed_logs": {
                "model_api_calls": model_call_log,
                "bash_executions": bash_execution_log,
                "agent_messages": agent_messages,
                "total_model_calls": len(model_call_log),
                "total_bash_commands": len(bash_execution_log),
                "conversation_length": len(agent_messages)
            }
        }

        # print(f"Completed in {total_time:.1f}s | Status: {exit_status}")

        return benchmark_summary
    
    def run_fibonacci_benchmark(self) -> Dict[str, Any]:
        """Run CPU-intensive Fibonacci computation benchmark."""
        print("Running Fibonacci benchmark")

        model = self.setup_model()

        # Load proper SWEBench config
        swe_config = self.load_swebench_config()
        agent_config = swe_config['agent']

        env = LocalEnvironment()

        # Create agent config with proper templates
        config = AgentConfig(
            system_template=agent_config.get('system_template', AgentConfig.system_template),
            instance_template=agent_config.get('instance_template', AgentConfig.instance_template),
            action_observation_template=agent_config.get('action_observation_template', AgentConfig.action_observation_template),
            format_error_template=agent_config.get('format_error_template', AgentConfig.format_error_template),
            timeout_template=agent_config.get('timeout_template', AgentConfig.timeout_template),
            step_limit=10,
            cost_limit=3.0
        )

        agent = DefaultAgent(model, env, config_class=lambda **kwargs: config)

        task_description = """
Fibonacci Computation Benchmark

Problem Description:
Write Python code to compute Fibonacci numbers up to 500. Benchmark each method and create a visualization script to compare their performance.

Instructions:
1. Implement multiple Fibonacci algorithms (recursive, iterative, memoization, etc.)
2. Benchmark each method for computing Fibonacci numbers
3. Create a visualization comparing performance
4. Test your implementations to verify correctness

Please implement a complete solution and finish by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""

        start_time = time.time()
        try:
            exit_status, result = agent.run(task_description)
            print(f"Exit status: {exit_status}")
        except Exception as e:
            print(f"Error during Fibonacci benchmark: {e}")
            exit_status, result = "Error", str(e)
        finally:
            total_time = time.time() - start_time

        # Collect detailed interaction logs
        model_call_log = []
        if hasattr(model, 'call_log'):
            model_call_log = model.call_log

        bash_execution_log = []
        if hasattr(env, 'execution_log'):
            bash_execution_log = env.execution_log

        agent_messages = agent.messages if hasattr(agent, 'messages') else []

        # Calculate timing summary
        total_llm_time = sum(call.get('duration_total_seconds', 0) for call in model_call_log)
        total_bash_time = sum(exec.get('duration_seconds', 0) for exec in bash_execution_log)
        successful_llm_calls = len([c for c in model_call_log if 'duration_total_seconds' in c])
        successful_bash_calls = len([e for e in bash_execution_log if 'duration_seconds' in e])

        avg_llm_time = total_llm_time / successful_llm_calls if successful_llm_calls > 0 else 0.0
        avg_bash_time = total_bash_time / successful_bash_calls if successful_bash_calls > 0 else 0.0

        timing_summary = {
            "total_llm_time_seconds": total_llm_time,
            "total_bash_time_seconds": total_bash_time,
            "average_llm_time_seconds": avg_llm_time,
            "average_bash_time_seconds": avg_bash_time,
            "total_llm_calls": successful_llm_calls,
            "total_bash_calls": successful_bash_calls,
            "llm_time_percentage": (total_llm_time / total_time * 100) if total_time > 0 else 0.0,
            "bash_time_percentage": (total_bash_time / total_time * 100) if total_time > 0 else 0.0,
            "other_time_seconds": max(0, total_time - total_llm_time - total_bash_time)
        }

        benchmark_summary = {
            "dataset": "fibonacci",
            "exit_status": exit_status,
            "result": result,
            "total_runtime": total_time,
            "total_wall_time": total_time,
            "task_preview": task_description[:200] + "..." if len(task_description) > 200 else task_description,
            "model_calls": model.n_calls,
            "model_cost": model.cost,
            "timing_summary": timing_summary,
            "detailed_logs": {
                "model_api_calls": model_call_log,
                "bash_executions": bash_execution_log,
                "agent_messages": agent_messages,
                "total_model_calls": len(model_call_log),
                "total_bash_commands": len(bash_execution_log),
                "conversation_length": len(agent_messages)
            }
        }

        print(f"Completed in {total_time:.1f}s | Status: {exit_status}")

        return benchmark_summary

    def run_numerical_integration_benchmark(self) -> Dict[str, Any]:
        """Run CPU-intensive numerical integration benchmark."""
        print("Running Numerical Integration benchmark")

        model = self.setup_model()

        # Load proper SWEBench config
        swe_config = self.load_swebench_config()
        agent_config = swe_config['agent']

        env = LocalEnvironment()

        # Create agent config with proper templates
        config = AgentConfig(
            system_template=agent_config.get('system_template', AgentConfig.system_template),
            instance_template=agent_config.get('instance_template', AgentConfig.instance_template),
            action_observation_template=agent_config.get('action_observation_template', AgentConfig.action_observation_template),
            format_error_template=agent_config.get('format_error_template', AgentConfig.format_error_template),
            timeout_template=agent_config.get('timeout_template', AgentConfig.timeout_template),
            step_limit=12,
            cost_limit=3.5
        )

        agent = DefaultAgent(model, env, config_class=lambda **kwargs: config)

        task_description = """
Numerical Integration Benchmark

Problem Description:
Write Python code to compute numerical integration of sin(x) from 0 to pi using trapezoidal rule with 1000000 and 10000000 steps. Compare numpy.trapz and repeated scipy.integrate.quad calls.

Instructions:
1. Implement the trapezoidal rule for numerical integration using (i)numpy.trapz and (ii) repeated scipy.integrate.quad calls.
2. Integrate sin(x) from 0 to pi with the specified step counts

Please implement a complete solution and finish by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""

        start_time = time.time()
        try:
            exit_status, result = agent.run(task_description)
            print(f"Exit status: {exit_status}")
        except Exception as e:
            print(f"Error during Numerical Integration benchmark: {e}")
            exit_status, result = "Error", str(e)
        finally:
            total_time = time.time() - start_time

        # Collect detailed interaction logs
        model_call_log = getattr(model, 'call_log', [])
        bash_execution_log = getattr(env, 'execution_log', [])
        agent_messages = getattr(agent, 'messages', [])

        # Calculate timing summary
        total_llm_time = sum(call.get('duration_total_seconds', 0) for call in model_call_log)
        total_bash_time = sum(exec.get('duration_seconds', 0) for exec in bash_execution_log)
        successful_llm_calls = len([c for c in model_call_log if 'duration_total_seconds' in c])
        successful_bash_calls = len([e for e in bash_execution_log if 'duration_seconds' in e])

        timing_summary = {
            "total_llm_time_seconds": total_llm_time,
            "total_bash_time_seconds": total_bash_time,
            "average_llm_time_seconds": total_llm_time / successful_llm_calls if successful_llm_calls > 0 else 0.0,
            "average_bash_time_seconds": total_bash_time / successful_bash_calls if successful_bash_calls > 0 else 0.0,
            "total_llm_calls": successful_llm_calls,
            "total_bash_calls": successful_bash_calls,
            "llm_time_percentage": (total_llm_time / total_time * 100) if total_time > 0 else 0.0,
            "bash_time_percentage": (total_bash_time / total_time * 100) if total_time > 0 else 0.0,
            "other_time_seconds": max(0, total_time - total_llm_time - total_bash_time)
        }

        benchmark_summary = {
            "dataset": "numerical_integration",
            "exit_status": exit_status,
            "result": result,
            "total_runtime": total_time,
            "total_wall_time": total_time,
            "task_preview": task_description[:200] + "..." if len(task_description) > 200 else task_description,
            "model_calls": model.n_calls,
            "model_cost": model.cost,
            "timing_summary": timing_summary,
            "detailed_logs": {
                "model_api_calls": model_call_log,
                "bash_executions": bash_execution_log,
                "agent_messages": agent_messages,
                "total_model_calls": len(model_call_log),
                "total_bash_commands": len(bash_execution_log),
                "conversation_length": len(agent_messages)
            }
        }

        print(f"Completed in {total_time:.1f}s | Status: {exit_status}")

        return benchmark_summary
    
    def run_swebench_benchmark(self, instance_ids: List[str] = None, max_instances: int = 10) -> Dict[str, Any]:
        """Run SWEBench benchmark on multiple instances."""
        
        print(f"🚀 Starting SWE-bench benchmark (max: {max_instances} instances)")
        
        # Load the actual SWE-bench dataset
        dataset = None
        dataset_list = []
        
        try:
            print("📚 Loading SWE-bench_Lite dataset...")
            dataset = load_dataset("princeton-nlp/SWE-Bench_Lite", split="test")
            dataset_list = list(dataset)
            print(f"✅ Successfully loaded {len(dataset_list)} real SWE-bench instances")
            
            # If no specific instances requested or using default, use first N real instances
            if instance_ids is None or instance_ids == ["swe-agent__test-repo-1"]:
                instance_ids = [inst["instance_id"] for inst in dataset_list[:max_instances]]
                print(f"📋 Using first {len(instance_ids)} real instances from SWE-bench_Lite:")
                print(f"   {instance_ids[:3]}..." if len(instance_ids) > 3 else f"   {instance_ids}")
            else:
                print(f"📋 Using provided instance IDs: {instance_ids[:3]}..." if len(instance_ids) > 3 else f"📋 Using provided instance IDs: {instance_ids}")
            
        except Exception as e:
            print(f"⚠️  Failed to load SWE-bench_Lite dataset: {e}")
            print("🔄 Trying fallback dataset...")
            try:
                dataset = load_dataset("klieret/swe-bench-dummy-test-dataset", split="test")
                dataset_list = list(dataset)
                if instance_ids is None or instance_ids == ["swe-agent__test-repo-1"]:
                    instance_ids = [inst["instance_id"] for inst in dataset_list[:max_instances]]
                print(f"✅ Using fallback dataset with {len(instance_ids)} instances")
            except Exception as e2:
                print(f"❌ Fallback dataset also failed: {e2}")
                print("🔧 Using synthetic test instance")
                dataset = None
                dataset_list = []
                instance_ids = ["test-instance-1"]
        
        # Initialize results tracking
        all_results = []
        successful_completions = 0
        total_instances = min(len(instance_ids), max_instances)
        results_file = self.output_dir / "swebench_full_benchmark.json"
        
        print(f"\n{'='*60}")
        print(f"STARTING FULL SWE-BENCH BENCHMARK")
        print(f"Model: {self.model_config.get('model_name', 'Unknown')}")
        print(f"Instances to process: {total_instances}")
        print(f"Results will be saved incrementally to: {results_file}")
        print(f"{'='*60}\n")
        
        for i, instance_id in enumerate(instance_ids[:max_instances]):
            print(f"\n[{i+1}/{total_instances}] Processing instance: {instance_id}")
            print("-" * 40)
            
            try:
                result = self._run_single_instance(instance_id, dataset_list, i+1, total_instances)
                all_results.append(result)
                
                # Check if instance completed successfully
                if result.get("exit_status") == "Submitted" or "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in str(result.get("result", "")):
                    successful_completions += 1
                    print(f"✅ Instance {instance_id} completed successfully")
                else:
                    print(f"❌ Instance {instance_id} did not complete")
                
                # Save incremental results after each instance
                self.save_incremental_results(results_file, all_results)
                    
            except Exception as e:
                print(f"❌ Error processing instance {instance_id}: {e}")
                error_result = {
                    "instance_id": instance_id,
                    "error": str(e),
                    "exit_status": "Error",
                    "total_runtime": 0
                }
                all_results.append(error_result)
                
                # Save incremental results even on error
                self.save_incremental_results(results_file, all_results)
        
        # Compile final benchmark summary
        total_time = sum(r.get("total_runtime", 0) for r in all_results)
        success_rate = (successful_completions / total_instances) * 100 if total_instances > 0 else 0
        
        final_summary = {
            "benchmark_type": "swebench_full",
            "model": self.model_config.get('model_name', 'Unknown'),
            "total_instances": total_instances,
            "successful_completions": successful_completions,
            "success_rate_percent": success_rate,
            "total_benchmark_time": total_time,
            "average_time_per_instance": total_time / total_instances if total_instances > 0 else 0,
            "individual_results": all_results,
            "timestamp": time.time(),
            "status": "completed"
        }
        
        # Save final comprehensive results
        with open(results_file, 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK COMPLETED!")
        print(f"Success Rate: {success_rate:.1f}% ({successful_completions}/{total_instances})")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Average Time: {total_time/total_instances:.1f}s per instance")
        print(f"Results saved to: {results_file}")
        print(f"{'='*60}")
        
        return final_summary
    
    def _run_single_instance(self, instance_id: str, dataset_list, current: int, total: int) -> Dict[str, Any]:
        """Run a single SWE-bench instance."""
        print(f"\n🔍 [{current}/{total}] Looking up instance: {instance_id}")
        
        # Find the specific instance in the dataset
        instance = None
        task = None
        
        if dataset_list and len(dataset_list) > 0:
            # Search for the specific instance
            for inst in dataset_list:
                if inst["instance_id"] == instance_id:
                    instance = inst
                    break
            
            if instance is not None:
                task = instance["problem_statement"]
                print(f"   ✅ Found real SWE-bench instance")
                print(f"   📝 Problem statement: {len(task)} characters")
            else:
                # Instance not found in dataset, but dataset exists
                print(f"   ⚠️  Instance '{instance_id}' not found in dataset")
                if instance_id == "test-instance-1":
                    # This is our synthetic test instance
                    task = "You are given a software engineering task. Please implement a simple 'Hello World' Python script. Create a file called hello_world.py that prints 'Hello, World!' when executed."
                    print(f"   🔧 Using synthetic test task")
                else:
                    # Use first real instance as fallback
                    instance = dataset_list[0]
                    task = instance["problem_statement"]
                    print(f"   🔄 Using first dataset instance as fallback")
        else:
            # No dataset available
            print(f"   📝 No dataset available, creating synthetic task")
            task = "You are given a software engineering task. Please implement a simple 'Hello World' Python script. Create a file called hello_world.py that prints 'Hello, World!' when executed."
        
        print(f"   📋 Task preview: {task[:150]}..." if len(task) > 150 else f"   📋 Task: {task}")
        
        model = self.setup_model()
        
        # Load proper SWEBench config
        swe_config = self.load_swebench_config()
        agent_config = swe_config['agent']
        
        # Use LocalEnvironment but with proper timeout
        env = LocalEnvironment()
        
        # Create agent config from YAML with reasonable limits for benchmarking
        config = AgentConfig(
            system_template=agent_config.get('system_template', AgentConfig.system_template),
            instance_template=agent_config.get('instance_template', AgentConfig.instance_template),
            action_observation_template=agent_config.get('action_observation_template', AgentConfig.action_observation_template),
            format_error_template=agent_config.get('format_error_template', AgentConfig.format_error_template),
            timeout_template=agent_config.get('timeout_template', AgentConfig.timeout_template),
            step_limit=50,  # Allow more steps for thorough investigation
            cost_limit=15.0  # Allow higher cost for complex tasks
        )
        
        agent = DefaultAgent(model, env, config_class=lambda **kwargs: config)
        
        # Create test file if needed (for missing colon test)
        if "missing_colon" in task.lower():
            import os
            broken_file_path = os.path.join(os.getcwd(), "missing_colon.py")
            try:
                with open(broken_file_path, 'w') as f:
                    f.write("def division(a: float, b: float) -> float\n    return a / b\n")
                print(f"Created test file: {broken_file_path}")
            except Exception as e:
                print(f"Warning: Could not create test file: {e}")
        
        start_time = time.time()
        try:
            exit_status, result = agent.run(task)
            print(f"Exit status: {exit_status}")
        except Exception as e:
            print(f"Error during execution: {e}")
            exit_status, result = "Error", str(e)
        finally:
            total_time = time.time() - start_time
        
        # Collect detailed interaction logs
        model_call_log = []
        if hasattr(model, 'call_log'):
            model_call_log = model.call_log
        
        bash_execution_log = []
        if hasattr(env, 'execution_log'):
            bash_execution_log = env.execution_log
        
        # Get agent conversation messages
        agent_messages = agent.messages if hasattr(agent, 'messages') else []
        
        # Calculate timing summary
        total_llm_time = 0.0
        total_bash_time = 0.0
        successful_llm_calls = 0
        successful_bash_calls = 0
        
        # Sum up LLM call times
        for call in model_call_log:
            if 'duration_total_seconds' in call:
                total_llm_time += call['duration_total_seconds']
                successful_llm_calls += 1
        
        # Sum up bash execution times
        for execution in bash_execution_log:
            if 'duration_seconds' in execution:
                total_bash_time += execution['duration_seconds']
                successful_bash_calls += 1
        
        # Calculate averages
        avg_llm_time = total_llm_time / successful_llm_calls if successful_llm_calls > 0 else 0.0
        avg_bash_time = total_bash_time / successful_bash_calls if successful_bash_calls > 0 else 0.0
        
        timing_summary = {
            "total_llm_time_seconds": total_llm_time,
            "total_bash_time_seconds": total_bash_time,
            "average_llm_time_seconds": avg_llm_time,
            "average_bash_time_seconds": avg_bash_time,
            "total_llm_calls": successful_llm_calls,
            "total_bash_calls": successful_bash_calls,
            "llm_time_percentage": (total_llm_time / total_time * 100) if total_time > 0 else 0.0,
            "bash_time_percentage": (total_bash_time / total_time * 100) if total_time > 0 else 0.0,
            "other_time_seconds": max(0, total_time - total_llm_time - total_bash_time)
        }
        
        benchmark_summary = {
            "instance_id": instance_id,
            "exit_status": exit_status,
            "result": result,
            "total_runtime": total_time,
            "task_preview": task[:200] + "..." if len(task) > 200 else task,
            "model_calls": model.n_calls,
            "model_cost": model.cost,
            "timing_summary": timing_summary,
            "detailed_logs": {
                "model_api_calls": model_call_log,
                "bash_executions": bash_execution_log,
                "agent_messages": agent_messages,
                "total_model_calls": len(model_call_log),
                "total_bash_commands": len(bash_execution_log),
                "conversation_length": len(agent_messages)
            }
        }
        
        print(f"Completed in {total_time:.1f}s | Status: {exit_status}")
        
        return benchmark_summary
    
    def run_prime_number_benchmark(self) -> Dict[str, Any]:
        """Run CPU-intensive prime number computation benchmark."""
        print("Running Prime Number benchmark")

        model = self.setup_model()

        # Load proper SWEBench config
        swe_config = self.load_swebench_config()
        agent_config = swe_config['agent']

        env = LocalEnvironment()

        # Create agent config with proper templates
        config = AgentConfig(
            system_template=agent_config.get('system_template', AgentConfig.system_template),
            instance_template=agent_config.get('instance_template', AgentConfig.instance_template),
            action_observation_template=agent_config.get('action_observation_template', AgentConfig.action_observation_template),
            format_error_template=agent_config.get('format_error_template', AgentConfig.format_error_template),
            timeout_template=agent_config.get('timeout_template', AgentConfig.timeout_template),
            step_limit=10,
            cost_limit=3.0
        )

        agent = DefaultAgent(model, env, config_class=lambda **kwargs: config)

        task_description = """
Prime Number Computation Benchmark

Problem Description:
Write a Python script to find all prime numbers up to 10000 using the Sieve of Eratosthenes algorithm.

Instructions:
1. Implement the Sieve of Eratosthenes algorithm
2. Find all prime numbers up to 10000
3. Measure and report the computation time
4. Verify the count of primes found (should be 1229)

Please implement a complete solution and finish by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""

        start_time = time.time()
        try:
            exit_status, result = agent.run(task_description)
            print(f"Exit status: {exit_status}")
        except Exception as e:
            print(f"Error during Prime Number benchmark: {e}")
            exit_status, result = "Error", str(e)
        finally:
            total_time = time.time() - start_time

        # Collect detailed interaction logs
        model_call_log = getattr(model, 'call_log', [])
        bash_execution_log = getattr(env, 'execution_log', [])
        agent_messages = getattr(agent, 'messages', [])

        # Calculate timing summary
        total_llm_time = sum(call.get('duration_total_seconds', 0) for call in model_call_log)
        total_bash_time = sum(exec.get('duration_seconds', 0) for exec in bash_execution_log)
        successful_llm_calls = len([c for c in model_call_log if 'duration_total_seconds' in c])
        successful_bash_calls = len([e for e in bash_execution_log if 'duration_seconds' in e])

        timing_summary = {
            "total_llm_time_seconds": total_llm_time,
            "total_bash_time_seconds": total_bash_time,
            "average_llm_time_seconds": total_llm_time / successful_llm_calls if successful_llm_calls > 0 else 0.0,
            "average_bash_time_seconds": total_bash_time / successful_bash_calls if successful_bash_calls > 0 else 0.0,
            "total_llm_calls": successful_llm_calls,
            "total_bash_calls": successful_bash_calls,
            "llm_time_percentage": (total_llm_time / total_time * 100) if total_time > 0 else 0.0,
            "bash_time_percentage": (total_bash_time / total_time * 100) if total_time > 0 else 0.0,
            "other_time_seconds": max(0, total_time - total_llm_time - total_bash_time)
        }

        benchmark_summary = {
            "dataset": "prime_numbers",
            "exit_status": exit_status,
            "result": result,
            "total_runtime": total_time,
            "total_wall_time": total_time,
            "task_preview": task_description[:200] + "..." if len(task_description) > 200 else task_description,
            "model_calls": model.n_calls,
            "model_cost": model.cost,
            "timing_summary": timing_summary,
            "detailed_logs": {
                "model_api_calls": model_call_log,
                "bash_executions": bash_execution_log,
                "agent_messages": agent_messages,
                "total_model_calls": len(model_call_log),
                "total_bash_commands": len(bash_execution_log),
                "conversation_length": len(agent_messages)
            }
        }

        print(f"Completed in {total_time:.1f}s | Status: {exit_status}")

        return benchmark_summary
    
    def run_sudoku_benchmark(self) -> Dict[str, Any]:
        """Run sudoku game development benchmark with complex operations."""
        print("Running Sudoku benchmark")

        model = self.setup_model()

        # Load proper SWEBench config
        swe_config = self.load_swebench_config()
        agent_config = swe_config['agent']

        env = LocalEnvironment()

        # Create agent config with proper templates
        config = AgentConfig(
            system_template=agent_config.get('system_template', AgentConfig.system_template),
            instance_template=agent_config.get('instance_template', AgentConfig.instance_template),
            action_observation_template=agent_config.get('action_observation_template', AgentConfig.action_observation_template),
            format_error_template=agent_config.get('format_error_template', AgentConfig.format_error_template),
            timeout_template=agent_config.get('timeout_template', AgentConfig.timeout_template),
            step_limit=25,
            cost_limit=5.0
        )

        agent = DefaultAgent(model, env, config_class=lambda **kwargs: config)

        task_description = """
Matrix Multiplication Benchmark

Problem Description:
Write a matrix multiplication code using Python and benchmark the latency of three large matrix multiplications - 200x200, 300x300, 400x400 matrices used in LLMs. Also write a visualization script to plot the results.

Instructions:
1. Implement matrix multiplication using NumPy
2. Benchmark matrix multiplication for 800x800, 900x900, and 1000x1000 matrices
3. Measure and report the computation time for each size
4. Create a visualization script to plot performance vs matrix size
5. Test your implementation to verify correctness

Please implement a complete solution and finish by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""

        start_time = time.time()
        try:
            exit_status, result = agent.run(task_description)
            print(f"Exit status: {exit_status}")
        except Exception as e:
            print(f"Error during Sudoku benchmark: {e}")
            exit_status, result = "Error", str(e)
        finally:
            total_time = time.time() - start_time

        # Collect detailed interaction logs
        model_call_log = getattr(model, 'call_log', [])
        bash_execution_log = getattr(env, 'execution_log', [])
        agent_messages = getattr(agent, 'messages', [])

        # Calculate timing summary
        total_llm_time = sum(call.get('duration_total_seconds', 0) for call in model_call_log)
        total_bash_time = sum(exec.get('duration_seconds', 0) for exec in bash_execution_log)
        successful_llm_calls = len([c for c in model_call_log if 'duration_total_seconds' in c])
        successful_bash_calls = len([e for e in bash_execution_log if 'duration_seconds' in e])

        timing_summary = {
            "total_llm_time_seconds": total_llm_time,
            "total_bash_time_seconds": total_bash_time,
            "average_llm_time_seconds": total_llm_time / successful_llm_calls if successful_llm_calls > 0 else 0.0,
            "average_bash_time_seconds": total_bash_time / successful_bash_calls if successful_bash_calls > 0 else 0.0,
            "total_llm_calls": successful_llm_calls,
            "total_bash_calls": successful_bash_calls,
            "llm_time_percentage": (total_llm_time / total_time * 100) if total_time > 0 else 0.0,
            "bash_time_percentage": (total_bash_time / total_time * 100) if total_time > 0 else 0.0,
            "other_time_seconds": max(0, total_time - total_llm_time - total_bash_time)
        }

        benchmark_summary = {
            "dataset": "matrix_multiplication",
            "exit_status": exit_status,
            "result": result,
            "total_runtime": total_time,
            "total_wall_time": total_time,
            "task_preview": task_description[:200] + "..." if len(task_description) > 200 else task_description,
            "model_calls": model.n_calls,
            "model_cost": model.cost,
            "timing_summary": timing_summary,
            "detailed_logs": {
                "model_api_calls": model_call_log,
                "bash_executions": bash_execution_log,
                "agent_messages": agent_messages,
                "total_model_calls": len(model_call_log),
                "total_bash_commands": len(bash_execution_log),
                "conversation_length": len(agent_messages)
            }
        }

        print(f"Completed in {total_time:.1f}s | Status: {exit_status}")

        return benchmark_summary

    def run_lu_decomposition_benchmark(self) -> Dict[str, Any]:
        """Run vectorized NumPy LU decomposition benchmark without using linalg.lu."""
        print("Running LU Decomposition benchmark")

        model = self.setup_model()

        # Load proper SWEBench config
        swe_config = self.load_swebench_config()
        agent_config = swe_config['agent']

        env = LocalEnvironment()

        # Create agent config with proper templates
        config = AgentConfig(
            system_template=agent_config.get('system_template', AgentConfig.system_template),
            instance_template=agent_config.get('instance_template', AgentConfig.instance_template),
            action_observation_template=agent_config.get('action_observation_template', AgentConfig.action_observation_template),
            format_error_template=agent_config.get('format_error_template', AgentConfig.format_error_template),
            timeout_template=agent_config.get('timeout_template', AgentConfig.timeout_template),
            step_limit=20,
            cost_limit=5.0
        )

        agent = DefaultAgent(model, env, config_class=lambda **kwargs: config)

        task_description = """
Vectorized NumPy LU Decomposition Benchmark

Problem Description:
Implement LU decomposition using vectorized NumPy operations (without calling linalg.lu) and benchmark it for matrices of sizes N={400, 800, 1200}. Compare the performance against a naive triple-loop implementation.

Instructions:
1. Implement a naive triple-loop LU decomposition
2. Implement a vectorized NumPy LU decomposition (without using linalg.lu)
3. Benchmark both implementations on 400x400, 800x800, and 1200x1200 matrices
4. Measure and report computation time for each matrix size and method
5. Compare results: compute the difference between your implementation and numpy.linalg.lu to verify correctness
6. Create a visualization comparing naive vs vectorized performance

Requirements:
- Use NumPy for vectorized operations but DO NOT use numpy.linalg.lu or scipy.linalg.lu
- Implement Doolittle's method (L has 1s on diagonal, U is upper triangular)
- Validate your implementation produces correct results
- Include error analysis comparing against numpy.linalg.lu

Please implement a complete solution and finish by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""

        start_time = time.time()
        try:
            exit_status, result = agent.run(task_description)
            print(f"Exit status: {exit_status}")
        except Exception as e:
            print(f"Error during LU Decomposition benchmark: {e}")
            exit_status, result = "Error", str(e)
        finally:
            total_time = time.time() - start_time

        # Collect detailed interaction logs
        model_call_log = getattr(model, 'call_log', [])
        bash_execution_log = getattr(env, 'execution_log', [])
        agent_messages = getattr(agent, 'messages', [])

        # Calculate timing summary
        total_llm_time = sum(call.get('duration_total_seconds', 0) for call in model_call_log)
        total_bash_time = sum(exec.get('duration_seconds', 0) for exec in bash_execution_log)
        successful_llm_calls = len([c for c in model_call_log if 'duration_total_seconds' in c])
        successful_bash_calls = len([e for e in bash_execution_log if 'duration_seconds' in e])

        timing_summary = {
            "total_llm_time_seconds": total_llm_time,
            "total_bash_time_seconds": total_bash_time,
            "average_llm_time_seconds": total_llm_time / successful_llm_calls if successful_llm_calls > 0 else 0.0,
            "average_bash_time_seconds": total_bash_time / successful_bash_calls if successful_bash_calls > 0 else 0.0,
            "total_llm_calls": successful_llm_calls,
            "total_bash_calls": successful_bash_calls,
            "llm_time_percentage": (total_llm_time / total_time * 100) if total_time > 0 else 0.0,
            "bash_time_percentage": (total_bash_time / total_time * 100) if total_time > 0 else 0.0,
            "other_time_seconds": max(0, total_time - total_llm_time - total_bash_time)
        }

        benchmark_summary = {
            "dataset": "lu_decomposition",
            "exit_status": exit_status,
            "result": result,
            "total_runtime": total_time,
            "total_wall_time": total_time,
            "task_preview": task_description[:200] + "..." if len(task_description) > 200 else task_description,
            "model_calls": model.n_calls,
            "model_cost": model.cost,
            "timing_summary": timing_summary,
            "detailed_logs": {
                "model_api_calls": model_call_log,
                "bash_executions": bash_execution_log,
                "agent_messages": agent_messages,
                "total_model_calls": len(model_call_log),
                "total_bash_commands": len(bash_execution_log),
                "conversation_length": len(agent_messages)
            }
        }

        print(f"Completed in {total_time:.1f}s | Status: {exit_status}")

        return benchmark_summary

    def run_knn_benchmark(self) -> Dict[str, Any]:
        """Run NumPy k-NN benchmark without scikit-learn."""
        print("Running k-NN benchmark")

        model = self.setup_model()

        # Load proper SWEBench config
        swe_config = self.load_swebench_config()
        agent_config = swe_config['agent']

        env = LocalEnvironment()

        # Create agent config with proper templates - optimized for quick completion
        config = AgentConfig(
            system_template=agent_config.get('system_template', AgentConfig.system_template),
            instance_template=agent_config.get('instance_template', AgentConfig.instance_template),
            action_observation_template=agent_config.get('action_observation_template', AgentConfig.action_observation_template),
            format_error_template=agent_config.get('format_error_template', AgentConfig.format_error_template),
            timeout_template=agent_config.get('timeout_template', AgentConfig.timeout_template),
            step_limit=8,  # Low limit for quick completion
            cost_limit=2.5
        )

        agent = DefaultAgent(model, env, config_class=lambda **kwargs: config)

        task_description = """
k-NN Benchmark 

Problem Description:
Implement a naive k-Nearest Neighbors classifier for k=5 on random datasets with shapes (4k×32) and (6k×32). Report latency and memory usage.


Please implement a complete solution and finish by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""

        start_time = time.time()
        try:
            exit_status, result = agent.run(task_description)
            print(f"Exit status: {exit_status}")
        except Exception as e:
            print(f"Error during k-NN benchmark: {e}")
            exit_status, result = "Error", str(e)
        finally:
            total_time = time.time() - start_time

        # Collect detailed interaction logs
        model_call_log = getattr(model, 'call_log', [])
        bash_execution_log = getattr(env, 'execution_log', [])
        agent_messages = getattr(agent, 'messages', [])

        # Calculate timing summary
        total_llm_time = sum(call.get('duration_total_seconds', 0) for call in model_call_log)
        total_bash_time = sum(exec.get('duration_seconds', 0) for exec in bash_execution_log)
        successful_llm_calls = len([c for c in model_call_log if 'duration_total_seconds' in c])
        successful_bash_calls = len([e for e in bash_execution_log if 'duration_seconds' in e])

        timing_summary = {
            "total_llm_time_seconds": total_llm_time,
            "total_bash_time_seconds": total_bash_time,
            "average_llm_time_seconds": total_llm_time / successful_llm_calls if successful_llm_calls > 0 else 0.0,
            "average_bash_time_seconds": total_bash_time / successful_bash_calls if successful_bash_calls > 0 else 0.0,
            "total_llm_calls": successful_llm_calls,
            "total_bash_calls": successful_bash_calls,
            "llm_time_percentage": (total_llm_time / total_time * 100) if total_time > 0 else 0.0,
            "bash_time_percentage": (total_bash_time / total_time * 100) if total_time > 0 else 0.0,
            "other_time_seconds": max(0, total_time - total_llm_time - total_bash_time)
        }

        benchmark_summary = {
            "dataset": "knn_numpy",
            "exit_status": exit_status,
            "result": result,
            "total_runtime": total_time,
            "total_wall_time": total_time,
            "task_preview": task_description[:200] + "..." if len(task_description) > 200 else task_description,
            "model_calls": model.n_calls,
            "model_cost": model.cost,
            "timing_summary": timing_summary,
            "detailed_logs": {
                "model_api_calls": model_call_log,
                "bash_executions": bash_execution_log,
                "agent_messages": agent_messages,
                "total_model_calls": len(model_call_log),
                "total_bash_commands": len(bash_execution_log),
                "conversation_length": len(agent_messages)
            }
        }

        print(f"Completed in {total_time:.1f}s | Status: {exit_status}")

        return benchmark_summary

    def run_fft_convolution_benchmark(self) -> Dict[str, Any]:
        """Run FFT-based 1D convolution benchmark: pure-Python FFT vs NumPy FFT."""
        print("Running FFT-based Convolution benchmark")

        model = self.setup_model()

        # Load proper SWEBench config
        swe_config = self.load_swebench_config()
        agent_config = swe_config['agent']

        env = LocalEnvironment()

        # Create agent config with proper templates - optimized for quick completion
        config = AgentConfig(
            system_template=agent_config.get('system_template', AgentConfig.system_template),
            instance_template=agent_config.get('instance_template', AgentConfig.instance_template),
            action_observation_template=agent_config.get('action_observation_template', AgentConfig.action_observation_template),
            format_error_template=agent_config.get('format_error_template', AgentConfig.format_error_template),
            timeout_template=agent_config.get('timeout_template', AgentConfig.timeout_template),
            step_limit=10,  # Moderate limit for FFT implementation
            cost_limit=3.0
        )

        agent = DefaultAgent(model, env, config_class=lambda **kwargs: config)

        task_description = """
FFT-based 1D Convolution Benchmark

Problem Description:
Implement FFT-based 1D convolution comparing pure-Python FFT vs NumPy FFT on signals of 1e6 and 5e6 samples. Plot the speedup.

Instructions:
1. Implement a simple pure-Python FFT (Cooley-Tukey algorithm, power-of-2 only)
2. Implement FFT-based convolution using both pure-Python and NumPy FFT
3. Generate random signals: 1e6 samples and 5e6 samples
4. Use a small kernel (e.g., 1000 samples) for convolution
5. Benchmark both methods on both signal sizes
6. Calculate and report speedup (NumPy FFT vs pure-Python FFT)
7. Create a visualization showing speedup vs signal size

Requirements:
- Implement basic Cooley-Tukey FFT in pure Python (recursive or iterative)
- Use numpy.fft.fft for NumPy comparison
- Measure wall-clock time for each method
- Report: execution time, speedup factor
- Plot speedup as bar chart or line graph
- Keep pure-Python FFT simple (focus on correctness, not optimization)

Please implement a complete solution and finish by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""

        start_time = time.time()
        try:
            exit_status, result = agent.run(task_description)
            print(f"Exit status: {exit_status}")
        except Exception as e:
            print(f"Error during FFT Convolution benchmark: {e}")
            exit_status, result = "Error", str(e)
        finally:
            total_time = time.time() - start_time

        # Collect detailed interaction logs
        model_call_log = getattr(model, 'call_log', [])
        bash_execution_log = getattr(env, 'execution_log', [])
        agent_messages = getattr(agent, 'messages', [])

        # Calculate timing summary
        total_llm_time = sum(call.get('duration_total_seconds', 0) for call in model_call_log)
        total_bash_time = sum(exec.get('duration_seconds', 0) for exec in bash_execution_log)
        successful_llm_calls = len([c for c in model_call_log if 'duration_total_seconds' in c])
        successful_bash_calls = len([e for e in bash_execution_log if 'duration_seconds' in e])

        timing_summary = {
            "total_llm_time_seconds": total_llm_time,
            "total_bash_time_seconds": total_bash_time,
            "average_llm_time_seconds": total_llm_time / successful_llm_calls if successful_llm_calls > 0 else 0.0,
            "average_bash_time_seconds": total_bash_time / successful_bash_calls if successful_bash_calls > 0 else 0.0,
            "total_llm_calls": successful_llm_calls,
            "total_bash_calls": successful_bash_calls,
            "llm_time_percentage": (total_llm_time / total_time * 100) if total_time > 0 else 0.0,
            "bash_time_percentage": (total_bash_time / total_time * 100) if total_time > 0 else 0.0,
            "other_time_seconds": max(0, total_time - total_llm_time - total_bash_time)
        }

        benchmark_summary = {
            "dataset": "fft_convolution",
            "exit_status": exit_status,
            "result": result,
            "total_runtime": total_time,
            "total_wall_time": total_time,
            "task_preview": task_description[:200] + "..." if len(task_description) > 200 else task_description,
            "model_calls": model.n_calls,
            "model_cost": model.cost,
            "timing_summary": timing_summary,
            "detailed_logs": {
                "model_api_calls": model_call_log,
                "bash_executions": bash_execution_log,
                "agent_messages": agent_messages,
                "total_model_calls": len(model_call_log),
                "total_bash_commands": len(bash_execution_log),
                "conversation_length": len(agent_messages)
            }
        }

        print(f"Completed in {total_time:.1f}s | Status: {exit_status}")

        return benchmark_summary

    def run_scicode_benchmark(self, problem_ids: List[str] = None, max_problems: int = 5, scicode_data_path: str = "./scicode_data") -> Dict[str, Any]:
        """Run SciCode benchmark on scientific computing problems."""
        
        print(f"🔬 Starting SciCode benchmark (max: {max_problems} problems)")
        
        # Initialize results tracking
        all_results = []
        successful_completions = 0
        results_file = self.output_dir / "scicode_benchmark.json"
        
        # Load SciCode problems (simplified approach)
        available_problems = [
            {
                "problem_id": "scicode_001",
                "domain": "Physics",
                "title": "Quantum Harmonic Oscillator Simulation", 
                "description": "Implement a quantum harmonic oscillator simulation using numerical methods. Calculate energy eigenvalues and plot wavefunctions for the first 5 energy levels.",
                "difficulty": "intermediate"
            },
            {
                "problem_id": "scicode_002",
                "domain": "DS-1000",
                "title": "csv", 
                "description": "Load a 5M-row CSV (synthetic if missing), compute grouped means and rolling stats in Pandas; measure wall-clock CPU time.",
                "difficulty": "intermediate"
            },
            {
                "problem_id": "scicode_006", 
                "domain": "Mathematics",
                "title": "FFT-based Signal Processing",
                "description": "Implement Fast Fourier Transform from scratch and use it to analyze a noisy signal. Remove noise and reconstruct the original signal.",
                "difficulty": "advanced"
            },
            {
                "problem_id": "scicode_003",
                "domain": "Chemistry", 
                "title": "Molecular Dynamics Simulation",
                "description": "Create a molecular dynamics simulation for a simple 2D gas system. Calculate temperature, pressure, and radial distribution functions.",
                "difficulty": "advanced"
            },
            {
                "problem_id": "scicode_004",
                "domain": "Biology",
                "title": "Population Dynamics Modeling",
                "description": "Implement and solve the Lotka-Volterra predator-prey equations. Analyze stability and create phase space plots.",
                "difficulty": "intermediate"  
            },
            {
                "problem_id": "scicode_005",
                "domain": "Material Science", 
                "title": "Crystal Structure Analysis",
                "description": "Generate a crystal lattice structure and calculate X-ray diffraction patterns. Compare with experimental data.",
                "difficulty": "advanced"
            },
                        {
                "problem_id": "scicode_007",
                "domain": "Quantum", 
                "title": "N_tangle",
                "description": "Write a function that returns the tensor product of matrices. Using this tensor function, write a function to compute the $n$-tangle of an $n$-qubit pure state for even $n$. Test the function against different test cases and plot the results.",
                "difficulty": "advanced"
            }


        ]
        
        # Select problems to run
        if problem_ids is None:
            selected_problems = available_problems[:max_problems]
        else:
            selected_problems = [p for p in available_problems if p["problem_id"] in problem_ids][:max_problems]
        
        total_problems = len(selected_problems)
        
        print(f"\n{'='*60}")
        print(f"STARTING SCICODE BENCHMARK")
        print(f"Model: {self.model_config.get('model_name', 'Unknown')}")
        print(f"Problems to process: {total_problems}")
        print(f"Results will be saved incrementally to: {results_file}")
        print(f"{'='*60}\n")
        
        for i, problem in enumerate(selected_problems):
            print(f"\n[{i+1}/{total_problems}] Processing problem: {problem['problem_id']}")
            print("-" * 40)
            
            try:
                result = self._run_single_scicode_problem(problem, i+1, total_problems)
                all_results.append(result)
                
                # Check if problem completed successfully
                if result.get("exit_status") == "Submitted" or "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in str(result.get("result", "")):
                    successful_completions += 1
                    print(f"✅ Problem {problem['problem_id']} completed successfully")
                else:
                    print(f"❌ Problem {problem['problem_id']} did not complete")
                
                # Save incremental results
                self.save_incremental_results(results_file, all_results, {"benchmark_type": "scicode"})
                    
            except Exception as e:
                print(f"❌ Error processing problem {problem['problem_id']}: {e}")
                error_result = {
                    "problem_id": problem['problem_id'],
                    "error": str(e),
                    "exit_status": "Error",
                    "total_runtime": 0
                }
                all_results.append(error_result)
                self.save_incremental_results(results_file, all_results, {"benchmark_type": "scicode"})
        
        # Compile final benchmark summary
        total_time = sum(r.get("total_runtime", 0) for r in all_results)
        success_rate = (successful_completions / total_problems) * 100 if total_problems > 0 else 0
        
        final_summary = {
            "benchmark_type": "scicode",
            "model": self.model_config.get('model_name', 'Unknown'),
            "total_problems": total_problems,
            "successful_completions": successful_completions,
            "success_rate_percent": success_rate,
            "total_benchmark_time": total_time,
            "average_time_per_problem": total_time / total_problems if total_problems > 0 else 0,
            "individual_results": all_results,
            "timestamp": time.time(),
            "status": "completed"
        }
        
        # Save final results
        with open(results_file, 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"SCICODE BENCHMARK COMPLETED!")
        print(f"Success Rate: {success_rate:.1f}% ({successful_completions}/{total_problems})")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Average Time: {total_time/total_problems:.1f}s per problem")
        print(f"Results saved to: {results_file}")
        print(f"{'='*60}")
        
        return final_summary

    def _run_single_scicode_problem(self, problem: Dict[str, Any], current: int, total: int) -> Dict[str, Any]:
        """Run a single SciCode problem."""
        print(f"\n🔍 [{current}/{total}] Processing problem: {problem['problem_id']}")
        print(f"   📝 Domain: {problem['domain']}")
        print(f"   📋 Title: {problem['title']}")
        print(f"   💡 Difficulty: {problem['difficulty']}")
        
        # Create detailed task description for the agent
        task_description = f"""
Scientific Computing Problem: {problem['title']}

Domain: {problem['domain']}
Difficulty: {problem['difficulty']}

Problem Description:
{problem['description']}

Instructions:
1. Analyze the problem requirements carefully
2. Import necessary scientific computing libraries (numpy, scipy, matplotlib, etc.)
3. Implement the required algorithms/simulations
4. Include proper error handling and validation
5. Generate visualizations where appropriate
6. Test your implementation with sample data
7. Document your approach and results

Requirements:
- Write clean, well-documented Python code
- Use appropriate numerical methods and algorithms
- Include performance considerations for computational efficiency
- Validate results where possible against analytical solutions

Please implement a complete solution and finish by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""
        
        model = self.setup_model()
        
        # Load proper SciCode config (similar to SWEBench)
        swe_config = self.load_swebench_config()
        agent_config = swe_config['agent']
        
        # Use LocalEnvironment
        env = LocalEnvironment()
        
        # Create agent config with higher limits for complex scientific problems
        config = AgentConfig(
            system_template=agent_config.get('system_template', AgentConfig.system_template),
            instance_template=agent_config.get('instance_template', AgentConfig.instance_template),
            action_observation_template=agent_config.get('action_observation_template', AgentConfig.action_observation_template),
            format_error_template=agent_config.get('format_error_template', AgentConfig.format_error_template),
            timeout_template=agent_config.get('timeout_template', AgentConfig.timeout_template),
            step_limit=100,  # Higher limit for complex scientific computing
            cost_limit=25.0  # Higher cost limit for computational problems
        )
        
        agent = DefaultAgent(model, env, config_class=lambda **kwargs: config)
        
        start_time = time.time()
        try:
            exit_status, result = agent.run(task_description)
            print(f"Exit status: {exit_status}")
        except Exception as e:
            print(f"Error during execution: {e}")
            exit_status, result = "Error", str(e)
        finally:
            total_time = time.time() - start_time
        
        # Collect detailed interaction logs (same as SWEBench)
        model_call_log = []
        if hasattr(model, 'call_log'):
            model_call_log = model.call_log
        
        bash_execution_log = []
        if hasattr(env, 'execution_log'):
            bash_execution_log = env.execution_log
        
        agent_messages = agent.messages if hasattr(agent, 'messages') else []
        
        # Calculate timing summary (same as other benchmarks)
        total_llm_time = 0.0
        total_bash_time = 0.0
        successful_llm_calls = 0
        successful_bash_calls = 0
        
        for call in model_call_log:
            if 'duration_total_seconds' in call:
                total_llm_time += call['duration_total_seconds']
                successful_llm_calls += 1
        
        for execution in bash_execution_log:
            if 'duration_seconds' in execution:
                total_bash_time += execution['duration_seconds']
                successful_bash_calls += 1
        
        avg_llm_time = total_llm_time / successful_llm_calls if successful_llm_calls > 0 else 0.0
        avg_bash_time = total_bash_time / successful_bash_calls if successful_bash_calls > 0 else 0.0
        
        timing_summary = {
            "total_llm_time_seconds": total_llm_time,
            "total_bash_time_seconds": total_bash_time,
            "average_llm_time_seconds": avg_llm_time,
            "average_bash_time_seconds": avg_bash_time,
            "total_llm_calls": successful_llm_calls,
            "total_bash_calls": successful_bash_calls,
            "llm_time_percentage": (total_llm_time / total_time * 100) if total_time > 0 else 0.0,
            "bash_time_percentage": (total_bash_time / total_time * 100) if total_time > 0 else 0.0,
            "other_time_seconds": max(0, total_time - total_llm_time - total_bash_time)
        }
        
        benchmark_summary = {
            "problem_id": problem['problem_id'],
            "domain": problem['domain'], 
            "title": problem['title'],
            "difficulty": problem['difficulty'],
            "exit_status": exit_status,
            "result": result,
            "total_runtime": total_time,
            "task_preview": problem['description'][:200] + "..." if len(problem['description']) > 200 else problem['description'],
            "model_calls": model.n_calls,
            "model_cost": model.cost,
            "timing_summary": timing_summary,
            "detailed_logs": {
                "model_api_calls": model_call_log,
                "bash_executions": bash_execution_log,
                "agent_messages": agent_messages,
                "total_model_calls": len(model_call_log),
                "total_bash_commands": len(bash_execution_log),
                "conversation_length": len(agent_messages)
            }
        }
        
        print(f"Completed in {total_time:.1f}s | Status: {exit_status}")
        
        return benchmark_summary
    
    def run_livecodebench_benchmark(self, problem_ids: List[str] = None, max_problems: int = 5, scenario: str = "code_generation") -> Dict[str, Any]:
        """Run LiveCodeBench benchmark on code-related tasks."""
        
        print(f"💻 Starting LiveCodeBench benchmark (max: {max_problems} problems, scenario: {scenario})")
        
        # Initialize results tracking
        all_results = []
        successful_completions = 0
        results_file = self.output_dir / "livecodebench_benchmark.json"
        
        # LiveCodeBench-style problems (simplified approach based on their website info)
        available_problems = [
            {
                "problem_id": "lcb_001",
                "platform": "LeetCode",
                "title": "Two Sum with Constraints",
                "difficulty": "medium",
                "scenario": "code_generation",
                "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. The array size can be up to 10^4 elements.",
                "test_cases": [
                    {"input": "[2,7,11,15], target=9", "expected": "[0,1]"},
                    {"input": "[3,2,4], target=6", "expected": "[1,2]"}
                ]
            },
            {
                "problem_id": "lcb_002", 
                "platform": "AtCoder",
                "title": "Dynamic Programming - Fibonacci Variants",
                "difficulty": "hard",
                "scenario": "code_generation",
                "description": "Implement a variant of Fibonacci sequence where F(n) = F(n-1) + F(n-2) + F(n-3) for n >= 3, with F(0)=0, F(1)=1, F(2)=1. Calculate F(n) modulo 10^9+7 for large n (up to 10^6).",
                "test_cases": [
                    {"input": "n=5", "expected": "7"},
                    {"input": "n=10", "expected": "149"}
                ]
            },
            {
                "problem_id": "lcb_003",
                "platform": "Codeforces", 
                "title": "Graph Shortest Path with Obstacles",
                "difficulty": "hard",
                "scenario": "code_generation",
                "description": "Find the shortest path in a grid from top-left to bottom-right, where some cells are blocked. You can move in 4 directions (up, down, left, right). Implement using BFS or Dijkstra's algorithm.",
                "test_cases": [
                    {"input": "3x3 grid with obstacles at (1,1)", "expected": "path_length=4"},
                    {"input": "5x5 grid with multiple obstacles", "expected": "optimal_path"}
                ]
            },
            {
                "problem_id": "lcb_004",
                "platform": "LeetCode",
                "title": "Binary Tree Maximum Path Sum",
                "difficulty": "hard", 
                "scenario": "self_repair",
                "description": "Given a non-empty binary tree, find the maximum path sum. A path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.",
                "buggy_code": "def maxPathSum(root):\n    if not root:\n        return 0\n    left = maxPathSum(root.left)\n    right = maxPathSum(root.right)\n    return root.val + left + right  # Bug: doesn't handle negative paths correctly",
                "test_cases": [
                    {"input": "[1,2,3]", "expected": "6"},
                    {"input": "[-10,9,20,null,null,15,7]", "expected": "42"}
                ]
            },
            {
                "problem_id": "lcb_005",
                "platform": "AtCoder",
                "title": "Array Manipulation with Range Updates", 
                "difficulty": "medium",
                "scenario": "test_output_prediction",
                "description": "Given an array and a series of range update operations, predict the final state of the array. Each operation adds a value to all elements in a given range [l, r].",
                "code": "def range_update(arr, operations):\n    for l, r, val in operations:\n        for i in range(l, r+1):\n            arr[i] += val\n    return arr",
                "test_input": "arr=[1,2,3,4,5], operations=[(0,2,10), (1,3,-5)]",
                "expected_output": "[11, 7, 8, -1, 5]"
            }
        ]
        
        # Select problems based on scenario
        if scenario == "self_repair":
            selected_problems = [p for p in available_problems if p["scenario"] == "self_repair"][:max_problems]
        elif scenario == "test_output_prediction":
            selected_problems = [p for p in available_problems if p["scenario"] == "test_output_prediction"][:max_problems]
        else:  # code_generation (default)
            selected_problems = [p for p in available_problems if p["scenario"] == "code_generation"][:max_problems]
        
        # If not enough problems of specific scenario, fill with others
        if len(selected_problems) < max_problems:
            remaining = [p for p in available_problems if p not in selected_problems]
            selected_problems.extend(remaining[:max_problems - len(selected_problems)])
        
        if problem_ids is not None:
            selected_problems = [p for p in selected_problems if p["problem_id"] in problem_ids][:max_problems]
        
        total_problems = len(selected_problems)
        
        print(f"\n{'='*60}")
        print(f"STARTING LIVECODEBENCH BENCHMARK")
        print(f"Model: {self.model_config.get('model_name', 'Unknown')}")
        print(f"Scenario: {scenario}")
        print(f"Problems to process: {total_problems}")
        print(f"Results will be saved incrementally to: {results_file}")
        print(f"{'='*60}\n")
        
        for i, problem in enumerate(selected_problems):
            print(f"\n[{i+1}/{total_problems}] Processing problem: {problem['problem_id']}")
            print("-" * 40)
            
            try:
                result = self._run_single_livecodebench_problem(problem, i+1, total_problems, scenario)
                all_results.append(result)
                
                # Check if problem completed successfully
                if result.get("exit_status") == "Submitted" or "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in str(result.get("result", "")):
                    successful_completions += 1
                    print(f"✅ Problem {problem['problem_id']} completed successfully")
                else:
                    print(f"❌ Problem {problem['problem_id']} did not complete")
                
                # Save incremental results
                self.save_incremental_results(results_file, all_results, {"benchmark_type": "livecodebench", "scenario": scenario})
                    
            except Exception as e:
                print(f"❌ Error processing problem {problem['problem_id']}: {e}")
                error_result = {
                    "problem_id": problem['problem_id'],
                    "error": str(e),
                    "exit_status": "Error",
                    "total_runtime": 0
                }
                all_results.append(error_result)
                self.save_incremental_results(results_file, all_results, {"benchmark_type": "livecodebench", "scenario": scenario})
        
        # Compile final benchmark summary
        total_time = sum(r.get("total_runtime", 0) for r in all_results)
        success_rate = (successful_completions / total_problems) * 100 if total_problems > 0 else 0
        
        final_summary = {
            "benchmark_type": "livecodebench",
            "scenario": scenario,
            "model": self.model_config.get('model_name', 'Unknown'),
            "total_problems": total_problems,
            "successful_completions": successful_completions,
            "success_rate_percent": success_rate,
            "total_benchmark_time": total_time,
            "average_time_per_problem": total_time / total_problems if total_problems > 0 else 0,
            "individual_results": all_results,
            "timestamp": time.time(),
            "status": "completed"
        }
        
        # Save final results
        with open(results_file, 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"LIVECODEBENCH BENCHMARK COMPLETED!")
        print(f"Scenario: {scenario}")
        print(f"Success Rate: {success_rate:.1f}% ({successful_completions}/{total_problems})")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Average Time: {total_time/total_problems:.1f}s per problem")
        print(f"Results saved to: {results_file}")
        print(f"{'='*60}")
        
        return final_summary

    def _run_single_livecodebench_problem(self, problem: Dict[str, Any], current: int, total: int, scenario: str) -> Dict[str, Any]:
        """Run a single LiveCodeBench problem."""
        print(f"\n🔍 [{current}/{total}] Processing problem: {problem['problem_id']}")
        print(f"   💻 Platform: {problem['platform']}")
        print(f"   📋 Title: {problem['title']}")
        print(f"   💡 Difficulty: {problem['difficulty']}")
        print(f"   🎯 Scenario: {problem['scenario']}")
        
        # Create task description based on scenario
        if scenario == "code_generation" or problem["scenario"] == "code_generation":
            task_description = f"""
LiveCodeBench Code Generation Problem: {problem['title']}

Platform: {problem['platform']}
Difficulty: {problem['difficulty']}

Problem Description:
{problem['description']}

Test Cases:
{chr(10).join([f"- Input: {tc['input']}, Expected: {tc['expected']}" for tc in problem.get('test_cases', [])])}

Instructions:
1. Analyze the problem requirements carefully
2. Design an efficient algorithm considering time/space complexity
3. Implement the solution in Python with proper error handling
4. Test your solution with the provided test cases
5. Optimize for the given constraints
6. Add comments explaining your approach

Requirements:
- Write clean, efficient Python code
- Handle edge cases appropriately
- Consider time and space complexity
- Validate your solution with test cases

Please implement a complete solution and finish by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""
        
        elif scenario == "self_repair" or problem["scenario"] == "self_repair":
            task_description = f"""
LiveCodeBench Self-Repair Problem: {problem['title']}

Platform: {problem['platform']}
Difficulty: {problem['difficulty']}

Problem Description:
{problem['description']}

Buggy Code:
```python
{problem.get('buggy_code', 'No buggy code provided')}
```

Test Cases:
{chr(10).join([f"- Input: {tc['input']}, Expected: {tc['expected']}" for tc in problem.get('test_cases', [])])}

Instructions:
1. Analyze the provided buggy code
2. Identify the bugs and issues
3. Fix the code to handle all edge cases correctly
4. Test the repaired code with provided test cases
5. Explain what was wrong and how you fixed it

Requirements:
- Fix all bugs in the provided code
- Ensure the solution passes all test cases
- Maintain the original algorithm structure if possible
- Add comments explaining the fixes

Please repair the code and finish by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""
        
        elif scenario == "test_output_prediction" or problem["scenario"] == "test_output_prediction":
            task_description = f"""
LiveCodeBench Test Output Prediction Problem: {problem['title']}

Platform: {problem['platform']}
Difficulty: {problem['difficulty']}

Problem Description:
{problem['description']}

Given Code:
```python
{problem.get('code', 'No code provided')}
```

Test Input:
{problem.get('test_input', 'No test input provided')}

Instructions:
1. Analyze the provided code carefully
2. Trace through the execution with the given input
3. Predict the exact output the code will produce
4. Verify your prediction by running the code
5. Explain your reasoning step by step

Expected Output:
{problem.get('expected_output', 'Predict this!')}

Requirements:
- Trace through the code execution step by step
- Predict the exact output format
- Verify by actually running the code
- Explain any complex logic or edge cases

Please predict and verify the output, then finish by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""
        
        else:
            # Fallback to code generation
            task_description = f"""
LiveCodeBench Problem: {problem['title']}

Platform: {problem['platform']}
Difficulty: {problem['difficulty']}

{problem['description']}

Please solve this problem and finish by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""
        
        model = self.setup_model()
        
        # Load proper config
        swe_config = self.load_swebench_config()
        agent_config = swe_config['agent']
        
        # Use LocalEnvironment
        env = LocalEnvironment()
        
        # Create agent config with appropriate limits for coding problems
        config = AgentConfig(
            system_template=agent_config.get('system_template', AgentConfig.system_template),
            instance_template=agent_config.get('instance_template', AgentConfig.instance_template),
            action_observation_template=agent_config.get('action_observation_template', AgentConfig.action_observation_template),
            format_error_template=agent_config.get('format_error_template', AgentConfig.format_error_template),
            timeout_template=agent_config.get('timeout_template', AgentConfig.timeout_template),
            step_limit=75,  # Reasonable limit for coding problems
            cost_limit=20.0  # Allow sufficient cost for complex problems
        )
        
        agent = DefaultAgent(model, env, config_class=lambda **kwargs: config)
        
        start_time = time.time()
        try:
            exit_status, result = agent.run(task_description)
            print(f"Exit status: {exit_status}")
        except Exception as e:
            print(f"Error during execution: {e}")
            exit_status, result = "Error", str(e)
        finally:
            total_time = time.time() - start_time
        
        # Collect detailed interaction logs
        model_call_log = []
        if hasattr(model, 'call_log'):
            model_call_log = model.call_log
        
        bash_execution_log = []
        if hasattr(env, 'execution_log'):
            bash_execution_log = env.execution_log
        
        agent_messages = agent.messages if hasattr(agent, 'messages') else []
        
        # Calculate timing summary
        total_llm_time = 0.0
        total_bash_time = 0.0
        successful_llm_calls = 0
        successful_bash_calls = 0
        
        for call in model_call_log:
            if 'duration_total_seconds' in call:
                total_llm_time += call['duration_total_seconds']
                successful_llm_calls += 1
        
        for execution in bash_execution_log:
            if 'duration_seconds' in execution:
                total_bash_time += execution['duration_seconds']
                successful_bash_calls += 1
        
        avg_llm_time = total_llm_time / successful_llm_calls if successful_llm_calls > 0 else 0.0
        avg_bash_time = total_bash_time / successful_bash_calls if successful_bash_calls > 0 else 0.0
        
        timing_summary = {
            "total_llm_time_seconds": total_llm_time,
            "total_bash_time_seconds": total_bash_time,
            "average_llm_time_seconds": avg_llm_time,
            "average_bash_time_seconds": avg_bash_time,
            "total_llm_calls": successful_llm_calls,
            "total_bash_calls": successful_bash_calls,
            "llm_time_percentage": (total_llm_time / total_time * 100) if total_time > 0 else 0.0,
            "bash_time_percentage": (total_bash_time / total_time * 100) if total_time > 0 else 0.0,
            "other_time_seconds": max(0, total_time - total_llm_time - total_bash_time)
        }
        
        benchmark_summary = {
            "problem_id": problem['problem_id'],
            "platform": problem['platform'], 
            "title": problem['title'],
            "difficulty": problem['difficulty'],
            "scenario": problem['scenario'],
            "exit_status": exit_status,
            "result": result,
            "total_runtime": total_time,
            "task_preview": problem['description'][:200] + "..." if len(problem['description']) > 200 else problem['description'],
            "model_calls": model.n_calls,
            "model_cost": model.cost,
            "timing_summary": timing_summary,
            "test_cases": problem.get('test_cases', []),
            "detailed_logs": {
                "model_api_calls": model_call_log,
                "bash_executions": bash_execution_log,
                "agent_messages": agent_messages,
                "total_model_calls": len(model_call_log),
                "total_bash_commands": len(bash_execution_log),
                "conversation_length": len(agent_messages)
            }
        }
        
        print(f"Completed in {total_time:.1f}s | Status: {exit_status}")
        
        return benchmark_summary
    
    def run_comprehensive_cpu_benchmark(self) -> List[Dict[str, Any]]:
        """Run comprehensive CPU-intensive benchmark across all workload types."""
        results = []
        
        # Prime numbers benchmark
        try:
            prime_result = self.run_prime_number_benchmark()
            results.append(prime_result)
        except Exception as e:
            print(f"Prime numbers benchmark failed: {e}")
            results.append({"dataset": "prime_numbers", "error": str(e)})
        
        # Sorting algorithms benchmark
        try:
            sorting_result = self.run_sorting_benchmark()
            results.append(sorting_result)
        except Exception as e:
            print(f"Sorting algorithms benchmark failed: {e}")
            results.append({"dataset": "sorting_algorithms", "error": str(e)})
        
        # Fibonacci computation benchmark
        try:
            fibonacci_result = self.run_fibonacci_benchmark()
            results.append(fibonacci_result)
        except Exception as e:
            print(f"Fibonacci benchmark failed: {e}")
            results.append({"dataset": "fibonacci", "error": str(e)})
        
        # Matrix multiplication benchmark (sudoku)
        try:
            matmul_result = self.run_sudoku_benchmark()
            results.append(matmul_result)
        except Exception as e:
            print(f"Matrix multiplication benchmark failed: {e}")
            results.append({"dataset": "matrix_multiplication", "error": str(e)})
        
        # Numerical integration benchmark
        try:
            integration_result = self.run_numerical_integration_benchmark()
            results.append(integration_result)
        except Exception as e:
            print(f"Numerical integration benchmark failed: {e}")
            results.append({"dataset": "numerical_integration", "error": str(e)})
        
        # Save comprehensive results
        comprehensive_results = {
            "timestamp": time.time(),
            "model_config": self.model_config,
            "results": results
        }
        
        output_file = self.output_dir / "comprehensive_cpu_benchmark.json"
        with open(output_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
            
        print(f"Comprehensive CPU benchmark results saved to: {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark mini-swe-agent with vLLM local server")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-Coder-32B-Instruct",
                      help="Path to local model")
    parser.add_argument("--base-url", default="http://localhost:5000", help="vLLM server base URL")
    parser.add_argument("--api-key", default="token-abc123", help="API key for vLLM server")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory for results")
    parser.add_argument("--benchmark-type", choices=["prime_numbers", "sorting", "fibonacci", "matmul", "integration", "lu_decomposition", "knn", "comprehensive_cpu", "swebench", "scicode", "livecodebench", "fft_convolution"],
                      default="comprehensive_cpu", help="Type of CPU-intensive benchmark to run")
    parser.add_argument("--github-issue", help="GitHub issue URL for github_issue benchmark")
    parser.add_argument("--swebench-instance", default="swe-agent__test-repo-1", 
                      help="Single SWEBench instance ID for single mode")
    parser.add_argument("--swebench-instances", nargs="*", default=None,
                      help="SWE-bench instance IDs to test (space separated, for batch mode)")
    parser.add_argument("--max-instances", type=int, default=10,
                      help="Maximum number of instances to test in full benchmark mode")
    parser.add_argument("--scicode-problems", nargs="*", default=None,
                      help="SciCode problem IDs to test (space separated). If not provided, uses first few problems")
    parser.add_argument("--scicode-data-path", default="./scicode_data", 
                      help="Path to SciCode dataset directory")
    parser.add_argument("--livecodebench-problems", nargs="*", default=None,
                      help="LiveCodeBench problem IDs to test (space separated). If not provided, uses first few problems")
    parser.add_argument("--livecodebench-scenario", choices=["code_generation", "self_repair", "test_output_prediction"], 
                      default="code_generation", help="LiveCodeBench scenario to test")
    parser.add_argument("--no-print", action="store_true")
    parser.add_argument("--job_id", type=int, 
                      default=1)
    start_time = timeit.default_timer()
 


    args = parser.parse_args()
    if not args.no_print:
        print(f"{args.job_id}: [TIMING] start: {start_time:.4f}s")
    
    model_config = {
        "model_name": args.model_path,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "model_kwargs": {}
    }
    
    benchmarker = LatencyBenchmarker(model_config, args.output_dir)
    
    if not args.no_print:
        print(f"Starting {args.benchmark_type} benchmark...")
        print(f"Model: {args.model_path}")
        print(f"vLLM Server: {args.base_url}")
        print(f"Output Directory: {args.output_dir}")
        print("-" * 50)
    
    if args.benchmark_type == "prime_numbers":
        result = benchmarker.run_prime_number_benchmark()
        print(f"Prime numbers benchmark completed. Total time: {result.get('total_wall_time', 'N/A')}s")
        # Save result to JSON
        result_file = benchmarker.output_dir / "prime_numbers_benchmark.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {result_file}")

    elif args.benchmark_type == "sorting":
        result = benchmarker.run_sorting_benchmark()
        if not args.no_print:
            print(f"Sorting algorithms benchmark completed. Total time: {result.get('total_wall_time', 'N/A')}s")
        elapsed = timeit.default_timer() - start_time
        # Save result to JSON
        result_file = benchmarker.output_dir / "sorting_benchmark.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"{args.job_id}: [TIMING] end: {elapsed:.4f}s")


    elif args.benchmark_type == "fibonacci":
        result = benchmarker.run_fibonacci_benchmark()
        print(f"Fibonacci benchmark completed. Total time: {result.get('total_wall_time', 'N/A')}s")
        # Save result to JSON
        result_file = benchmarker.output_dir / "fibonacci_benchmark.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {result_file}")

    elif args.benchmark_type == "matmul":
        result = benchmarker.run_sudoku_benchmark()
        print(f"Matrix multiplication benchmark completed. Total time: {result.get('total_wall_time', 'N/A')}s")
        # Save result to JSON
        result_file = benchmarker.output_dir / "matmul_benchmark.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {result_file}")

    elif args.benchmark_type == "integration":
        result = benchmarker.run_numerical_integration_benchmark()
        print(f"Numerical integration benchmark completed. Total time: {result.get('total_wall_time', 'N/A')}s")
        # Save result to JSON
        result_file = benchmarker.output_dir / "integration_benchmark.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {result_file}")

    elif args.benchmark_type == "lu_decomposition":
        result = benchmarker.run_lu_decomposition_benchmark()
        print(f"LU decomposition benchmark completed. Total time: {result.get('total_wall_time', 'N/A')}s")
        # Save result to JSON
        result_file = benchmarker.output_dir / "lu_decomposition_benchmark.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {result_file}")

    elif args.benchmark_type == "fft_convolution":
        result = benchmarker.run_fft_convolution_benchmark()
        print(f"FFT convolution benchmark completed. Total time: {result.get('total_wall_time', 'N/A')}s")
        # Save result to JSON
        result_file = benchmarker.output_dir / "fft_convolution_benchmark.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {result_file}")

    elif args.benchmark_type == "knn":
        result = benchmarker.run_knn_benchmark()
        print(f"k-NN benchmark completed. Total time: {result.get('total_wall_time', 'N/A')}s")
        # Save result to JSON
        result_file = benchmarker.output_dir / "knn_benchmark.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {result_file}")

    elif args.benchmark_type == "swebench":
        if args.swebench_instances:
            result = benchmarker.run_swebench_benchmark(args.swebench_instances, args.max_instances)
        else:
            # Single instance mode for backward compatibility
            result = benchmarker.run_swebench_benchmark([args.swebench_instance], 1)
        
        if "total_benchmark_time" in result:
            print(f"SWEBench benchmark completed. Total time: {result.get('total_benchmark_time', 'N/A'):.1f}s")
            print(f"Success rate: {result.get('success_rate_percent', 0):.1f}% ({result.get('successful_completions', 0)}/{result.get('total_instances', 0)})")
        else:
            print(f"SWEBench single instance completed. Total time: {result.get('total_runtime', 'N/A')}s")
        
    elif args.benchmark_type == "scicode":
        result = benchmarker.run_scicode_benchmark(
            problem_ids=args.scicode_problems, 
            max_problems=args.max_instances, 
            scicode_data_path=args.scicode_data_path
        )
        
        if "total_benchmark_time" in result:
            print(f"SciCode benchmark completed. Total time: {result.get('total_benchmark_time', 'N/A'):.1f}s")
            print(f"Success rate: {result.get('success_rate_percent', 0):.1f}% ({result.get('successful_completions', 0)}/{result.get('total_problems', 0)})")
        else:
            print(f"SciCode single problem completed. Total time: {result.get('total_runtime', 'N/A')}s")
        
    elif args.benchmark_type == "livecodebench":
        result = benchmarker.run_livecodebench_benchmark(
            problem_ids=args.livecodebench_problems, 
            max_problems=args.max_instances, 
            scenario=args.livecodebench_scenario
        )
        
        if "total_benchmark_time" in result:
            print(f"LiveCodeBench benchmark completed. Total time: {result.get('total_benchmark_time', 'N/A'):.1f}s")
            print(f"Scenario: {result.get('scenario', 'N/A')}")
            print(f"Success rate: {result.get('success_rate_percent', 0):.1f}% ({result.get('successful_completions', 0)}/{result.get('total_problems', 0)})")
        else:
            print(f"LiveCodeBench single problem completed. Total time: {result.get('total_runtime', 'N/A')}s")
        
    elif args.benchmark_type == "comprehensive_cpu":
        results = benchmarker.run_comprehensive_cpu_benchmark()
        print(f"Comprehensive CPU benchmark completed. {len(results)} tests run.")
        
        # Print summary
        for result in results:
            if "error" not in result:
                print(f"  {result['dataset']}: {result.get('total_wall_time', 'N/A')}s "
                     f"({result.get('model_calls', 'N/A')} calls)")
            else:
                print(f"  {result['dataset']}: ERROR - {result['error']}")


if __name__ == "__main__":
    main()