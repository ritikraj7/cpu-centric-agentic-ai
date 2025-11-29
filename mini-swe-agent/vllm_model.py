#!/usr/bin/env python3
"""
Direct vLLM model class that bypasses LiteLLM complexities.
"""

import requests
import json
import time
from dataclasses import dataclass, asdict
from typing import Any

from minisweagent.models import GLOBAL_MODEL_STATS


@dataclass 
class VLLMModelConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "token-abc123"
    model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    max_tokens: int = 4096
    temperature: float = 0.0
    cost_per_call: float = 0.01
    timeout: int = 120  # Increased timeout for complex prompts
    max_retries: int = 3


class VLLMModel:
    """Direct vLLM model that connects to OpenAI-compatible vLLM server."""
    
    def __init__(self, **kwargs):
        self.config = VLLMModelConfig(**kwargs)
        self.n_calls = 0
        self.cost = 0.0
        self.call_log = []  # Store all API calls for logging
    
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Query the vLLM server directly using OpenAI-compatible API with retry logic."""
        
        # Start timing the API call
        start_time = time.time()
        
        # Prepare the request
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        # Get timeout from kwargs or config
        timeout = kwargs.get("timeout", self.config.timeout)
        
        # Retry logic for handling timeouts and server errors
        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                # The base_url already includes /v1, so just add the endpoint
                if "/v1" in self.config.base_url:
                    url = f"{self.config.base_url}/chat/completions"
                else:
                    url = f"{self.config.base_url}/v1/chat/completions"
                
                # print(f"🔄 Making vLLM request (attempt {attempt + 1}/{self.config.max_retries}, timeout={timeout}s)")
                
                # Make the request to vLLM server
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                
                if response.status_code != 200:
                    raise Exception(f"vLLM server returned {response.status_code}: {response.text}")
                
                result = response.json()
                
                # Extract the content
                content = result["choices"][0]["message"]["content"]
                
                # Calculate total duration
                end_time = time.time()
                duration_seconds = end_time - start_time
                
                # Update stats
                self.n_calls += 1
                self.cost += self.config.cost_per_call
                GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
                
                # Log this API call with full details including timing
                call_record = {
                    "call_number": self.n_calls,
                    "timestamp": start_time,
                    "duration_total_seconds": duration_seconds,
                    "attempt_number": attempt + 1,
                    "request": {
                        "messages": [msg.copy() for msg in messages],
                        "model": self.config.model_name,
                        "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                        "temperature": kwargs.get("temperature", self.config.temperature)
                    },
                    "response": {
                        "content": content,
                        "usage": result.get("usage", {}),
                        "model": result.get("model", ""),
                    }
                }
                self.call_log.append(call_record)
                
                # print(f"✅ vLLM request successful after {attempt + 1} attempt(s), took {duration_seconds:.2f}s")
                return {"content": content}
                
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exception = e
                wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10 seconds
                print(f"⚠️ Request timeout/connection error on attempt {attempt + 1}: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    print(f"🔄 Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    # Increase timeout for next attempt
                    timeout = min(timeout * 1.5, 300)  # Max 5 minutes
                continue
                
            except Exception as e:
                last_exception = e
                print(f"❌ Unexpected error on attempt {attempt + 1}: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    wait_time = 2
                    print(f"🔄 Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                continue
                break
        
        # All attempts failed, log the failure and raise
        end_time = time.time()
        duration_seconds = end_time - start_time
        
        error_call_record = {
            "call_number": self.n_calls + 1,
            "timestamp": start_time,
            "duration_total_seconds": duration_seconds,
            "attempts_made": self.config.max_retries,
            "request": {
                "messages": [msg.copy() for msg in messages],
                "model": self.config.model_name,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature)
            },
            "error": str(last_exception)
        }
        self.call_log.append(error_call_record)
        
        raise Exception(f"vLLM query failed after {self.config.max_retries} attempts: {last_exception}")
    
    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}