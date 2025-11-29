from typing import Optional
import requests

import langchain
from dotenv import load_dotenv
from langchain import PromptTemplate, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.base import BaseLLM
from pydantic import Field

from pydantic import ValidationError
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor


class VLLMOpenAI(BaseLLM):
    """Custom VLLM OpenAI-compatible client to replace langchain_community dependency"""
    
    base_url: str = Field(default="http://localhost:8000/v1")
    model: str = Field(default="")
    temperature: float = Field(default=0.0)
    openai_api_key: str = Field(default="EMPTY")
    
    @property
    def _llm_type(self) -> str:
        return "vllm_openai"
    
    def _call(self, prompt: str, stop=None) -> str:
        """Call the VLLM server with OpenAI-compatible API"""
        import time
        
        print(f"\n🔥 VLLM CALL: Making request to {self.base_url}/completions")
        print(f"🔥 VLLM PROMPT LENGTH: {len(prompt)} characters")
        print(f"🔥 VLLM PROMPT START: {prompt[:200]}...")
        print(f"🔥 VLLM STOP: {stop}")
        
        start_time = time.time()
        
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": self.temperature,
                "stop": stop or []
            }
            
            print(f"🔥 VLLM REQUEST: {data}")
            
            response = requests.post(
                f"{self.base_url}/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            call_time = time.time() - start_time
            print(f"🔥 VLLM RESPONSE TIME: {call_time:.4f}s")
            print(f"🔥 VLLM STATUS: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["text"]
                print(f"🔥 VLLM RESPONSE: {response_text[:100]}...")
                return response_text
            else:
                error_msg = f"Error: VLLM server returned status {response.status_code}: {response.text}"
                print(f"🔥 VLLM ERROR: {error_msg}")
                return error_msg
                
        except Exception as e:
            call_time = time.time() - start_time
            error_msg = f"Error calling VLLM server: {str(e)}"
            print(f"🔥 VLLM EXCEPTION after {call_time:.4f}s: {error_msg}")
            return error_msg
    
    def _generate(self, prompts, stop=None, run_manager=None):
        """Generate responses for multiple prompts"""
        from langchain.schema import LLMResult, Generation
        
        generations = []
        for prompt in prompts:
            response = self._call(prompt, stop=stop)
            generations.append([Generation(text=response)])
        
        return LLMResult(generations=generations)

from .prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, REPHRASE_TEMPLATE, SUFFIX
from .tools import make_tools


def _make_llm(model, temp, api_key, streaming: bool = False):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        llm = langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,
            max_tokens=200,  # Strict limit for 50-word responses to minimize LLM time
            cache=None,  # Disable LangChain caching for accurate benchmarking
        )
    elif model.startswith("text-"):
        llm = langchain.OpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,
            max_tokens=200,  # Strict limit for 50-word responses to minimize LLM time
            cache=None,  # Disable LangChain caching for accurate benchmarking
        )
    elif model == "vllm" or model.startswith("vllm"):
        llm = VLLMOpenAI(
            base_url="http://localhost:8000/v1",
            model="/usr/scratch/ritik/hugging_face/hub/models--meta-llama--Llama-2-7B-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
            temperature=temp,
            openai_api_key="EMPTY",
            cache=None  # Disable LangChain caching for accurate benchmarking
        )
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm


class ChemCrow:
    def __init__(
        self,
        tools=None,
        model="gpt-4-0613",
        tools_model="gpt-3.5-turbo-0613",
        temp=0.1,
        max_iterations=40,
        verbose=True,
        streaming: bool = True,
        openai_api_key: Optional[str] = None,
        api_keys: dict = {},
        local_rxn: bool = False,
    ):
        """Initialize ChemCrow agent."""

        load_dotenv()
        try:
            self.llm = _make_llm(model, temp, openai_api_key, streaming)
            print(self.llm)
        except ValidationError:
            raise ValueError("Invalid OpenAI API key")

        if tools is None:
            api_keys["OPENAI_API_KEY"] = openai_api_key
            tools_llm = _make_llm(tools_model, temp, openai_api_key, streaming)
            tools = make_tools(tools_llm, api_keys=api_keys, local_rxn=local_rxn, verbose=verbose)

        # Initialize agent
        self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=ChatZeroShotAgent.from_llm_and_tools(
                self.llm,
                tools,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
                question_prompt=QUESTION_PROMPT,
            ),
            verbose=True,
            max_iterations=max_iterations,
        )

        rephrase = PromptTemplate(
            input_variables=["question", "agent_ans"], template=REPHRASE_TEMPLATE
        )

        self.rephrase_chain = chains.LLMChain(prompt=rephrase, llm=self.llm)

    def run(self, prompt):
        outputs = self.agent_executor({"input": prompt})
        return outputs["output"]
    
    def run_with_timing(self, prompt):
        """Run ChemCrow with detailed timing metrics for benchmarking"""
        import time
        
        # Initialize timing tracking
        timing_metrics = {
            'total_start': time.time(),
            'llm_time': 0.0,
            'tool_time': 0.0,
            'total_time': 0.0
        }
        
        # Create a custom callback to track LLM calls
        from langchain.callbacks.base import BaseCallbackHandler
        
        class TimingCallback(BaseCallbackHandler):
            def __init__(self, timing_metrics):
                super().__init__()
                self.timing_metrics = timing_metrics
                self.llm_start = None
                self.tool_start = None
                self.current_tool_name = None
                self.tool_active = False
                self.last_tool_completed = None  # Track the last tool that completed
                self.tool_result_pending = False  # Flag for LLM calls right after tool completion
                
            def _is_in_tool_context(self):
                """Check if we're currently executing within a tool by examining the call stack."""
                import inspect
                import traceback
                
                # Get the current call stack
                stack = traceback.extract_stack()
                
                # Look for tool-related patterns in the call stack
                tool_indicators = [
                    'tools/safety.py',
                    'tools/search.py', 
                    'tools/converters.py',
                    'tools/rdkit.py',
                    'SafetySummary',
                    'Scholar2ResultLLM',
                    'MoleculeSafety',
                    'MolSimilarity',
                    'Query2CAS',
                    'Query2SMILES',
                    'ControlChemCheck',
                    'ExplosiveCheck',
                    'SMILES2Weight',
                    'FuncGroups',
                    'PatentCheck',
                    'GetMoleculePrice',
                    '_run'  # Tool execution method
                ]
                
                # Check if any frame in the stack indicates we're in a tool
                for frame in stack:
                    frame_info = f"{frame.filename}:{frame.name}"
                    if any(indicator in frame_info for indicator in tool_indicators):
                        # Extract tool name if possible
                        if 'SafetySummary' in frame_info or 'safety.py' in frame_info:
                            return True, 'SafetySummary'
                        elif 'Scholar2ResultLLM' in frame_info or 'LiteratureSearch' in frame_info:
                            return True, 'LiteratureSearch'
                        elif 'MolSimilarity' in frame_info:
                            return True, 'MolSimilarity'
                        elif 'Query2CAS' in frame_info:
                            return True, 'Query2CAS'
                        elif 'Query2SMILES' in frame_info:
                            return True, 'Query2SMILES'
                        elif 'ControlChemCheck' in frame_info:
                            return True, 'ControlChemCheck'
                        elif 'ExplosiveCheck' in frame_info:
                            return True, 'ExplosiveCheck'
                        elif 'SMILES2Weight' in frame_info:
                            return True, 'SMILES2Weight'
                        elif 'FuncGroups' in frame_info:
                            return True, 'FuncGroups'
                        elif 'PatentCheck' in frame_info:
                            return True, 'PatentCheck'
                        elif 'GetMoleculePrice' in frame_info:
                            return True, 'GetMoleculePrice'
                        elif any(tool in frame_info for tool in ['converters.py', 'rdkit.py']):
                            return True, 'ChemTool'
                        else:
                            return True, 'Tool'
                
                return False, None
                
            def on_llm_start(self, serialized, prompts, **kwargs):
                # Check if we're in a tool context by examining the call stack
                in_tool, tool_name = self._is_in_tool_context()
                
                # Also check if we just completed a tool and this might be result interpretation
                if in_tool:
                    print(f"⏱️  TIMING: LLM call started (within {tool_name} tool)")
                    self.tool_active = True
                    self.current_tool_name = tool_name
                elif self.tool_result_pending and self.last_tool_completed:
                    print(f"⏱️  TIMING: LLM call started (interpreting {self.last_tool_completed} result)")
                    self.tool_active = True  # Treat as tool-related
                    self.current_tool_name = self.last_tool_completed
                else:
                    print("⏱️  TIMING: LLM call started (direct agent call)")
                    self.tool_active = False
                    
                self.llm_start = time.time()
                
            def on_llm_end(self, response, **kwargs):
                if self.llm_start:
                    llm_duration = time.time() - self.llm_start
                    
                    # Determine if this LLM call should be counted as tool time
                    # This includes: calls within tools, or calls interpreting tool results
                    if self.tool_active and self.current_tool_name:
                        # LLM call within a tool context - count as tool time
                        self.timing_metrics['tool_time'] += llm_duration
                        print(f"⏱️  TIMING: LLM call ended (within/for {self.current_tool_name}), duration: {llm_duration:.4f}s - COUNTED AS TOOL TIME")
                        
                        # Track individual tool times
                        if 'individual_tool_times' not in self.timing_metrics:
                            self.timing_metrics['individual_tool_times'] = {}
                        if self.current_tool_name not in self.timing_metrics['individual_tool_times']:
                            self.timing_metrics['individual_tool_times'][self.current_tool_name] = 0
                        self.timing_metrics['individual_tool_times'][self.current_tool_name] += llm_duration
                        
                        # Add to tools used
                        if 'tools_used' not in self.timing_metrics:
                            self.timing_metrics['tools_used'] = []
                        if self.current_tool_name not in self.timing_metrics['tools_used']:
                            self.timing_metrics['tools_used'].append(self.current_tool_name)
                    else:
                        # Direct LLM call by agent - count as LLM time
                        self.timing_metrics['llm_time'] += llm_duration
                        print(f"⏱️  TIMING: LLM call ended (direct agent), duration: {llm_duration:.4f}s - COUNTED AS LLM TIME")
                    
                    # Clear the tool result pending flag after processing
                    if self.tool_result_pending:
                        self.tool_result_pending = False
                        self.last_tool_completed = None
                    
                    # Removed 5-second delay to get accurate timing measurements
                    
                    self.llm_start = None
                else:
                    print("⏱️  TIMING: LLM end called but no start time recorded")
                    
            def on_tool_start(self, serialized, input_str, **kwargs):
                tool_name = serialized.get('name', 'Unknown')
                self.current_tool_name = tool_name
                self.tool_active = True  # Mark that we're now inside a tool
                print(f"🔧 TOOL START: {tool_name}")
                print(f"⏱️  DEBUG: Tool serialized: {serialized}")
                print(f"⏱️  DEBUG: Tool input: {input_str[:100] if input_str else 'None'}...")
                self.tool_start = time.time()
                
                # Track tools used
                if 'tools_used' not in self.timing_metrics:
                    self.timing_metrics['tools_used'] = []
                if tool_name not in self.timing_metrics['tools_used']:
                    self.timing_metrics['tools_used'].append(tool_name)
                
            def on_tool_end(self, output, **kwargs):
                print(f"🔧 TOOL END: {self.current_tool_name}")
                print(f"⏱️  DEBUG: Tool end - start_time: {self.tool_start}, output: {str(output)[:50] if output else 'None'}...")
                if self.tool_start and self.current_tool_name:
                    tool_duration = time.time() - self.tool_start
                    self.timing_metrics['tool_time'] += tool_duration
                    print(f"⏱️  TIMING: {self.current_tool_name} ended, duration: {tool_duration:.4f}s - COUNTED AS TOOL TIME")
                    
                    # Track individual tool timings
                    if 'individual_tool_times' not in self.timing_metrics:
                        self.timing_metrics['individual_tool_times'] = {}
                    
                    if self.current_tool_name not in self.timing_metrics['individual_tool_times']:
                        self.timing_metrics['individual_tool_times'][self.current_tool_name] = 0
                    
                    self.timing_metrics['individual_tool_times'][self.current_tool_name] += tool_duration
                    
                    # Mark that a tool just completed - next LLM call is likely interpreting its result
                    self.last_tool_completed = self.current_tool_name
                    self.tool_result_pending = True
                    
                    self.tool_start = None
                    self.current_tool_name = None
                    self.tool_active = False  # Mark that we're no longer inside a tool
                else:
                    print(f"⏱️  TIMING: Tool end called but no start time recorded - tool_start: {self.tool_start}, current_tool: {self.current_tool_name}")
                    self.tool_active = False  # Reset even on error
                    
            def on_tool_error(self, error, **kwargs):
                """Handle tool errors and still record timing"""
                if self.tool_start and self.current_tool_name:
                    tool_duration = time.time() - self.tool_start
                    self.timing_metrics['tool_time'] += tool_duration
                    print(f"⏱️  TIMING: Tool call errored: {self.current_tool_name}, duration: {tool_duration:.4f}s, error: {error}")
                    
                    # Track individual tool timings even on error
                    if 'individual_tool_times' not in self.timing_metrics:
                        self.timing_metrics['individual_tool_times'] = {}
                    
                    if self.current_tool_name not in self.timing_metrics['individual_tool_times']:
                        self.timing_metrics['individual_tool_times'][self.current_tool_name] = 0
                    
                    self.timing_metrics['individual_tool_times'][self.current_tool_name] += tool_duration
                    
                    self.tool_start = None
                    self.current_tool_name = None
                    self.tool_active = False
                else:
                    print(f"⏱️  TIMING: Tool error called but no start time recorded - error: {error}")
                    self.tool_active = False
                    
            def on_chain_start(self, serialized, inputs, **kwargs):
                # Required method for BaseCallbackHandler
                pass
                
            def on_chain_end(self, outputs, **kwargs):
                # Required method for BaseCallbackHandler
                pass
        
        # Create timing callback
        timing_callback = TimingCallback(timing_metrics)
        
        # Store original callbacks 
        original_callbacks = self.agent_executor.callbacks or []
        original_llm_callbacks = self.llm.callbacks or []
        
        # Create external tool timing tracker (avoid Pydantic field restrictions)
        tool_timing_tracker = {}
        
        # Add timing callback to agent executor, main LLM, and all tools
        self.agent_executor.callbacks = original_callbacks + [timing_callback]
        self.llm.callbacks = original_llm_callbacks + [timing_callback]
        
        # IMPORTANT: Add callbacks to tools that use LLMs for proper timing tracking
        for i, tool in enumerate(self.agent_executor.tools):
            try:
                print(f"⏱️  DEBUG: Processing tool {i}: {tool.name} (type: {type(tool)})")
                
                # Due to Pydantic v2 restrictions, we cannot modify tool._run methods
                # Instead, we rely on improved callback detection and post-execution analysis
                # Store tool names for later reference
                tool_timing_tracker[tool.name] = {
                    'type': type(tool).__name__,
                    'wrapped': False
                }
                
                # Add callbacks to LLMs used by tools
                if hasattr(tool, 'llm') and tool.llm is not None:
                    try:
                        original_tool_callbacks = getattr(tool.llm, 'callbacks', []) or []
                        tool.llm.callbacks = original_tool_callbacks + [timing_callback]
                        print(f"⏱️  Added timing callback to tool: {tool.name}")
                        
                        # Special handling for SafetySummary tool
                        if tool.name == "SafetySummary":
                            # Add callback to the main llm_chain
                            if hasattr(tool, 'llm_chain') and hasattr(tool.llm_chain, 'llm'):
                                tool.llm_chain.llm.callbacks = (getattr(tool.llm_chain.llm, 'callbacks', []) or []) + [timing_callback]
                                print(f"⏱️  Added timing callback to SafetySummary.llm_chain")
                            
                            # Add callback to MoleculeSafety's LLM if it exists
                            if hasattr(tool, 'mol_safety') and hasattr(tool.mol_safety, 'llm') and tool.mol_safety.llm is not None:
                                tool.mol_safety.llm.callbacks = (getattr(tool.mol_safety.llm, 'callbacks', []) or []) + [timing_callback]
                                print(f"⏱️  Added timing callback to SafetySummary.mol_safety.llm")
                        
                        # Special handling for Scholar2ResultLLM tool
                        elif tool.name == "LiteratureSearch":  # Note: actual name is LiteratureSearch
                            # The Scholar2ResultLLM tool has its own llm attribute used in paper_search
                            # Add callback to that LLM as well to capture internal LLM calls
                            if hasattr(tool, 'llm') and tool.llm is not None:
                                original_scholar_callbacks = getattr(tool.llm, 'callbacks', []) or []
                                tool.llm.callbacks = original_scholar_callbacks + [timing_callback]
                                print(f"⏱️  Added timing callback to LiteratureSearch.llm")
                            print(f"⏱️  Configured timing for LiteratureSearch tool")
                    except Exception as e:
                        print(f"⏱️  ERROR: Failed to add LLM callbacks to tool {tool.name}: {e}")
                
            except Exception as e:
                print(f"⏱️  ERROR: Exception processing tool {i} ({getattr(tool, 'name', 'unknown')}): {e}")
                import traceback
                traceback.print_exc()
        
        print(f"⏱️  DEBUG: Added timing callback to agent, LLM, and processed {len(self.agent_executor.tools)} tools")
        print(f"⏱️  CALLBACK: Agent executor callbacks: {len(self.agent_executor.callbacks)}")
        print(f"⏱️  CALLBACK: Main LLM callbacks: {len(self.llm.callbacks)}")
        
        # Ensure agent executor has the proper callback manager
        if hasattr(self.agent_executor, 'agent') and hasattr(self.agent_executor.agent, 'llm_chain'):
            chain_callbacks = getattr(self.agent_executor.agent.llm_chain, 'callbacks', []) or []
            self.agent_executor.agent.llm_chain.callbacks = chain_callbacks + [timing_callback]
            print(f"⏱️  CALLBACK: Added callback to agent LLM chain")
        
        try:
            # Run the agent
            outputs = self.agent_executor({"input": prompt})
            result = outputs["output"]
            
            # Calculate total time
            timing_metrics['total_time'] = time.time() - timing_metrics['total_start']
            
            # Restore original callbacks for all components
            self.agent_executor.callbacks = original_callbacks
            self.llm.callbacks = original_llm_callbacks
            
            # Restore tool callbacks  
            for tool in self.agent_executor.tools:
                if hasattr(tool, 'llm') and tool.llm is not None:
                    # Remove our callback from tool
                    if hasattr(tool.llm, 'callbacks') and tool.llm.callbacks:
                        tool.llm.callbacks = [cb for cb in tool.llm.callbacks if cb != timing_callback]
                    
                    # Special cleanup for SafetySummary tool
                    if tool.name == "SafetySummary":
                        # Remove callback from llm_chain
                        if hasattr(tool, 'llm_chain') and hasattr(tool.llm_chain, 'llm') and hasattr(tool.llm_chain.llm, 'callbacks'):
                            tool.llm_chain.llm.callbacks = [cb for cb in tool.llm_chain.llm.callbacks if cb != timing_callback]
                        
                        # Remove callback from MoleculeSafety's LLM
                        if hasattr(tool, 'mol_safety') and hasattr(tool.mol_safety, 'llm') and hasattr(tool.mol_safety.llm, 'callbacks'):
                            tool.mol_safety.llm.callbacks = [cb for cb in tool.mol_safety.llm.callbacks if cb != timing_callback]
            
            return result, timing_metrics
            
        except Exception as e:
            # Restore original callbacks even on error  
            self.agent_executor.callbacks = original_callbacks
            self.llm.callbacks = original_llm_callbacks
            
            # Restore tool callbacks on error too
            for tool in self.agent_executor.tools:
                if hasattr(tool, 'llm') and tool.llm is not None:
                    if hasattr(tool.llm, 'callbacks') and tool.llm.callbacks:
                        tool.llm.callbacks = [cb for cb in tool.llm.callbacks if cb != timing_callback]
                    
                    # Special cleanup for SafetySummary tool on error
                    if tool.name == "SafetySummary":
                        # Remove callback from llm_chain
                        if hasattr(tool, 'llm_chain') and hasattr(tool.llm_chain, 'llm') and hasattr(tool.llm_chain.llm, 'callbacks'):
                            tool.llm_chain.llm.callbacks = [cb for cb in tool.llm_chain.llm.callbacks if cb != timing_callback]
                        
                        # Remove callback from MoleculeSafety's LLM
                        if hasattr(tool, 'mol_safety') and hasattr(tool.mol_safety, 'llm') and hasattr(tool.mol_safety.llm, 'callbacks'):
                            tool.mol_safety.llm.callbacks = [cb for cb in tool.mol_safety.llm.callbacks if cb != timing_callback]
            
            timing_metrics['total_time'] = time.time() - timing_metrics['total_start']
            return f"Error: {str(e)}", timing_metrics
