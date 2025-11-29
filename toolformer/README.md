# Math Toolformer
 
A Python implementation of the Toolformer methodology for mathematical problem solving, featuring GPT-J-6B language model augmented with external calculator tools (Wolfram Alpha API and local AST-based calculator).
 
## Overview
 
This project replicates the Toolformer approach for teaching language models to autonomously use external tools. The system demonstrates how a 6B parameter language model can learn to invoke calculator tools for solving mathematical word problems without explicit fine-tuning on tool usage.
 
## Key Features
 
### Model Architecture
- **GPT-J-6B Language Model**: 6-billion parameter autoregressive transformer
- **vLLM Backend**: High-performance inference with OpenAI-compatible API
- **LangChain Integration**: Simplified LLM orchestration and tool management
  
### Evaluation Datasets
- **ASDiv**: Academia Sinica Diverse Math Word Problems
- **SVAMP**: Simple Variations on Arithmetic Math Problems
- **MAWPS**: Math Word Problem Solver (128 problems across 6 categories)
 
## Architecture
 
```
User Question → Prompt Builder → GPT-J-6B → Calculator Detection → Tool Execution → Final Answer
                                 (vLLM)        [Calculator("...")]    (Wolfram/AST)
```
 
### Processing Pipeline
 
1. **Prompt Construction**: Problem embedded in few-shot learning context
2. **Initial Generation**: GPT-J generates solution with calculator placeholders
3. **Tool Detection**: Regex extraction of `[Calculator("expression")]` calls
4. **Tool Execution**: External API or local AST evaluation
5. **Result Integration**: Calculator outputs inserted into solution
6. **Final Generation**: Refined answer based on computed results
 
## Dependencies
 
### Python Requirements
If already setup, activate langchain conda environment-

`conda activate main`

Or, if you want to setup from scratch- 

```bash
pip install langchain-community requests numpy tqdm
```
 
 
### Wolfram Alpha API Setup
 
1. Go to [WolframAlpha API website](https://products.wolframalpha.com/api/) to request an API key.
    - Click on 'Get API Access' button.
    - Click on 'Don’t have an account?Create one' button.
    - Sign In after creating the account.
    - Click on ‘Get an APP ID’ button.
    - Fill any Name and Description.
    - Select ‘SimpleAPI’ from the API list.
    - Click on ‘Submit’ button.
    - Copy the APP ID.
2. Set environment variable:
 
```bash
export WOLFRAM_ALPHA_APPID="your-app-id-here"
```
 
### vLLM Server Setup
 
Start vLLM server with GPT-J-6B:
 
```bash
vllm serve EleutherAI/gpt-j-6B \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 2048
```
 
## Usage
  
```bash
python math_toolformer.py
```