#!/bin/bash

# Activate the virtual environment if it exists in the parent directory
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment from .venv"
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
    echo "Activated virtual environment from ../.venv"
fi

# Define model
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

echo "Starting vLLM server with model $MODEL on port 8000..."

# Verify if vLLM is installed
if ! python -c "import vllm" &> /dev/null; then
    echo "Error: vllm is not installed. Please run 'pip install vllm' (and other reqs)."
    exit 1
fi

# Attempt to start the server
# Use --dtype float32 for CPU compatibility sometimes if supported
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --enforce-eager


# If the above fails, check logs.
