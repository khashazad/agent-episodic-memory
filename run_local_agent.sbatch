#!/bin/bash
#SBATCH --job-name=local_agent
#SBATCH --output=local_agent_%j.out
#SBATCH --error=local_agent_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpunodes
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx_a2000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=khashayar1924@gmail.com

# =============================================================================
# Load environment variables from .env file (if it exists)
# =============================================================================
# This will automatically export all variables from .env
# Variables in .env will override any defaults set below
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    set -a  # Automatically export all variables
    source .env
    set +a  # Turn off automatic export
    echo "Environment variables loaded successfully."
else
    echo "No .env file found. Using defaults or variables set in environment."
fi
echo ""

# =============================================================================
# Local Agent Configuration
# =============================================================================

# Model configuration
LOCAL_MODEL_NAME=${LOCAL_MODEL_NAME:-"microsoft/Phi-3-mini-4k-instruct"}
LOCAL_MODEL_DEVICE=${LOCAL_MODEL_DEVICE:-"auto"}
LOCAL_MODEL_MAX_NEW_TOKENS=${LOCAL_MODEL_MAX_NEW_TOKENS:-256}
LOCAL_MODEL_TEMPERATURE=${LOCAL_MODEL_TEMPERATURE:-0.2}
LOCAL_MODEL_DTYPE=${LOCAL_MODEL_DTYPE:-"float16"}

# Agent configuration
USE_RAG=${USE_RAG:-"false"}
RENDER=${RENDER:-"false"}
TEST_RUNS=${TEST_RUNS:-5}
MAX_FRAMES=${MAX_FRAMES:-500}
USE_MAX_FRAMES=${USE_MAX_FRAMES:-"true"}
EMBEDDING_METHOD=${EMBEDDING_METHOD:-"local_llm"}

# Server configuration
USE_REMOTE_SERVER=${USE_REMOTE_SERVER:-"true"}
MINERL_SERVER_URL=${MINERL_SERVER_URL:-"http://127.0.0.1:5001"}

# Export all variables for the Python script
export LOCAL_MODEL_NAME
export LOCAL_MODEL_DEVICE
export LOCAL_MODEL_MAX_NEW_TOKENS
export LOCAL_MODEL_TEMPERATURE
export LOCAL_MODEL_DTYPE
export USE_RAG
export RENDER
export TEST_RUNS
export MAX_FRAMES
export USE_MAX_FRAMES
export EMBEDDING_METHOD
export USE_REMOTE_SERVER
export MINERL_SERVER_URL

# Print job information
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Working directory: $PWD"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# Print configuration
echo ""
echo "Configuration:"
echo "  Model: $LOCAL_MODEL_NAME"
echo "  Device: $LOCAL_MODEL_DEVICE"
echo "  Max New Tokens: $LOCAL_MODEL_MAX_NEW_TOKENS"
echo "  Temperature: $LOCAL_MODEL_TEMPERATURE"
echo "  Dtype: $LOCAL_MODEL_DTYPE"
echo ""
echo "  Use RAG: $USE_RAG"
echo "  Render: $RENDER"
echo "  Test Runs: $TEST_RUNS"
echo "  Max Frames: $MAX_FRAMES"
echo "  Use Max Frames: $USE_MAX_FRAMES"
echo "  Embedding Method: $EMBEDDING_METHOD"
echo ""
echo "  Use Remote Server: $USE_REMOTE_SERVER"
echo "  Server URL: $MINERL_SERVER_URL"
echo "=============================================="
echo ""

# Ensure we're in the project directory
cd "$SLURM_SUBMIT_DIR"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not available."
    exit 1
fi

# Check for HuggingFace token if using a gated model
if [[ "$LOCAL_MODEL_NAME" == *"llama"* ]] || [[ "$LOCAL_MODEL_NAME" == *"Llama"* ]]; then
    if [ -z "$HF_TOKEN" ]; then
        echo "Warning: HF_TOKEN not set. Llama models require authentication."
        echo "Set HF_TOKEN environment variable or login with 'huggingface-cli login'"
    fi
fi

# Create results directory if it doesn't exist
mkdir -p Agent/Results

echo "Running command: uv run python -u Agent/agent_local.py"
echo ""

# Run the local agent script using uv with unbuffered output (-u flag)
# This ensures progress messages appear immediately in sbatch output
uv run python -u Agent/agent_local.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Job completed successfully at: $(date)"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "Job failed at: $(date)"
    echo "=============================================="
    exit 1
fi
