#!/bin/bash
#SBATCH --job-name=multimodal_agent
#SBATCH --output=.slurm_logs/%j/output.out
#SBATCH --error=.slurm_logs/%j/error.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpunodes
#SBATCH -c 4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=khashayar1924@gmail.com

# =============================================================================
# Multimodal Agent with Fused Embeddings (RAG v4)
# =============================================================================
# This script runs the multimodal agent that uses fused embeddings combining:
#   - MineCLIP video embeddings (512-dim)
#   - Qwen2.5-VL-7B generated descriptions
#   - MineCLIP text embeddings (512-dim)
#   - Fused embedding = (video + text) / 2
#
# Memory requirements:
#   - MineCLIP: ~2GB VRAM
#   - Qwen2.5-VL-7B: ~16GB VRAM (float16)
#   - Total: ~20GB VRAM recommended (using A6000 with 48GB)
# =============================================================================

# Create log directory
mkdir -p .slurm_logs/$SLURM_JOB_ID

# =============================================================================
# Load environment variables from .env file (if it exists)
# =============================================================================
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    set -a
    source .env
    set +a
    echo "Environment variables loaded successfully."
else
    echo "No .env file found. Using defaults or variables set in environment."
fi
echo ""

# =============================================================================
# Multimodal Agent Configuration
# =============================================================================

# LLM selection: "true" for OpenAI, "false" for local HuggingFace model
USE_OPENAI_LLM=${USE_OPENAI_LLM:-"false"}

# OpenAI configuration (used when USE_OPENAI_LLM=true)
OPENAI_MODEL_NAME=${OPENAI_MODEL_NAME:-"gpt-4o-mini"}
OPENAI_TEMPERATURE=${OPENAI_TEMPERATURE:-0.2}

# Local model configuration (used when USE_OPENAI_LLM=false)
LOCAL_MODEL_NAME=${LOCAL_MODEL_NAME:-"microsoft/Phi-3-mini-4k-instruct"}
LOCAL_MODEL_DEVICE=${LOCAL_MODEL_DEVICE:-"auto"}
LOCAL_MODEL_MAX_NEW_TOKENS=${LOCAL_MODEL_MAX_NEW_TOKENS:-256}
LOCAL_MODEL_TEMPERATURE=${LOCAL_MODEL_TEMPERATURE:-0.2}
LOCAL_MODEL_DTYPE=${LOCAL_MODEL_DTYPE:-"float16"}

# Agent configuration - RAG v4 (fused multimodal) is enabled by default
USE_RAG=${USE_RAG:-"true"}
RAG_CONFIG=${RAG_CONFIG:-"v4"}
RENDER=${RENDER:-"false"}
TEST_RUNS=${TEST_RUNS:-5}
MAX_FRAMES=${MAX_FRAMES:-500}
USE_MAX_FRAMES=${USE_MAX_FRAMES:-"true"}
EMBEDDING_METHOD="fused_multimodal"

# Server configuration
USE_REMOTE_SERVER=${USE_REMOTE_SERVER:-"true"}
MINERL_SERVER_URL=${MINERL_SERVER_URL:-"http://127.0.0.1:5001"}

# ChromaDB configuration
CHROMA_HOST=${CHROMA_HOST:-"localhost"}
CHROMA_PORT=${CHROMA_PORT:-"8000"}

# Export all variables for the Python script
export USE_OPENAI_LLM
export OPENAI_MODEL_NAME
export OPENAI_TEMPERATURE
export LOCAL_MODEL_NAME
export LOCAL_MODEL_DEVICE
export LOCAL_MODEL_MAX_NEW_TOKENS
export LOCAL_MODEL_TEMPERATURE
export LOCAL_MODEL_DTYPE
export USE_RAG
export RAG_CONFIG
export RENDER
export TEST_RUNS
export MAX_FRAMES
export USE_MAX_FRAMES
export EMBEDDING_METHOD
export USE_REMOTE_SERVER
export MINERL_SERVER_URL
export CHROMA_HOST
export CHROMA_PORT

# Print job information
echo "=============================================="
echo "Multimodal Agent with Fused Embeddings (v4)"
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
echo "  Use OpenAI LLM: $USE_OPENAI_LLM"
if [ "$USE_OPENAI_LLM" = "true" ]; then
    echo "  OpenAI Model: $OPENAI_MODEL_NAME"
    echo "  OpenAI Temperature: $OPENAI_TEMPERATURE"
else
    echo "  Local Model: $LOCAL_MODEL_NAME"
    echo "  Device: $LOCAL_MODEL_DEVICE"
    echo "  Max New Tokens: $LOCAL_MODEL_MAX_NEW_TOKENS"
    echo "  Temperature: $LOCAL_MODEL_TEMPERATURE"
    echo "  Dtype: $LOCAL_MODEL_DTYPE"
fi
echo ""
echo "  Use RAG: $USE_RAG"
echo "  RAG Config: $RAG_CONFIG (fused multimodal)"
echo "  Render: $RENDER"
echo "  Test Runs: $TEST_RUNS"
echo "  Max Frames: $MAX_FRAMES"
echo "  Use Max Frames: $USE_MAX_FRAMES"
echo "  Embedding Method: $EMBEDDING_METHOD"
echo ""
echo "  Use Remote Server: $USE_REMOTE_SERVER"
echo "  Server URL: $MINERL_SERVER_URL"
echo "  ChromaDB: $CHROMA_HOST:$CHROMA_PORT"
echo "=============================================="
echo ""

# Ensure we're in the project directory
cd "$SLURM_SUBMIT_DIR"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not available."
    exit 1
fi

# Check for required API keys based on mode
if [ "$USE_OPENAI_LLM" = "true" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Warning: OPENAI_API_KEY not set. OpenAI mode requires an API key."
        echo "Set OPENAI_API_KEY in .env file or environment."
    fi
else
    # Check for HuggingFace token if using a gated model
    if [[ "$LOCAL_MODEL_NAME" == *"llama"* ]] || [[ "$LOCAL_MODEL_NAME" == *"Llama"* ]]; then
        if [ -z "$HF_TOKEN" ]; then
            echo "Warning: HF_TOKEN not set. Llama models require authentication."
            echo "Set HF_TOKEN environment variable or login with 'huggingface-cli login'"
        fi
    fi
fi

# Create results directory if it doesn't exist
mkdir -p Agent/Results

echo "Running command: uv run python -u Agent/agent_multimodal.py"
echo "RAG Configuration: $RAG_CONFIG (fused multimodal embeddings)"
echo ""

# Run the multimodal agent script using uv with unbuffered output (-u flag)
uv run python -u Agent/agent_multimodal.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Multimodal Agent job completed successfully at: $(date)"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "Multimodal Agent job failed at: $(date)"
    echo "=============================================="
    exit 1
fi
