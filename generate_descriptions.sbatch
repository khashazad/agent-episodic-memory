#!/bin/bash
#SBATCH --job-name=gen_descriptions
#SBATCH --output=gen_descriptions_%j.out
#SBATCH --error=gen_descriptions_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpunodes
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx_a2000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=khashayar1924@gmail.com

VIDEOS_DIR=${VIDEOS_DIR:-".data/MineRLTreechop-v0"}
CHUNKS_DIR=${CHUNKS_DIR:-".data/chunked_dataset_with_embeddings"}
MODEL=${MODEL:-"Qwen/Qwen2.5-VL-7B-Instruct"}
NUM_FRAMES=${NUM_FRAMES:-32}

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Working directory: $PWD"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"

# Print configuration
echo "Configuration:"
echo "  Videos directory: $VIDEOS_DIR"
echo "  Chunks directory: $CHUNKS_DIR"
echo "  Model: $MODEL"
echo "  Num frames: $NUM_FRAMES"

# Ensure we're in the project directory
cd "$SLURM_SUBMIT_DIR"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not available."
    exit 1
fi

if [ ! -d "$VIDEOS_DIR" ]; then
    echo "Error: Videos directory $VIDEOS_DIR does not exist"
    exit 1
fi

if [ ! -d "$CHUNKS_DIR" ]; then
    echo "Error: Chunks directory $CHUNKS_DIR does not exist"
    exit 1
fi

# Build command arguments
ARGS="--videos-dir $VIDEOS_DIR --chunks-dir $CHUNKS_DIR --model $MODEL --num-frames $NUM_FRAMES --resume"

echo "Running command: uv run python DatasetProcessing/generate_descriptions.py $ARGS"

# Run the description generation script using uv
uv run python DatasetProcessing/generate_descriptions.py $ARGS

# Check exit status
if [ $? -eq 0 ]; then
    echo "Job completed successfully at: $(date)"
else
    echo "Job failed at: $(date)"
    exit 1
fi
