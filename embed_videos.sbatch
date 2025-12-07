#!/bin/bash
#SBATCH --job-name=embed_videos
#SBATCH --output=embed_videos_%j.out
#SBATCH --error=embed_videos_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=khashayar1924@gmail.com

# Set default values (can be overridden with environment variables)
DATA_DIR=${DATA_DIR:-".data/chunked_dataset"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-".ckpts/attn.pth"}
BATCH_SIZE=${BATCH_SIZE:-16}

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Working directory: $PWD"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"

# Print configuration
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Checkpoint path: $CHECKPOINT_PATH"
echo "  Batch size: $BATCH_SIZE"
echo "  Force recompute: $FORCE_RECOMPUTE"
echo "  Single mode: $SINGLE_MODE"

# Ensure we're in the project directory
cd "$SLURM_SUBMIT_DIR"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not available."
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file $CHECKPOINT_PATH does not exist"
    exit 1
fi

# Build command arguments
ARGS="--data-dir $DATA_DIR --checkpoint $CHECKPOINT_PATH --batch-size $BATCH_SIZE"

echo "Running command: uv run python DatasetProcessing/embed_videos.py $ARGS"

# Run the embedding script using uv
uv run python DatasetProcessing/embed_videos.py $ARGS

# Check exit status
if [ $? -eq 0 ]; then
    echo "Job completed successfully at: $(date)"
else
    echo "Job failed at: $(date)"
    exit 1
fi
