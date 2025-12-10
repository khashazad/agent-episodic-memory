#!/bin/bash
#SBATCH --job-name=embed_text_descriptions
#SBATCH --output=embed_text_descriptions_%j.out
#SBATCH --error=embed_text_descriptions_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpunodes
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx_a2000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=khashayar1924@gmail.com

DATA_DIR=${DATA_DIR:-".data/chunked_dataset_with_embeddings_and_llm_descriptions"}
CHECKPOINT=${CHECKPOINT:-".ckpts/attn.pth"}
BATCH_SIZE=${BATCH_SIZE:-64}
FORCE_RECOMPUTE=${FORCE_RECOMPUTE:-false}

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Working directory: $PWD"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"

# Print configuration
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  MineCLIP checkpoint: $CHECKPOINT"
echo "  Batch size: $BATCH_SIZE"
echo "  Force recompute: $FORCE_RECOMPUTE"

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

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: MineCLIP checkpoint $CHECKPOINT does not exist"
    exit 1
fi

# Build command arguments
ARGS="--data-dir $DATA_DIR --checkpoint $CHECKPOINT --batch-size $BATCH_SIZE"

if [ "$FORCE_RECOMPUTE" = "true" ]; then
    ARGS="$ARGS --force-recompute"
fi

echo "Running command: uv run python DatasetProcessing/embed_text_descriptions.py $ARGS"

# Run the text embedding script using uv
uv run python DatasetProcessing/embed_text_descriptions.py $ARGS

# Check exit status
if [ $? -eq 0 ]; then
    echo "Job completed successfully at: $(date)"
else
    echo "Job failed at: $(date)"
    exit 1
fi