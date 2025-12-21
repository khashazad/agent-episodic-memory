#!/bin/bash
#SBATCH --job-name=sliding_window_pipeline
#SBATCH --output=sliding_window_pipeline_%j.out
#SBATCH --error=sliding_window_pipeline_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpunodes
#SBATCH -c 8
#SBATCH --mem=80GB
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=khashayar1924@gmail.com

# Set default values (can be overridden with environment variables)
INPUT_DIR=${INPUT_DIR:-".data/MineRLTreechop-v0"}
OUTPUT_DIR=${OUTPUT_DIR:-".data/sliding_window_dataset_complete"}
WINDOW_SIZE=${WINDOW_SIZE:-16}
STRIDE=${STRIDE:-8}
EPISODES=${EPISODES:-""}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-".ckpts/attn.pth"}
BATCH_SIZE=${BATCH_SIZE:-64}
VLM_MODEL=${VLM_MODEL:-"Qwen/Qwen2.5-VL-7B-Instruct"}
DEVICE=${DEVICE:-"auto"}
START_WINDOW=${START_WINDOW:-0}
END_WINDOW=${END_WINDOW:-""}
NO_RESUME=${NO_RESUME:-false}
SKIP_WINDOWING=${SKIP_WINDOWING:-true}
SKIP_EMBEDDINGS=${SKIP_EMBEDDINGS:-true}
SKIP_DESCRIPTIONS=${SKIP_DESCRIPTIONS:-false}

# Print job information
echo "=========================================="
echo "SLIDING WINDOW PIPELINE JOB"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Working directory: $PWD"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo ""

# Print configuration
echo "Configuration:"
echo "  Input directory: $INPUT_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Window size: $WINDOW_SIZE"
echo "  Stride: $STRIDE"
echo "  Episodes limit: ${EPISODES:-All}"
echo "  Checkpoint path: $CHECKPOINT_PATH"
echo "  Batch size: $BATCH_SIZE"
echo "  VLM model: $VLM_MODEL"
echo "  Device: $DEVICE"
echo "  Start window: $START_WINDOW"
echo "  End window: ${END_WINDOW:-All}"
echo "  No resume: $NO_RESUME"
echo "  Skip windowing: $SKIP_WINDOWING"
echo "  Skip embeddings: $SKIP_EMBEDDINGS"
echo "  Skip descriptions: $SKIP_DESCRIPTIONS"
echo ""

# Ensure we're in the project directory
cd "$SLURM_SUBMIT_DIR"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not available."
    exit 1
fi

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist"
    exit 1
fi

# Check if checkpoint exists (only if not skipping embeddings)
if [ "$SKIP_EMBEDDINGS" != "true" ] && [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file $CHECKPOINT_PATH does not exist"
    exit 1
fi

# Build command arguments
ARGS="--input-dir $INPUT_DIR --output-dir $OUTPUT_DIR"
ARGS="$ARGS --window-size $WINDOW_SIZE --stride $STRIDE"
ARGS="$ARGS --checkpoint $CHECKPOINT_PATH --batch-size $BATCH_SIZE"
ARGS="$ARGS --vlm-model $VLM_MODEL --device $DEVICE"
ARGS="$ARGS --start-window $START_WINDOW"

# Add optional arguments
if [ -n "$EPISODES" ]; then
    ARGS="$ARGS --episodes $EPISODES"
fi

if [ -n "$END_WINDOW" ]; then
    ARGS="$ARGS --end-window $END_WINDOW"
fi

if [ "$NO_RESUME" = "true" ]; then
    ARGS="$ARGS --no-resume"
fi

if [ "$SKIP_WINDOWING" = "true" ]; then
    ARGS="$ARGS --skip-windowing"
fi

if [ "$SKIP_EMBEDDINGS" = "true" ]; then
    ARGS="$ARGS --skip-embeddings"
fi

if [ "$SKIP_DESCRIPTIONS" = "true" ]; then
    ARGS="$ARGS --skip-descriptions"
fi

echo "Running command: uv run python DatasetProcessing/process_sliding_window_pipeline.py $ARGS"
echo ""

# Start timing
START_TIME=$(date +%s)

# Run the pipeline script using uv
uv run python DatasetProcessing/process_sliding_window_pipeline.py $ARGS

# Calculate runtime
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
HOURS=$((RUNTIME / 3600))
MINUTES=$(((RUNTIME % 3600) / 60))
SECONDS=$((RUNTIME % 60))

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "JOB COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo "Job completed at: $(date)"
    echo "Total runtime: $(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)"
    echo "Output dataset: $OUTPUT_DIR"

    # Print dataset summary if available
    if [ -f "$OUTPUT_DIR/dataset_summary.json" ]; then
        echo ""
        echo "Dataset Summary:"
        cat "$OUTPUT_DIR/dataset_summary.json" | python -m json.tool
    fi

else
    echo ""
    echo "=========================================="
    echo "JOB FAILED!"
    echo "=========================================="
    echo "Job failed at: $(date)"
    echo "Runtime before failure: $(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)"
    exit 1
fi
