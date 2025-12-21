#!/bin/bash
#SBATCH --job-name=processing_pipeline
#SBATCH --output=.slurm_logs/%j/output.out
#SBATCH --error=.slurm_logs/%j/error.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpunodes
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=khashayar1924@gmail.com

uv run python DatasetProcessing/run_pipeline.py --skip-step1 --skip-step2 --skip-step3 --skip-step4 --no-resume

echo "Job completed successfully at: $(date)"
