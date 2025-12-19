#!/bin/bash
#SBATCH --job-name=processing_pipeline
#SBATCH --output=.slurm_logs/run_processing_pipeline_%j.out
#SBATCH --error=.slurm_logs/run_processing_pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpunodes
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=khashayar1924@gmail.com

uv run python DatasetProcessing/run_pipeline.py --skip-step1 --skip-step2 --skip-step3 --skip-step4 --no-resume

print "Job completed successfully at: $(date)"
