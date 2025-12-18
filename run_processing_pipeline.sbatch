#!/bin/bash
#SBATCH --job-name=rag_test
#SBATCH --output=rag_test_%A_%a.out
#SBATCH --error=rag_test_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpunodes
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=khashayar1924@gmail.com

uv run python DatasetProcessing/run_processing_pipeline.py

print "Job completed successfully at: $(date)"
