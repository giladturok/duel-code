#!/bin/bash
#SBATCH --constraint="[a100|h100|h200]"
#SBATCH --partition=gpu
#SBATCH --exclude=snavely-compute-02
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --mem=256G                  # We require 128GB memoray per GPU for dataset tokenization
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out     # %x = job name, %j = job ID 
#SBATCH --error=logs/%x-%j.err      # %x = job name, %j = job ID
#SBATCH --open-mode=append          # Do not overwrite logs
#SBATCH --requeue                   # Requeue upon preemption

export HF_HOME=/share/kuleshov/gt345/.cache/huggingface
export HF_TRUST_REMOTE_CODE=1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Running command: $@"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate dllm

cd /home/gt345/projects/scaling/exact_ll

eval "$@"

echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="