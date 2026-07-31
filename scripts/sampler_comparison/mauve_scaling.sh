#!/bin/bash
#SBATCH -J mauve_scaling                # Job name
#SBATCH -o watch_folder/%x_%j.out       # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err       # log file (out & err)
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64G
#SBATCH -t 0:40:00                      # short: helps backfill onto an exclusive node
#SBATCH --partition=gpu
# NOTE: --exclusive was requested first (jobs 666921/666968, 666975/666976). With 0 of
# 38 a6000 nodes fully idle, slurm's estimated start was 2.5-6.5 h out, past the rebuttal
# budget. Falling back to shared: co-tenants are logged at start/end and per-sample
# variance is reported, so contention is auditable. Treat timings as UPPER BOUNDS.
##SBATCH --exclusive
#SBATCH --exclude=snavely-compute-02,abdelfattah-compute-02,seo-compute-02,davis-compute-01
#SBATCH --constraint="[a6000]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --requeue

# NeurIPS 2026 rebuttal, Reviewer 2: how MAUVE's cost scales with sample count,
# so we can state its cost at MAUVE's recommended ~5000 rather than guess.

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate bd3lm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results logs

echo "=== NODE: ${SLURMD_NODENAME:-unknown}  JOB: ${SLURM_JOB_ID:-none} ==="
echo "=== exclusive: ${SLURM_JOB_EXCLUSIVE:-unset} ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true

python -u scripts/mauve_scaling.py \
    --sample-file "${PWD}/sample_logs/samples_bd3lm_len1024_blocksize16_block-greedy_k1.txt" \
    --out "${PWD}/results/mauve_scaling.json" \
    2>&1 | tee "${PWD}/logs/mauve_scaling.log"

echo "=== co-tenants at end ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true
echo "ALL DONE"
