#!/bin/bash
#SBATCH -J gen_timing_k16               # Job name
#SBATCH -o watch_folder/%x_%j.out       # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err       # log file (out & err)
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH --partition=gpu
#SBATCH --nodelist=fang-compute-01      # SAME NODE as the metric timings (job 667533),
                                        # or the metric:generation ratio is corrupted.
#SBATCH --constraint="[a6000]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --requeue

# NeurIPS 2026 rebuttal, Reviewer 2: the k=16 generation arm.
#
# At k=16 = L' the whole block is revealed in a single unmasking step, so the true
# forward count is 64 blocks x (ceil(16/16) unmask + 1 block-commit) = 128 -- the
# cheapest sampler on the axis, and therefore the arm where a fixed metric cost is
# the largest fraction of total cost. This is the high-k end that makes the trend
# argument land.
#
# NOTE: there is no k=16 file in sample_logs/ (the paper swept k in {1,2,4,8}), but
# none is needed: this measures GENERATION only.

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate bd3lm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_NUM_STAGES=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUTDIR="${PWD}/results/generation_timing"
mkdir -p "${OUTDIR}" logs

echo "=== NODE: ${SLURMD_NODENAME:-unknown}  JOB: ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true

python -u scripts/generation_timing.py \
    --strategy block_greedy \
    --k 16 \
    --n 50 \
    --out "${OUTDIR}/greedy_k16.json" \
    2>&1 | tee "${PWD}/logs/gen_timing_greedy_k16.log"

echo "=== co-tenants at end ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true
echo "ALL DONE"
