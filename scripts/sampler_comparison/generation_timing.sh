#!/bin/bash
#SBATCH -J gen_timing                   # Job name
#SBATCH -o watch_folder/%x_%j.out       # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err       # log file (out & err)
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH --get-user-env                  # retrieve the users login environment
#SBATCH --mem=64G                       # server memory requested (per node)
#SBATCH -t 1:00:00                      # gen timing (4 arms) + same-node eval re-measure
#SBATCH --partition=gpu                 # Request partition
# NOTE: --exclusive was requested first (jobs 666921/666968, 666975/666976). With 0 of
# 38 a6000 nodes fully idle, slurm's estimated start was 2.5-6.5 h out, past the rebuttal
# budget. Falling back to shared: co-tenants are logged at start/end and per-sample
# variance is reported, so contention is auditable. Treat timings as UPPER BOUNDS.
##SBATCH --exclusive
#SBATCH --exclude=snavely-compute-02,abdelfattah-compute-02,seo-compute-02,davis-compute-01
#SBATCH --constraint="[a6000]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                    # Type/number of GPUs needed
#SBATCH --open-mode=append              # Do not overwrite logs
#SBATCH --requeue                       # Requeue upon preemption

# NeurIPS 2026 rebuttal, Reviewer 2: GENERATION wall-clock per NFE arm, so the
# cost table can report generation + metric per method rather than metric alone.
#
# Times a SUBSET of samples per arm (full 1000 at eval_batch_size=1 is ~1.5 h/arm)
# and reports mean +- sd per sample. Extrapolation is done in the writeup, not here.

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
echo "=== exclusive: ${SLURM_JOB_EXCLUSIVE:-unset} ==="
echo "=== co-tenant compute apps on this GPU at job start ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true

N=${N:-50}

for k in 1 2 4 8; do
    echo "=============================================================="
    echo "generation timing: block_greedy k=${k}  (n=${N})"
    echo "=============================================================="
    python -u scripts/generation_timing.py \
        --strategy block_greedy \
        --k ${k} \
        --n ${N} \
        --out "${OUTDIR}/greedy_k${k}.json" \
        2>&1 | tee "${PWD}/logs/gen_timing_greedy_k${k}.log"
done

# ---------------------------------------------------------------------------
# Re-measure the EVALUATION side on THIS SAME NODE.
#
# The ratio metric-cost : generation-cost is the number the rebuttal needs, and a
# ratio is only meaningful if numerator and denominator come from the same machine.
# The earlier evaluation numbers (job 665639) were taken on luxlab-compute-01;
# node-to-node spread on this cluster is up to ~2.4x (Agent D). Re-running the
# decomposition here makes the ratio internally consistent and node-invariant.
# gen-ppl/MAUVE were shown to be flat across NFE arms (0.06%), so one arm suffices.
# ---------------------------------------------------------------------------
echo "=============================================================="
echo "same-node evaluation re-measurement (k=1)"
echo "=============================================================="
mkdir -p "${PWD}/results/cost_decomposition"
python -u scripts/cost_decomposition.py \
    --sample-file "${PWD}/sample_logs/samples_bd3lm_len1024_blocksize16_block-greedy_k1.txt" \
    --label "samenode_greedy_k1" \
    --out "${PWD}/results/cost_decomposition/samenode_greedy_k1.json" \
    2>&1 | tee "${PWD}/logs/cost_decomp_samenode_k1.log"

echo "=== co-tenant compute apps at job end ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true
echo "ALL DONE"
