#!/bin/bash
#SBATCH -J cost_decomp                  # Job name
#SBATCH -o watch_folder/%x_%j.out       # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err       # log file (out & err)
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH --get-user-env                  # retrieve the users login environment
#SBATCH --mem=64G                       # server memory requested (per node)
#SBATCH -t 4:00:00                      # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                 # Request partition
#SBATCH --exclude=snavely-compute-02,abdelfattah-compute-02,seo-compute-02,davis-compute-01
#SBATCH --constraint="[a6000]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                    # Type/number of GPUs needed
#SBATCH --open-mode=append              # Do not overwrite logs
#SBATCH --requeue                       # Requeue upon preemption

# NeurIPS 2026 rebuttal, Reviewer 2: cost decomposition of sample-set
# evaluation. Substantiates (or refutes) the claim at
# sections/evaluation.tex:20-24 that DUEL is "strictly cheaper than generative
# perplexity".
#
# READ-ONLY: generates nothing. Consumes the 1000-sample files already in
# sample_logs/ and times each evaluation component separately.

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate bd3lm

# Identical across every component, or the memory numbers are incomparable.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUTDIR="${PWD}/results/cost_decomposition"
mkdir -p "${OUTDIR}" logs

run_one() {
    local file=$1
    local label=$2
    shift 2

    echo "=============================================================="
    echo "cost decomposition: ${label}"
    echo "=============================================================="

    local mem="${OUTDIR}/${label}_mem.csv"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -l 2 \
        > "${mem}" &
    local SMI=$!

    python -u scripts/cost_decomposition.py \
        --sample-file "${file}" \
        --label "${label}" \
        --out "${OUTDIR}/${label}.json" \
        "$@" 2>&1 | tee "${PWD}/logs/cost_decomp_${label}.log"

    kill ${SMI} 2>/dev/null || true
    wait ${SMI} 2>/dev/null || true

    echo "[peak nvidia-smi MiB] $(sort -n "${mem}" | tail -1)"
}

SL="${PWD}/sample_logs"

# High-NFE arm (k=1, ~1024 NFE) and low-NFE arm (k=8, ~128 NFE). Both BD3-LM:
# sample_logs/ contains no MDLM sample set (see rebuttal writeup).
run_one "${SL}/samples_bd3lm_len1024_blocksize16_block-greedy_k1.txt" \
        "bd3lm_greedy_k1"
run_one "${SL}/samples_bd3lm_len1024_blocksize16_block-greedy_k8.txt" \
        "bd3lm_greedy_k8"

echo "ALL DONE"
