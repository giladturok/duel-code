#!/bin/bash
#SBATCH -J ek_gen                       # Job name
#SBATCH -o watch_folder/%x_%A_%a.out    # log file (out)
#SBATCH -e watch_folder/%x_%A_%a.err    # log file (err)
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH --partition=kuleshov,gpu
#SBATCH --exclude=sun-compute-03,snavely-compute-02,portal-compute-01,lancer-compute-01,sun-compute-01,kuleshov-compute-02
#SBATCH --constraint="[a6000|a100|h100|h200|a5000|6000ada|a40|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --requeue

# Stage 1: exact entropy H(P_F) of each deterministic unmasking rule.
#   sbatch --array=0-11 run_stage1.sh [n] [batch_size]
# Array index -> cell:  idx = rule_idx * 4 + k_idx,  k in {1,2,4,8}.
# Rules are the three DETERMINISTIC ones only; `uniform` is excluded on purpose
# (for a stochastic policy the path log-prob is a bound, not the likelihood).

set -u

PY=/home/gt345/.conda/envs/bd3lm/bin/python
ROOT=/home/gt345/projects/scaling/duel
EK=${ROOT}/scripts/entropy_kl
OUT=${EK}/out

# Memory: `_unmask_block_with_strategy` holds three [B, L, V] float32 tensors
# (full_logits, p_x0, log_probs) once the sequence reaches full width, i.e.
# 3 * B * 1024 * 50258 * 4 B = 0.6 GB per sequence in the batch. bs=8 peaks at
# ~5 GB and fits a 24 GB 3090; bs=32 OOMs there.
N=${1:-256}
BS=${2:-8}

RULES=(block_greedy block_left_to_right block_probability_margin)
KS=(1 2 4 8)

IDX=${SLURM_ARRAY_TASK_ID:-0}
RULE=${RULES[$((IDX / 4))]}
K=${KS[$((IDX % 4))]}

mkdir -p "${OUT}" "${ROOT}/watch_folder"
cd "${ROOT}" || exit 1
export TRITON_NUM_STAGES=1
echo "=== stage1 ${RULE} k=${K} n=${N} bs=${BS} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

${PY} -u "${EK}/gen_entropy.py" \
    --strategy "${RULE}" --k "${K}" -n "${N}" --batch-size "${BS}" \
    --gen-ppl \
    --out "${OUT}/gen_${RULE}_k${K}_n${N}.npz" || exit 1

echo "=== done stage1 ${RULE} k=${K} ==="
