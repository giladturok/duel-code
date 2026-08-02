#!/bin/bash
#SBATCH -J ek_kl                        # Job name
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

# Stage 2: exact KL(P_A || P_B) between two unmasking rules at a fixed NFE budget.
#   sbatch --array=0-8 --dependency=afterok:<stage1_jobid> run_stage2.sh <k> [n] [bs]
# Array index -> (A, B):  A = idx / 3 (generating rule), B = idx % 3 (scoring rule).
# The diagonal A == B is the correctness gate: it must reproduce Stage 1's
# accumulated path log-prob sequence by sequence, i.e. KL(P_A||P_A) = 0.

set -u

PY=/home/gt345/.conda/envs/bd3lm/bin/python
ROOT=/home/gt345/projects/scaling/duel
EK=${ROOT}/scripts/entropy_kl
OUT=${EK}/out

# BS MUST MATCH THE BATCH SIZE USED TO GENERATE THE SAMPLES (run_stage1.sh, 8).
# The model forward is not batch-size invariant at the ~1e-6 level, and the two
# confidence-based rules turn that into a *different reveal order*, which changes
# the exact likelihood by ~0.016 nats/token (1.6% in perplexity). Measured: the
# greedy self-scoring diagonal is 0.0164 at bs=16 vs 2.2e-6 at bs=8. Rules whose
# order is position-based (block_left_to_right) are immune. See run_diag.sh.
#
# Memory: `_pad_block_logits_to_full` allocates one [B, L, V] float32 tensor per
# step, 0.2 GB per sequence in the batch; bs=8 peaks at ~2 GB.
K=${1:?need k}
N=${2:-256}
BS=${3:-8}

RULES=(block_greedy block_left_to_right block_probability_margin)
IDX=${SLURM_ARRAY_TASK_ID:-0}
A=${RULES[$((IDX / 3))]}
B=${RULES[$((IDX % 3))]}

mkdir -p "${OUT}" "${ROOT}/watch_folder"
cd "${ROOT}" || exit 1
export TRITON_NUM_STAGES=1
echo "=== stage2 gen=${A} score=${B} k=${K} n=${N} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

${PY} -u "${EK}/score_cross.py" \
    --samples "${OUT}/gen_${A}_k${K}_n${N}.npz" \
    --score-strategy "${B}" --score-k "${K}" \
    --batch-size "${BS}" \
    --out "${OUT}/kl_${A}__${B}_k${K}_n${N}.npz" || exit 1

echo "=== done stage2 ${A} -> ${B} k=${K} ==="
