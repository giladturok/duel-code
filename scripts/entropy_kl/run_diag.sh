#!/bin/bash
#SBATCH -J ek_diag
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --partition=kuleshov,gpu
#SBATCH --exclude=sun-compute-03,snavely-compute-02,portal-compute-01,lancer-compute-01,sun-compute-01,kuleshov-compute-02
#SBATCH --constraint="[a6000|a100|h100|h200|a5000|6000ada|a40|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append

# Diagnostic for the non-zero KL(P_A||P_A) diagonal on the two confidence-based
# rules. Hypothesis: the reveal ORDER diverges between generation and scoring
# because the model forward is only reproducible at a fixed batch size, and a
# ~1e-6 perturbation of the logits flips near-ties in the argmax over positions.
# Test: re-score at the SAME batch size that generated the samples (8). If the
# diagonal collapses to fp noise, the hypothesis holds and the fix is to match
# batch sizes; if not, the cause is elsewhere.
set -u

PY=/home/gt345/.conda/envs/bd3lm/bin/python
ROOT=/home/gt345/projects/scaling/duel
EK=${ROOT}/scripts/entropy_kl
OUT=${EK}/out/diag

mkdir -p "${OUT}" "${ROOT}/watch_folder"
cd "${ROOT}" || exit 1
export TRITON_NUM_STAGES=1
echo "=== diag on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for BS in 8 16; do
  for RULE in block_greedy block_probability_margin; do
    echo "--- self-score ${RULE} k=8 at batch_size=${BS} (generation used 8)"
    ${PY} -u "${EK}/score_cross.py" \
        --samples "${EK}/out/gen_${RULE}_k8_n64.npz" \
        --score-strategy "${RULE}" --score-k 8 --batch-size "${BS}" \
        --out "${OUT}/self_${RULE}_k8_bs${BS}.npz" 2>&1 | tail -6
  done
done
echo "=== diag done ==="
