#!/bin/bash
#SBATCH -J ek_smoke
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH --partition=kuleshov,gpu
#SBATCH --exclude=sun-compute-03,snavely-compute-02,portal-compute-01,lancer-compute-01,sun-compute-01,kuleshov-compute-02
#SBATCH --constraint="[a6000|a100|h100|h200|a5000|6000ada|a40|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append

# End-to-end smoke test of the entropy/KL pipeline on 4 sequences.
#   1. generate under block_greedy k=8, tracing both estimators
#   2. re-score the SAME sequences under block_greedy k=8 -> KL must be ~0
#   3. re-score under block_left_to_right k=8 -> KL must be > 0
#   4. regression gate: the stock OWT DUEL scorer path is unchanged
set -u

PY=/home/gt345/.conda/envs/bd3lm/bin/python
ROOT=/home/gt345/projects/scaling/duel
EK=${ROOT}/scripts/entropy_kl
OUT=${EK}/out/smoke

mkdir -p "${OUT}" "${ROOT}/watch_folder"
cd "${ROOT}" || exit 1
export TRITON_NUM_STAGES=1
echo "=== smoke on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "--- (1) generate"
${PY} -u "${EK}/gen_entropy.py" --strategy block_greedy --k 8 -n 4 \
    --batch-size 4 --out "${OUT}/gen_greedy_k8_n4.npz" || exit 1

echo "--- (2) self-score (KL must be ~0)"
${PY} -u "${EK}/score_cross.py" --samples "${OUT}/gen_greedy_k8_n4.npz" \
    --score-strategy block_greedy --score-k 8 --batch-size 4 \
    --out "${OUT}/kl_greedy_greedy_k8.npz" || exit 1

echo "--- (3) cross-score under left_to_right (KL must be > 0)"
${PY} -u "${EK}/score_cross.py" --samples "${OUT}/gen_greedy_k8_n4.npz" \
    --score-strategy block_left_to_right --score-k 8 --batch-size 4 \
    --out "${OUT}/kl_greedy_l2r_k8.npz" || exit 1

echo "--- (4) regression gate: stock OWT scorer unchanged (expect 22.5060 on 320 seqs)"
${PY} -u "${ROOT}/scripts/selflik_anchors/duel_ll.py" --source owt --limit 320 \
    --precision bf16 --batch-size 64 --out "${OUT}/regress_owt320.npz" || exit 1

echo "=== smoke done ==="
