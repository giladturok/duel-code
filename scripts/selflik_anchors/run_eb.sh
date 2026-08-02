#!/bin/bash
#SBATCH -J eb_selflik                   # Job name
#SBATCH -o watch_folder/%x_%j.out       # log file (out)
#SBATCH -e watch_folder/%x_%j.err       # log file (err)
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --partition=kuleshov,gpu
#SBATCH --exclude=sun-compute-03,snavely-compute-02,portal-compute-01,lancer-compute-01,sun-compute-01
#SBATCH --constraint="[a6000|a100|h100|h200|a5000|6000ada|a40]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --requeue

# E-B: self-likelihood anchor check.
#   D_hat = log(gen_ppl) - log(duel_ppl), per passage, per-token nats.
# usage: sbatch run_eb.sh <stage> [n_per_cell]
#   stage=gate     reproduce val/exact_ppl=22.0434 on openwebtext-valid-1k
#   stage=anchors  DUEL exact LL + GPT-2 Large ppl on the 4 anchor cells

set -u

PY=/home/gt345/.conda/envs/bd3lm/bin/python
ROOT=/home/gt345/projects/scaling/duel
SA=${ROOT}/scripts/selflik_anchors
OUT=${SA}/out

STAGE=${1:-anchors}
N=${2:-256}

mkdir -p "${OUT}" "${ROOT}/watch_folder"
cd "${ROOT}" || exit 1
export TRITON_NUM_STAGES=1
echo "=== ${STAGE} n=${N} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

case "${STAGE}" in
  gate)
    # (1) 320-sequence bf16-vs-fp32 sensitivity, then (2) the full 1088-sequence
    # reproduction of the published block_greedy k=1 number.
    ${PY} -u "${SA}/duel_ll.py" --source owt --limit 320 --precision bf16 \
        --batch-size 64 --out "${OUT}/gate_owt320_bf16.npz" || exit 1
    ${PY} -u "${SA}/duel_ll.py" --source owt --limit 320 --precision fp32 \
        --batch-size 64 --out "${OUT}/gate_owt320_fp32.npz" || exit 1
    # (3) batching invariance on variable-length anchors: bs=1, unsorted, vs the
    # length-sorted bs=64 production run.
    ${PY} -u "${SA}/duel_ll.py" --source anchors --n-per-cell 8 --batch-size 1 \
        --no-sort --out "${OUT}/gate_anchors_bs1.npz" || exit 1
    # (4) the full 1088-sequence reproduction of the published number.
    ${PY} -u "${SA}/duel_ll.py" --source owt --precision bf16 \
        --batch-size 64 --out "${OUT}/gate_owt_full_bf16.npz" || exit 1
    ;;
  anchors)
    ${PY} -u "${SA}/gen_ppl.py" --n-per-cell "${N}" \
        --out "${OUT}/genppl_n${N}.npz" || exit 1
    ${PY} -u "${SA}/duel_ll.py" --source anchors --n-per-cell "${N}" \
        --batch-size 64 --out "${OUT}/duel_n${N}.npz" || exit 1
    ;;
  cell)
    # stage=cell <n> <cell-name>: one anchor kind, for parallel n=1000 runs.
    CELL=${3:?need a cell name}
    ${PY} -u "${SA}/duel_ll.py" --source anchors --cells "${CELL}" \
        --n-per-cell "${N}" --batch-size 64 \
        --out "${OUT}/duel_n${N}_${CELL}.npz" || exit 1
    ;;
  genppl)
    ${PY} -u "${SA}/gen_ppl.py" --n-per-cell "${N}" \
        --out "${OUT}/genppl_n${N}.npz" || exit 1
    ;;
  *)
    echo "unknown stage ${STAGE}"; exit 1;;
esac

echo "=== done ${STAGE} ==="
