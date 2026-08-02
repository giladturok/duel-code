#!/bin/bash
#SBATCH -J ek_owtbs
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH --partition=gpu
#SBATCH --exclude=sun-compute-03,snavely-compute-02,portal-compute-01,lancer-compute-01,sun-compute-01,kuleshov-compute-02
#SBATCH --constraint="[a6000|a100|h100|h200|6000ada|a40]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append

# How much does the PAPER's headline DUEL perplexity move with eval batch size?
# scripts/sampler_comparison/duel_ppl.sh runs block_greedy k=1 at eval_batch_size=64.
# The confidence-based reveal order is not batch-size invariant (see run_diag.sh),
# so re-run the identical 320-sequence OWT slice at 64 / 16 / 8 and compare.
# Reference: scripts/selflik_anchors/out/gate_owt320_bf16.npz -> micro ppl 22.505990.
set -u

PY=/home/gt345/.conda/envs/bd3lm/bin/python
ROOT=/home/gt345/projects/scaling/duel
OUT=${ROOT}/scripts/entropy_kl/out/owtbs

mkdir -p "${OUT}" "${ROOT}/watch_folder"
cd "${ROOT}" || exit 1
export TRITON_NUM_STAGES=1
echo "=== owt batch-size sweep on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for BS in 64 16 8; do
  echo "--- OWT 320 seqs, block_greedy k=1, eval_batch_size=${BS}"
  ${PY} -u "${ROOT}/scripts/selflik_anchors/duel_ll.py" --source owt --limit 320 \
      --precision bf16 --batch-size "${BS}" \
      --out "${OUT}/owt320_bs${BS}.npz" 2>&1 | grep -E "micro-averaged|Error|Traceback"
done
echo "=== owt batch-size sweep done ==="
