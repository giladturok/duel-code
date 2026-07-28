#!/bin/bash
#SBATCH -J oracle_decomp                # Job name
#SBATCH -o watch_folder/%x_%j.out       # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err       # log file (out & err)
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH --get-user-env                  # retrieve the users login environment
#SBATCH --mem=64G                       # server memory requested (per node)
#SBATCH -t 24:00:00                     # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                 # Request partition
#SBATCH --exclude=sun-compute-03,snavely-compute-02,portal-compute-01,kuleshov-compute-02,lancer-compute-01,sun-compute-01,snavely-compute-04,snavely-compute-11,snavely-compute-12,sablab-gpu-04
#SBATCH --constraint="[a6000|a100|h100|h200]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1                    # Type/number of GPUs needed
#SBATCH --open-mode=append              # Do not overwrite logs
#SBATCH --requeue                       # Requeue upon preemption

# NeurIPS 2026 rebuttal, Reviewer 3 W2/Q1: four-way decomposition of the
# DUEL gap on AG News with BD3-LM L'=4.
#
# The oracle path already enumerates all 4! = 24 within-block unmask orders and
# evaluates the block log-likelihood under each. Because every permutation ends
# the block fully revealed to ground truth, the post-block state is identical
# across orders, so block contributions are independent and the same 24-value
# table reduces three ways at no extra cost:
#
#   oracle          = sum_blocks max_j  ll_j          (existing, Table 7)
#   uniform-order   = sum_blocks mean_j ll_j          (new; = the MDM ELBO in theory)
#   mixture         = sum_blocks (logsumexp_j ll_j - log 24)   (new)
#
# Logged as val/exact_ppl, val/exact_ppl_uniform_order, val/exact_ppl_mixture.
#
# MODE=oracle  reproduces val/exact_ppl = 36.336 (outputs/ag_news/2026.05.04/162506)
#              and emits the two new columns in the same job. ~2.5 h at bs=128.
#              (Published Table 7 says 36.47; the on-disk measurement is 36.336.
#               Unreconciled -- see ai_diary.md.)
# MODE=greedy  block greedy-confidence DUEL, same token set. ~1/24 the cost.
# MODE=elbo    ELBO / NELBO, same token set.
#
# Single-GPU only: multi-GPU is broken here (Lightning SLURM auto-detect hang,
# and NCCL ncclCommInit failures on several nodes). See ai_diary.md 2026-07-28.
#
# Smoke test:
#   MODE=oracle SMOKE=1 sbatch scripts/sampler_comparison/oracle_decomposition.sh
# Real runs:
#   MODE=oracle sbatch scripts/sampler_comparison/oracle_decomposition.sh
#   MODE=greedy sbatch scripts/sampler_comparison/oracle_decomposition.sh
#   MODE=elbo   sbatch scripts/sampler_comparison/oracle_decomposition.sh

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate bd3lm

export TRITON_NUM_STAGES=1

MODE="${MODE:-oracle}"
SMOKE="${SMOKE:-0}"

DATA="ag_news"
BLOCK_SIZE=4
MODEL_LENGTH=1024
CKPT="kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE}"

# Matches outputs/ag_news/2026.05.04/162506 (the run that produced 36.336).
EVAL_BATCH_SIZE=128
EXTRA=()

if [ "${SMOKE}" = "1" ]; then
    EVAL_BATCH_SIZE=1
    MODEL_LENGTH=64
    EXTRA+=(trainer.limit_val_batches=1)
    TAG="smoke_${MODE}"
else
    TAG="${MODE}"
fi

case "${MODE}" in
  oracle)
    EXTRA+=(mode=duel_ppl
            eval.exact_ll_strategy=block_permutation
            eval.exact_ll_k=1
            +eval.exact_ll_use_kv_cache=true
            sampling.kv_cache=true
            model.attn_backend=sdpa)
    ;;
  greedy)
    EXTRA+=(mode=duel_ppl
            eval.exact_ll_strategy=block_greedy
            eval.exact_ll_k=1
            +eval.exact_ll_use_kv_cache=true
            sampling.kv_cache=true
            model.attn_backend=sdpa)
    ;;
  elbo)
    # Matches outputs/ag_news/2026.03.24/151831 (val/ppl = 61.6733, i.e. the
    # published Table 7 "ELBO <= 61.67"), so this column is validated too.
    [ "${SMOKE}" = "1" ] || EVAL_BATCH_SIZE=16
    EXTRA+=(mode=elbo_ppl
            model.attn_backend=flex)
    ;;
  *)
    echo "Unknown MODE=${MODE} (expected oracle|greedy|elbo)" >&2
    exit 1
    ;;
esac

mkdir -p "${PWD}/logs" "${PWD}/watch_folder"
LOG="${PWD}/logs/bd3lm_${DATA}_block_size${BLOCK_SIZE}_decomp_${TAG}.log"

echo "MODE=${MODE} SMOKE=${SMOKE} bs=${EVAL_BATCH_SIZE} len=${MODEL_LENGTH} -> ${LOG}"

python -u main.py \
    loader.eval_batch_size=${EVAL_BATCH_SIZE} \
    model=small \
    algo=bd3lm \
    algo.backbone=hf_dit \
    data=${DATA} \
    +data.insert_valid_eos=False \
    model.length=${MODEL_LENGTH} \
    block_size=${BLOCK_SIZE} \
    eval.checkpoint_path=${CKPT} \
    wandb.project=duel \
    +wandb.name="decomp_${TAG}_${DATA}_block_size${BLOCK_SIZE}" \
    "${EXTRA[@]}" 2>&1 | tee "${LOG}"
