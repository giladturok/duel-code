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
#SBATCH --cpus-per-task=8               # >=8 PER RANK or multi-GPU deadlocks (see trap 3)
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
# MODE=oracle_dp  the subset-lattice DP. Same answer as MODE=oracle (identical
#              oracle / uniform-order / mixture columns), but it costs 2^L'-1
#              forwards per block instead of L'! * L': the L'! orders visit only
#              2^L'-1 distinct prefix SETS, and the block log-likelihood depends
#              on the revealed set, not the order it was revealed in. At L'=8 and
#              model.length=1024 that is 32,512 forwards per sequence versus
#              41,287,680 for brute-force enumeration -- a 1270x saving, and the
#              only reason L'=8 is feasible at all. At L'=4 it is 480 vs 96, so
#              prefer MODE=oracle there for the published number and use
#              MODE=oracle_dp as the cross-check.
# MODE=l2r / margin / conf_thresh
#              the other three selection rules for the Table 6 rows, on the same
#              token set and the same exact-LL path. conf_thresh uses
#              exact_ll_k=0.95, which for that strategy is the probability
#              threshold tau (NOT a token count).
#
# Multi-GPU is exact (verified 2026-07-28, jobs 540766/540767/540769: 1 vs 2 vs 4
# GPUs agree to 1e-6 in NLL) but has THREE traps. Read all three before using it.
#   1. Requires srun with ntasks == devices; this script adds it automatically.
#   2. The validation DistributedSampler PADS BY DUPLICATING samples. The AG News
#      val set is exactly 384 sequences at model.length=1024, so keep
#      N_gpus * EVAL_BS a divisor of 384 or the average is silently biased.
#      Clean: 4x48, 4x32, 4x16, 2x64, 8x48, 6x64, 1x128.
#   3. FIXED, but do not regress it: multi-GPU needs >=8 CPUs PER RANK. At the
#      old --cpus-per-task=2 the ranks hang in an early 1-element collective
#      (SeqNum 3-4, entered ~50 s in, before any batch) and die at the 30 min
#      watchdog timeout with "GIL held inside a CUDA api" -- NCCL's proxy
#      threads starved of cores. Jobs 540775/542922 died this way at cpus=2;
#      547246 is the same srun launcher at cpus=8 and passes. This is the
#      long-standing BD3-LM multi-GPU hang (ai_diary.md 2026-07-28) and it does
#      NOT reproduce at MODEL_LEN=64, so smoke tests pass and lie to you.
#
# Smoke test:
#   MODE=oracle SMOKE=1 sbatch scripts/sampler_comparison/oracle_decomposition.sh
# Real runs (single GPU):
#   MODE=oracle sbatch scripts/sampler_comparison/oracle_decomposition.sh
#   MODE=greedy sbatch scripts/sampler_comparison/oracle_decomposition.sh
#   MODE=elbo   sbatch scripts/sampler_comparison/oracle_decomposition.sh
# Real run, 4 GPUs (~4x faster). Note --cpus-per-task=8: fewer deadlocks (trap 3).
#   MODE=oracle EVAL_BS=16 sbatch --gres=gpu:4 --ntasks-per-node=4 \
#       --cpus-per-task=8 --mem=128G --constraint="[6000ada|a6000]" \
#       scripts/sampler_comparison/oracle_decomposition.sh
# L'=8 oracle via the subset DP, 8 GPUs (8*48 = 384 = the whole val set, trap 2):
#   BLOCK_SZ=8 MODE=oracle_dp EVAL_BS=48 sbatch --gres=gpu:8 --ntasks-per-node=8 \
#       --cpus-per-task=8 --mem=256G --constraint="[6000ada]" \
#       scripts/sampler_comparison/oracle_decomposition.sh

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate bd3lm

export TRITON_NUM_STAGES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NOTE: at eval_batch_size=128 the duel_ppl path needs an 80 GB+ GPU.
# exact_likelihood._pad_block_logits_to_full allocates [B, 1024, 50258] fp32
# = 26 GB per forward, of which all but the current block is -inf and never
# read. A 48 GB a6000 OOMs. Launch the oracle/greedy modes with
#   sbatch --constraint="[h200|h100|a100]" ...
# (the a100 feature also covers a 40 GB variant on snavely-compute-02, which
# the --exclude line above already removes).

MODE="${MODE:-oracle}"
SMOKE="${SMOKE:-0}"

# Multi-GPU. Lightning's SLURM auto-detect requires srun with ntasks == devices;
# a bare `python` under --ntasks-per-node>1 hangs at rendezvous forever (see
# ai_diary.md 2026-07-28, jobs 488883 / 489013). trainer.devices resolves from
# torch.cuda.device_count() (config.yaml:68), so it follows --gres automatically.
# Override both on the sbatch line, e.g.
#   sbatch --gres=gpu:4 --ntasks-per-node=4 ...
# LAUNCHER=srun   : slurm launches the ranks (needs --ntasks-per-node == devices).
#                   Gets past rendezvous but hangs in early NCCL collectives at
#                   model.length=1024 -- see trap 3. This is the historical
#                   BD3-LM disease (ai_diary.md 2026-07-28, job 554446).
# LAUNCHER=spawn  : Lightning spawns the ranks itself. Requires defeating
#                   lightning's SLURM auto-detect, which triggers on SLURM_NTASKS
#                   being set with a job name outside {bash, interactive}; we
#                   unset it. This is the ONLY multi-GPU pattern ever verified to
#                   pass a live all_reduce in this env (job 489013). Launch with
#                   --ntasks-per-node=1 and --gres=gpu:N; trainer.devices comes
#                   from torch.cuda.device_count() so it still sees all N.
LAUNCHER="${LAUNCHER:-srun}"
NTASKS="${SLURM_NTASKS_PER_NODE:-1}"
if [ "${LAUNCHER}" = "spawn" ]; then
    unset SLURM_NTASKS SLURM_NTASKS_PER_NODE
    LAUNCH=()
elif [ "${NTASKS}" -gt 1 ]; then
    LAUNCH=(srun)
else
    LAUNCH=()
fi

DATA="ag_news"
# BLOCK_SZ (not BLOCK_SIZE -- avoids colliding with the hydra key name below).
# WARNING: configs/config.yaml:16 defines `block_size: ${model.length}`, i.e. it
# defaults to 1024, and NOTHING anywhere asserts that the hydra block_size equals
# the block size the checkpoint was trained with. On a mismatch the SDPA block
# attention mask is rebuilt at one stride while the KV cache is advanced at
# another: no error, no warning, just wrong numbers. So block_size MUST always be
# passed explicitly on the command line (the launch block below does), and it
# must track the checkpoint -- which is why CKPT interpolates BLOCK_SIZE.
BLOCK_SIZE="${BLOCK_SZ:-4}"
MODEL_LENGTH="${MODEL_LEN:-1024}"
CKPT="kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE}"

# Matches outputs/ag_news/2026.05.04/162506 (the run that produced 36.336).
# Override with EVAL_BS=64 to fit a 48 GB a6000. Every selection rule here is
# per-sequence, so batch size changes nothing but fp accumulation order.
EVAL_BATCH_SIZE="${EVAL_BS:-128}"
EXTRA=()

if [ "${SMOKE}" = "1" ]; then
    EVAL_BATCH_SIZE=1
    MODEL_LENGTH=64
    EXTRA+=(trainer.limit_val_batches=1)
    TAG="smoke_${MODE}"
else
    TAG="${MODE}${TAG_SUFFIX:-}"
fi

# Optional cap, PER RANK. To compare an N-GPU run against 1 GPU on the SAME
# samples, set LIMIT_VAL so that N * LIMIT_VAL * bs is equal in both runs --
# the DistributedSampler hands rank r the strided indices r::N, so N ranks with
# LIMIT_VAL=k jointly cover exactly the first N*k*bs samples, in a different
# order but the same set.
if [ -n "${LIMIT_VAL:-}" ]; then
    EXTRA+=(trainer.limit_val_batches=${LIMIT_VAL})
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
  oracle_dp)
    # Subset-lattice DP: same oracle/uniform-order/mixture columns as MODE=oracle,
    # 2^L'-1 forwards per block instead of L'!*L'. Required for L'=8.
    EXTRA+=(mode=duel_ppl
            eval.exact_ll_strategy=block_subset_dp
            eval.exact_ll_k=1
            +eval.exact_ll_use_kv_cache=true
            sampling.kv_cache=true
            model.attn_backend=sdpa)
    ;;
  l2r)
    EXTRA+=(mode=duel_ppl
            eval.exact_ll_strategy=block_left_to_right
            eval.exact_ll_k=1
            +eval.exact_ll_use_kv_cache=true
            sampling.kv_cache=true
            model.attn_backend=sdpa)
    ;;
  margin)
    EXTRA+=(mode=duel_ppl
            eval.exact_ll_strategy=block_probability_margin
            eval.exact_ll_k=1
            +eval.exact_ll_use_kv_cache=true
            sampling.kv_cache=true
            model.attn_backend=sdpa)
    ;;
  conf_thresh)
    # For this strategy exact_ll_k is the probability threshold tau, not a token
    # count. tau=0.95 is the value used in the paper
    # (duel-manuscript/appendix/experiments.tex:173).
    EXTRA+=(mode=duel_ppl
            eval.exact_ll_strategy=block_confidence_threshold
            eval.exact_ll_k=0.95
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
    echo "Unknown MODE=${MODE} (expected oracle|oracle_dp|greedy|l2r|margin|conf_thresh|elbo)" >&2
    exit 1
    ;;
esac

mkdir -p "${PWD}/logs" "${PWD}/watch_folder"
LOG="${PWD}/logs/bd3lm_${DATA}_block_size${BLOCK_SIZE}_decomp_${TAG}.log"

echo "MODE=${MODE} SMOKE=${SMOKE} bs=${EVAL_BATCH_SIZE} len=${MODEL_LENGTH} block=${BLOCK_SIZE} ckpt=${CKPT} ntasks=${NTASKS} -> ${LOG}"

${LAUNCH[@]+"${LAUNCH[@]}"} python -u main.py \
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
