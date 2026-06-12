#!/bin/bash
#SBATCH -J oracle_unmask                # Job name
#SBATCH -o watch_folder/%x_%j.out       # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err       # log file (out & err)
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH --get-user-env                  # retrieve the users login environment
#SBATCH --mem=64G                       # server memory requested (per node)
#SBATCH -t 960:00:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                 # Request partition
#SBATCH --exclude=sun-compute-03,snavely-compute-02,portal-compute-01,kuleshov-compute-02,lancer-compute-01,sun-compute-01
#SBATCH --constraint="[a6000|a100|h100|h200]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1                    # Type/number of GPUs needed
#SBATCH --open-mode=append              # Do not overwrite logs
#SBATCH --requeue                       # Requeue upon preemption

# Reproduces Table 7 (oracle unmasking) from arxiv 2603.01367.
# Dataset: AG News. Model: BD3-LM L'=4 (block_size=4). Metric: DUEL perplexity.
# Paper numbers (BD3-LM rows, AG News, lower is better):
#   ELBO            <= 61.67
#   Left-to-right     54.94
#   Greedy conf       56.73
#   Prob. margin      57.80
#   Conf. threshold   55.68
#   Oracle            36.47   <-- this script
#   ARM baseline      52.11
#
# Oracle algorithm: for each block of 4 tokens, exhaustively enumerate all
# 4! = 24 unmask-order permutations, evaluate DUEL log-likelihood under each,
# keep the permutation minimizing NLL. Strategy name `block_permutation`
# matches the LLaDA version under large_scale/exact_likelihood/.
#
# Implementation: selection_strategies.py:BlockPermutationStrategy (marker)
# dispatches diffusion._compute_exact_ll into
# exact_likelihood.compute_exact_loglikelihood_cached_permutations.

export TRITON_NUM_STAGES=1

DATA="ag_news"
BLOCK_SIZE=4
EVAL_BATCH_SIZE=16     # oracle = 24 permutations/block; bs=1 keeps memory + model-call accounting clean
MODEL_LENGTH=1024
CKPT="kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE}"

mkdir -p "${PWD}/logs"

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
    eval.exact_ll_strategy=block_permutation \
    eval.exact_ll_k=1 \
    +eval.exact_ll_use_kv_cache=true \
    sampling.kv_cache=true \
    wandb.project=duel +wandb.name="${SLURM_JOB_NAME}_${DATA}_block_size${BLOCK_SIZE}" \
    mode=duel_ppl \
    model.attn_backend=sdpa > ${PWD}/logs/bd3lm_${DATA}_block_size${BLOCK_SIZE}_exact_ll_block_permutation.log
