#!/bin/bash
#SBATCH -J elbo_seed                    # Job name
#SBATCH -o watch_folder/%x_%A_%a.out    # log file (out & err)
#SBATCH -e watch_folder/%x_%A_%a.err
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="[a6000|a100|h100|h200|6000ada]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --array=1-16

# Measure the Monte Carlo standard error of the BD3-LM ELBO on AG News.
#
# WHY: the rebuttal (comment_r3.md) reads the 62.21 -> 60.72 move from ELBO to
# the exact uniform-order average as "removing the ELBO's Monte Carlo error".
# But the CT-NELBO with the linear schedule is *identically equal* to
# E_{sigma~Unif}[-log p(x|sigma)] (see ai_diary), so the MC estimator is
# UNBIASED for the enumerated number: its error is zero-mean, and the sign of
# any single-draw gap is arbitrary. This sweep measures the size of that error.
#
# Single-GPU MODE=elbo config, byte-identical to
# scripts/sampler_comparison/oracle_decomposition.sh MODE=elbo, except `seed`.
# `seed` drives both L.seed_everything and the valid dataloader generator
# (main.py:136,141). AG News val is exactly 384 sequences, and the whole set is
# consumed every run, so the seed reshuffles the ORDER but never changes the
# token set -- the sweep varies only the t / mask draw.
#
#   sbatch scripts/sampler_comparison/elbo_seed_sweep.sh
#   python scripts/sampler_comparison/elbo_seed_sweep_parse.py

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate bd3lm

export TRITON_NUM_STAGES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="${SLURM_ARRAY_TASK_ID:-1}"
BLOCK_SIZE="${BLOCK_SZ:-4}"
# IGNORE_BOS=False makes the ELBO score all 1024 positions (denominator 1024),
# which is the ONLY setting in which it covers the same token set as the
# enumeration path: at the default True the ELBO scores 1..1023 while
# exact_likelihood.py:244-254 builds a PREFIX mask from lengths=1023 and so
# scores 0..1022 (masking BOS, and handing token 1023 to the last block as free
# clean context). Both differences push the exact number down.
IGNORE_BOS="${IGNORE_BOS:-True}"
TAG="${TAG:-bos${IGNORE_BOS}}"
DATA="${DATA:-ag_news}"
ALGO="${ALGO:-bd3lm}"
# MDLM leaves block_size = model.length (configs/config.yaml:16), so _sample_t
# draws ONE t per sequence instead of L/L' -- a far noisier ELBO. That is the
# reason to sweep it: ALGO=mdlm DATA=lambada CKPT=kuleshov-group/mdlm-owt.
if [ "${ALGO}" = "mdlm" ]; then
    CKPT="${CKPT:-kuleshov-group/mdlm-owt}"
    BLOCK_ARGS=()
else
    CKPT="${CKPT:-kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE}}"
    BLOCK_ARGS=(block_size=${BLOCK_SIZE} model.attn_backend=flex)
fi

mkdir -p "${PWD}/logs/elbo_seed" "${PWD}/watch_folder"
LOG="${PWD}/logs/elbo_seed/${ALGO}_${DATA}_block${BLOCK_SIZE}_elbo_${TAG}_seed${SEED}.log"

python -u main.py \
    loader.eval_batch_size=16 \
    model=small \
    algo=${ALGO} \
    algo.backbone=hf_dit \
    data=${DATA} \
    +data.insert_valid_eos=False \
    model.length=1024 \
    "${BLOCK_ARGS[@]}" \
    eval.checkpoint_path=${CKPT} \
    seed=${SEED} \
    algo.ignore_bos=${IGNORE_BOS} \
    mode=elbo_ppl \
    wandb.project=duel \
    +wandb.name="elbo_${ALGO}_${TAG}_seed${SEED}_${DATA}" \
    2>&1 | tee "${LOG}"
