#!/bin/bash
#SBATCH -J exact_bosF                   # Job name
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH --partition=gpu
#SBATCH --exclude=sun-compute-03,snavely-compute-02,portal-compute-01,kuleshov-compute-02,lancer-compute-01,sun-compute-01,snavely-compute-04,snavely-compute-11,snavely-compute-12,sablab-gpu-04
#SBATCH --constraint="[a6000|a100|h100|h200|6000ada]"
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --open-mode=append
#SBATCH --requeue

# Exact uniform-order / mixture / oracle on AG News L'=4 with algo.ignore_bos=False.
#
# WHY: at the default ignore_bos=True the ELBO row and the enumeration row do NOT
# score the same tokens. The ELBO holds BOS clean and zeroes its loss
# (diffusion.py:1516-1517, :1556-1557) => positions 1..1023. The enumeration
# rebuilds validity as a PREFIX from lengths=1023 (exact_likelihood.py:244-254)
# => positions 0..1022: BOS is masked and scored, and token 1023 is handed to the
# last block as free clean context. Both differences push the exact number DOWN,
# so part of the published 62.21 -> 60.72 "Monte Carlo error" is this off-by-one.
# ai_diary.md:1323 has this as an open TODO ("quantify the ignore_bos off-by-one").
#
# With ignore_bos=False both paths score all 1024 positions with denominator
# 1024 and no leaked context, so ELBO and uniform-order become directly
# comparable. Pair this with:
#   IGNORE_BOS=False TAG=bosFalse sbatch scripts/sampler_comparison/elbo_seed_sweep.sh
#
# Traps carried over from oracle_decomposition.sh: srun with ntasks==devices;
# N_gpus*bs must divide 384 (4*16=64 -> 6 batches); >=8 CPUs per rank or NCCL hangs.

set -euo pipefail

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate bd3lm

export TRITON_NUM_STAGES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BLOCK_SIZE="${BLOCK_SZ:-4}"
IGNORE_BOS="${IGNORE_BOS:-False}"
EVAL_BS="${EVAL_BS:-16}"
DATA="ag_news"
CKPT="kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE}"

mkdir -p "${PWD}/logs" "${PWD}/watch_folder"
LOG="${PWD}/logs/bd3lm_${DATA}_block${BLOCK_SIZE}_exact_bos${IGNORE_BOS}.log"

srun python -u main.py \
    loader.eval_batch_size=${EVAL_BS} \
    model=small \
    algo=bd3lm \
    algo.backbone=hf_dit \
    data=${DATA} \
    +data.insert_valid_eos=False \
    model.length=1024 \
    block_size=${BLOCK_SIZE} \
    eval.checkpoint_path=${CKPT} \
    algo.ignore_bos=${IGNORE_BOS} \
    mode=duel_ppl \
    eval.exact_ll_strategy=block_permutation \
    eval.exact_ll_k=1 \
    +eval.exact_ll_use_kv_cache=true \
    sampling.kv_cache=true \
    model.attn_backend=sdpa \
    wandb.project=duel \
    +wandb.name="exact_bos${IGNORE_BOS}_${DATA}_block_size${BLOCK_SIZE}" \
    2>&1 | tee "${LOG}"
