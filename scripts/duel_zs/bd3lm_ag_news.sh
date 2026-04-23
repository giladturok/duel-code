#!/bin/bash
#SBATCH -J duel_zs_bd3lm_ag_news
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=32G
#SBATCH -t 960:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="[a5000|a6000|a100]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --requeue

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bd3lm

export TRITON_NUM_STAGES=1

BLOCK_SIZE=16
EVAL_BATCH_SIZE=16

python -u main.py \
    loader.num_workers=4 \
    loader.eval_batch_size=${EVAL_BATCH_SIZE} \
    model=small \
    algo=bd3lm \
    algo.backbone=hf_dit \
    data=ag_news \
    +data.insert_valid_eos=False \
    model.length=1024 \
    block_size=${BLOCK_SIZE} \
    eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
    eval.exact_ll_strategy=block_greedy \
    eval.exact_ll_k=1 \
    +eval.exact_ll_use_kv_cache=true \
    sampling.kv_cache=true \
    wandb.project=duel +wandb.name=$SLURM_JOB_NAME \
    mode=duel_ppl \
    model.attn_backend=sdpa > $PWD/logs/bd3lm_ag_news_block_size${BLOCK_SIZE}_exact_ll.log
