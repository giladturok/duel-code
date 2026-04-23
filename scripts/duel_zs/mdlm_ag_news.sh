#!/bin/bash
#SBATCH -J duel_zs_mdlm_ag_news
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64G
#SBATCH -t 150:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="[a5000|a6000|a100]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --requeue

eval "$(conda shell.bash hook)"
conda activate bd3lm

export TRITON_NUM_STAGES=1

BLOCK_SIZE=4
EVAL_BATCH_SIZE=16

python -u main.py \
    loader.eval_batch_size=${EVAL_BATCH_SIZE} \
    model=small \
    algo=mdlm \
    algo.backbone=hf_dit \
    data=ag_news \
    +data.insert_valid_eos=False \
    model.length=1024 \
    block_size=${BLOCK_SIZE} \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    eval.exact_ll_strategy=block_greedy \
    eval.exact_ll_k=1 \
    wandb.project=duel +wandb.name=$SLURM_JOB_NAME \
    mode=duel_ppl \
    model.attn_backend=sdpa > $PWD/logs/mdlm_ag_news_block_size${BLOCK_SIZE}_exact_ll.log
