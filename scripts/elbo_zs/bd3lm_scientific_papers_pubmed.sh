#!/bin/bash
#SBATCH -J elbo_zs_bd3lm_scientific_papers_pubmed
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=32G
#SBATCH -t 96:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="[a5000|a6000|3090]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --requeue

BLOCK_SIZE=4

srun python -u main.py \
    loader.num_workers=4 \
    loader.eval_batch_size=16 \
    model=small \
    algo=bd3lm \
    algo.backbone=hf_dit \
    data=scientific_papers_pubmed \
    +data.insert_valid_eos=False \
    model.length=1024 \
    block_size=${BLOCK_SIZE} \
    eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
    wandb.project=duel +wandb.name=$SLURM_JOB_NAME \
    mode=elbo_ppl \
    model.attn_backend=flex > $PWD/logs/bd3lm_scientific_papers_pubmed_block_size${BLOCK_SIZE}.log
