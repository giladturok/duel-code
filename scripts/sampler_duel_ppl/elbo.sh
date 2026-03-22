#!/bin/bash
#SBATCH -J ppl_samplers_ag-news_bd3lm_elbo              # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -e watch_folder/%x_%j.err     # error log file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 96:00:00                  # Time limit (hh:mm:ss)
#SBATCH --exclude=sun-compute-03,snavely-compute-02,portal-compute-01,kuleshov-compute-02,lancer-compute-01,sun-compute-01 # exclude a problematic node
#SBATCH --partition=gpu         # Request partition
#SBATCH --constraint="[a5000|a6000|a100|h100|h200]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                 # Type/number of GPUs needed
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

data="openwebtext-split"
BLOCK_SIZE=16

echo "$data"
python -u main.py \
    loader.eval_batch_size=16 \
    model=small \
    algo.backbone=hf_dit \
    algo=bd3lm \
    data=$data \
    data.valid=openwebtext-valid-1k \
    data.insert_valid_eos=False \
    model.length=1024 \
    block_size=${BLOCK_SIZE} \
    eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
    wandb.project=duel wandb.name=bd3lm-${data}-block_size${BLOCK_SIZE}_elbo \
    mode=elbo_ppl \
    model.attn_backend=flex > $PWD/logs/bd3lm_${data}_block_size${BLOCK_SIZE}.log
