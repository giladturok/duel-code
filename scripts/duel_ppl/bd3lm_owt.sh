#!/bin/bash
#SBATCH -J exact_ppl_bd3lm_owt      # Job name
#SBATCH -o watch_folder/%x_%j.out   # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err   # log file (out & err)
#SBATCH -N 1                        # Total number of nodes requested
#SBATCH --get-user-env              # retrieve the users login environment
#SBATCH --mem=32G                   # server memory requested (per node)
#SBATCH -t 960:00:00                # Time limit (hh:mm:ss)
#SBATCH --partition=gpu             # Request partition
#SBATCH --constraint="[a100|h200|h100]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                # Type/number of GPUs needed
#SBATCH --open-mode=append          # Do not overwrite logs
#SBATCH --requeue                   # Requeue upon preemption

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bd3lm

export TRITON_NUM_STAGES=1
BLOCK_SIZE=16

srun python -u main.py \
    loader.num_workers=4 \
    loader.eval_batch_size=64 \
    model=small \
    algo=bd3lm \
    algo.backbone=hf_dit \
    data=openwebtext-split \
    data.insert_valid_special=False \
    model.length=1024 \
    model.attn_backend=sdpa \
    block_size=${BLOCK_SIZE} \
    eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
    eval.exact_ll_strategy=block_greedy \
    eval.exact_ll_k=1 \
    +eval.exact_ll_use_kv_cache=true \
    sampling.kv_cache=true \
    wandb.project=duel wandb.name=bd3lm-owt-block_size${BLOCK_SIZE}_exact_ll \
    mode=duel_ppl > $PWD/logs/bd3lm_owt_block_size${BLOCK_SIZE}_exact_ll.log
