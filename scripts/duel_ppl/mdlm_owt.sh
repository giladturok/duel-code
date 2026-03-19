#!/bin/bash
#SBATCH -J exact_ppl_mdlm_owt      # Job name
#SBATCH -o watch_folder/%x_%j.out   # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err   # log file (out & err)
#SBATCH --exclude snavely-compute-02
#SBATCH -N 1                        # Total number of nodes requested
#SBATCH --get-user-env              # retrieve the users login environment
#SBATCH --mem=32G                   # server memory requested (per node)
#SBATCH -t 960:00:00                # Time limit (hh:mm:ss)
#SBATCH --partition=gpu             # Request partition
#SBATCH --constraint="[h100|h200]"
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                # Type/number of GPUs needed
#SBATCH --open-mode=append          # Do not overwrite logs
#SBATCH --requeue                   # Requeue upon preemption

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dllm

export TRITON_NUM_STAGES=1
BLOCK_SIZE=4

srun python -u main.py \
    loader.num_workers=4 \
    loader.eval_batch_size=64 \
    model=small \
    algo=mdlm \
    algo.backbone=hf_dit \
    algo.ignore_bos=false \
    data=openwebtext-split \
    model.length=1024 \
    block_size=${BLOCK_SIZE} \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    eval.exact_ll_strategy=block_greedy \
    eval.exact_ll_k=1 \
    wandb.project=duel wandb.name=mdlm-owt-block_size${BLOCK_SIZE}-exact_ll \
    mode=duel_ppl > $PWD/logs/mdlm_owt_block_size${BLOCK_SIZE}-exact_ll.log