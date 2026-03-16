#!/bin/bash
#SBATCH -J exact_ppl_mdlm_lm1b      # Job name
#SBATCH -o watch_folder/%x_%j.out   # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err   # log file (out & err)
#SBATCH --exclude snavely-compute-02
#SBATCH -N 1                        # Total number of nodes requested
#SBATCH --get-user-env              # retrieve the users login environment
#SBATCH --mem=64G                   # server memory requested (per node)
#SBATCH -t 960:00:00                # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov             # Request partition
#SBATCH --constraint="[a5000|a6000]"
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                # Type/number of GPUs needed
#SBATCH --open-mode=append          # Do not overwrite logs
#SBATCH --requeue                   # Requeue upon preemption

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dllm

export TRITON_NUM_STAGES=1
BLOCK_SIZE=4

srun python -u main.py \
    loader.eval_batch_size=256 \
    model=small \
    algo=mdlm \
    data=lm1b-wrap \
    model.length=128 \
    block_size=${BLOCK_SIZE} \
    eval.checkpoint_path=/share/kuleshov/ma2238/textdiffusion/checkpoints/lm1b_wrap_pretrain/checkpoints/last_copy.ckpt \
    eval.exact_ll_strategy=block_greedy \
    eval.exact_ll_k=1 \
    wandb.name=mdlm-lm1b-block_size${BLOCK_SIZE}-exact_ll \
    mode=exact_ppl > $PWD/logs/mdlm_lm1b_block_size${BLOCK_SIZE}-exact_ll.log