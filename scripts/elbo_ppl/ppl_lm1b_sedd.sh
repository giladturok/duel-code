#!/bin/bash
#SBATCH -J ppl_lm1b_sedd                # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu          # Request partition
#SBATCH --constraint="[a5000|a6000|3090|a100]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

# Debug/diagnostic exports
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900
export NCCL_SOCKET_IFNAME=^docker0,lo

python -u main.py \
    loader.num_workers=1 \
    loader.eval_batch_size=16 \
    model=small \
    algo=sedd \
    data=lm1b-wrap \
    model.length=128 \
    eval.checkpoint_path=/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-lm1b-sedd128-v2/checkpoints/last.ckpt \
    wandb.project=duel +wandb.name=elbo-lm1b_sedd \
    mode=elbo_ppl > $PWD/logs/sedd_lm1b_wrap.log
