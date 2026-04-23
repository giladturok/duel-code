#!/bin/bash
#SBATCH -J elbo_sedd_owt                # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu          # Request partition
#SBATCH --constraint="[a5000|a6000|a100]"
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
    algo.ignore_bos=false \
    data=openwebtext-split \
    model.length=1024 \
    eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diffusion-exp-v4-nBm2gE-small-param-sedd_data-openwebtext-split_seqlen-1024_maxs-1300001_bs-512/checkpoints/last.ckpt \
    wandb.project=duel +wandb.name=$SLURM_JOB_NAME \
    mode=elbo_ppl > $PWD/logs/sedd_owt.log