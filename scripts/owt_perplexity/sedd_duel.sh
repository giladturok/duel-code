#!/bin/bash
#SBATCH -J sedd_duel_owt                # Job name
#SBATCH -o watch_folder/%x_%j.out       # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err       # log file (out & err)
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH --get-user-env                  # retrieve the users login environment
#SBATCH --mem=32G                       # server memory requested (per node)
#SBATCH -t 960:00:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                 # Request partition
#SBATCH --constraint="[a100|h100|h100|h200]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                    # Type/number of GPUs needed
#SBATCH --open-mode=append              # Do not overwrite logs
#SBATCH --requeue                       # Requeue upon preemption


export TRITON_NUM_STAGES=1

python -u main.py \
    loader.num_workers=4 \
    loader.eval_batch_size=64 \
    model=small \
    algo=sedd \
    algo.ignore_bos=false \
    data=openwebtext-split \
    model.length=1024 \
    block_size=4 \
    eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diffusion-exp-v4-nBm2gE-small-param-sedd_data-openwebtext-split_seqlen-1024_maxs-1300001_bs-512/checkpoints/last.ckpt \
    eval.exact_ll_strategy=block_greedy \
    eval.exact_ll_k=1 \
    wandb.project=duel +wandb.name=$SLURM_JOB_NAME \
    mode=duel_ppl > $PWD/logs/sedd_owt_duel.log
