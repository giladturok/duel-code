#!/bin/bash
#SBATCH -J debug_nosrun
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=32G
#SBATCH -t 00:10:00
#SBATCH --partition=gpu
#SBATCH --constraint="[a5000|a6000|3090|a100]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4

# Disable Lightning's SLURM detection so it uses its internal
# subprocess launcher to manage multi-GPU (like training scripts do)
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE

python -u main.py \
    loader.num_workers=4 \
    loader.eval_batch_size=128 \
    model=small \
    algo=sedd \
    data=lm1b-wrap \
    model.length=128 \
    eval.checkpoint_path=/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-lm1b-sedd128-v2/checkpoints/last.ckpt \
    wandb.project=duel +wandb.name=debug-nosrun \
    mode=elbo_ppl > $PWD/logs/debug_nosrun.log
