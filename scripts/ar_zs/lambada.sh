#!/bin/bash
#SBATCH -J ar_zs_lambada
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=32G
#SBATCH -t 96:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="[a5000|a6000|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --requeue

srun python -u main.py \
    loader.num_workers=4 \
    loader.eval_batch_size=16 \
    model=small \
    algo=ar \
    data=lambada \
    +data.insert_valid_eos=False \
    model.length=1024 \
    eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diffusion-exp-v4-AgBZrc-small-ar-param-ar_data-openwebtext-split_seqlen-1024_maxs-1300001_bs-512/checkpoints/last.ckpt \
    wandb.project=duel +wandb.name=$SLURM_JOB_NAME \
    mode=elbo_ppl > $PWD/logs/ar_lambada.log
