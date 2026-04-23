#!/bin/bash
#SBATCH -J sedd_duel_zs                 # Job name
#SBATCH -o watch_folder/%x_%j.out       # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err       # log file (out & err)
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH --get-user-env                  # retrieve the users login environment
#SBATCH --mem=64G                       # server memory requested (per node)
#SBATCH -t 960:00:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                 # Request partition
#SBATCH --constraint="[a5000|a6000|a100]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                    # Type/number of GPUs needed
#SBATCH --open-mode=append              # Do not overwrite logs
#SBATCH --requeue                       # Requeue upon preemption

eval "$(conda shell.bash hook)"
conda activate bd3lm

export TRITON_NUM_STAGES=1

BLOCK_SIZE=4
EVAL_BATCH_SIZE=16

# Loop over 7 zero-shot datasets
for DATASET in ag_news lambada lm1b-gpt2 ptb scientific_papers_arxiv scientific_papers_pubmed wikitext103; do
    python -u main.py \
        loader.eval_batch_size=${EVAL_BATCH_SIZE} \
        model=small \
        algo=sedd \
        data=${DATASET} \
        +data.insert_valid_eos=False \
        model.length=1024 \
        block_size=${BLOCK_SIZE} \
        eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diffusion-exp-v4-nBm2gE-small-param-sedd_data-openwebtext-split_seqlen-1024_maxs-1300001_bs-512/checkpoints/last.ckpt \
        eval.exact_ll_strategy=block_greedy \
        eval.exact_ll_k=1 \
        wandb.project=duel +wandb.name="${SLURM_JOB_NAME}_${DATASET}" \
        mode=duel_ppl > $PWD/logs/sedd_${DATASET}_duel_block_size${BLOCK_SIZE}.log
done
