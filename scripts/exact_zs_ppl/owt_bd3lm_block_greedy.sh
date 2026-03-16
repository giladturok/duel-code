#!/bin/bash
#SBATCH -J exact_ppl_zs_owt_bd3lm    # Job name
#SBATCH -o watch_folder/%x_%j.out   # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err   # log file (out & err)
#SBATCH -N 1                        # Total number of nodes requested
#SBATCH --get-user-env              # retrieve the users login environment
#SBATCH --mem=32G                   # server memory requested (per node)
#SBATCH -t 96:00:00                 # Time limit (hh:mm:ss)
#SBATCH --partition=gpu             # Request partition
#SBATCH --constraint="[h200]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                # Type/number of GPUs needed
#SBATCH --cpus-per-task=1           # Number of CPU cores per task
#SBATCH --open-mode=append          # Do not overwrite logs
#SBATCH --requeue                   # Requeue upon pre-emption

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bd3lm

export TRITON_NUM_STAGES=1

datasets=(
    "ag_news"
    "lambada"
    "ptb"
    "wikitext2" # not used in BD3-LM paper
    "wikitext103" # used in BD3-LM paper
    "scientific_papers_pubmed"
    # "scientific_papers_arxiv"
    # "lm1b-gpt2"
)

BLOCK_SIZE=4
EVAL_BATCH_SIZE=128

for data in "${datasets[@]}"; do
    echo "$data"
    srun python -u main.py \
        loader.eval_batch_size=${EVAL_BATCH_SIZE} \
        model=small \
        algo=bd3lm \
        algo.backbone=hf_dit \
        data=$data \
        +data.insert_valid_eos=False \
        model.length=1024 \
        block_size=${BLOCK_SIZE} \
        eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
        eval.exact_ll_strategy=block_greedy \
        eval.exact_ll_k=1 \
        +eval.exact_ll_use_kv_cache=true \
        sampling.kv_cache=true \
        wandb.name=bd3lm-${data}-block_size${BLOCK_SIZE}_exact_ll \
        mode=exact_ppl \
        model.attn_backend=sdpa > $PWD/logs/bd3lm_${data}_block_size${BLOCK_SIZE}_exact_ll.log
done
