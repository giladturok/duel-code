#!/bin/bash
#SBATCH -J exact_ppl_zs_owt_sedd    # Job name
#SBATCH -o watch_folder/%x_%j.out   # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err   # log file (out & err)
#SBATCH -N 1                        # Total number of nodes requested
#SBATCH --get-user-env              # retrieve the users login environment
#SBATCH --mem=64G                   # server memory requested (per node)
#SBATCH -t 150:00:00                 # Time limit (hh:mm:ss)
#SBATCH --exclude=sun-compute-03,snavely-compute-02,kuleshov-compute-02 # exclude a problematic node
#SBATCH --partition=gpu             # Request partition
#SBATCH --constraint="[a100|h100|h200|a6000]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # Type/number of GPUs needed
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --open-mode=append          # Do not overwrite logs
#SBATCH --requeue                   # Requeue upon pre-emption

eval "$(conda shell.bash hook)"
conda activate bd3lm

export TRITON_NUM_STAGES=1

datasets=(
    "ag_news"
    "lambada"
    "ptb"
    # "wikitext2" # not used in BD3-LM paper
    "wikitext103" # used in BD3-LM paper
    "scientific_papers_pubmed"
    "scientific_papers_arxiv"
    "lm1b-gpt2"
)

BLOCK_SIZE=4
EVAL_BATCH_SIZE=32

for data in "${datasets[@]}"; do
    echo "$data"
    srun python -u main.py \
        loader.eval_batch_size=${EVAL_BATCH_SIZE} \
        model=small \
        algo=sedd \
        data=${data} \
        +data.insert_valid_eos=False \
        model.length=1024 \
        block_size=${BLOCK_SIZE} \
        eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diffusion-exp-v4-nBm2gE-small-param-sedd_data-openwebtext-split_seqlen-1024_maxs-1300001_bs-512/checkpoints/last.ckpt \
        eval.exact_ll_strategy=block_greedy \
        eval.exact_ll_k=1 \
        wandb.name=sedd-${data}-block_size${BLOCK_SIZE}_exact_ll \
        mode=exact_ppl > $PWD/logs/sedd_${data}_block_size${BLOCK_SIZE}_exact_ll.log
done
