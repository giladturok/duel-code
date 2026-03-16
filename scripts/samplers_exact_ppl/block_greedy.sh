#!/bin/bash
#SBATCH -J exact_ppl_bd3lm_block_greedy    # Job name
#SBATCH -o watch_folder/%x_%j.out   # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err   # log file (out & err)
#SBATCH -N 1                        # Total number of nodes requested
#SBATCH --get-user-env              # retrieve the users login environment
#SBATCH --mem=64G                   # server memory requested (per node)
#SBATCH -t 96:00:00                 # Time limit (hh:mm:ss)
#SBATCH --exclude=sun-compute-03,snavely-compute-02,portal-compute-01,kuleshov-compute-02,lancer-compute-01,sun-compute-01 # exclude a problematic node
#SBATCH --partition=gpu             # Request partition
#SBATCH --constraint="[a6000|a100|h100|h200]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # Type/number of GPUs needed
#SBATCH --cpus-per-task=2           # Number of CPU cores per task
#SBATCH --open-mode=append          # Do not overwrite logs
#SBATCH --requeue                   # Requeue upon pre-emption

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bd3lm

export TRITON_NUM_STAGES=1

data="openwebtext-split"
BLOCK_SIZE=16
EVAL_BATCH_SIZE=64

sampling_params=(1 2 4 8)

for k in "${sampling_params[@]}"; do
    echo "$data"
    srun python -u main.py \
        loader.eval_batch_size=${EVAL_BATCH_SIZE} \
        model=small \
        algo=bd3lm \
        algo.backbone=hf_dit \
        data=$data \
        data.valid=openwebtext-valid-1k \
        data.insert_valid_eos=False \
        model.length=1024 \
        block_size=${BLOCK_SIZE} \
        eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
        eval.exact_ll_strategy=block_greedy \
        eval.exact_ll_k=$k \
        +eval.exact_ll_use_kv_cache=true \
        sampling.kv_cache=true \
        wandb.name=bd3lm-${data}-block_size${BLOCK_SIZE}_exact_ll_${k} \
        mode=exact_ppl \
        model.attn_backend=sdpa > $PWD/logs/bd3lm_${data}_block_size${BLOCK_SIZE}_exact_ll_${k}.log
done
