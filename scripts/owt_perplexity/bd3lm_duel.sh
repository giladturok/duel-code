#!/bin/bash
#SBATCH -J bd3lm_duel_owt               # Job name
#SBATCH -o watch_folder/%x_%j.out       # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err       # log file (out & err)
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH --get-user-env                  # retrieve the users login environment
#SBATCH --mem=32G                       # server memory requested (per node)
#SBATCH -t 960:00:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                 # Request partition
#SBATCH --constraint="[a100|h200|h100]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                    # Type/number of GPUs needed
#SBATCH --open-mode=append              # Do not overwrite logs
#SBATCH --requeue                       # Requeue upon preemption


export TRITON_NUM_STAGES=1

# Loop over block sizes for BD3-LM DUEL evaluation
for BLOCK_SIZE in 4 8 16; do
    python -u main.py \
        loader.num_workers=4 \
        loader.eval_batch_size=64 \
        model=small \
        algo=bd3lm \
        algo.backbone=hf_dit \
        data=openwebtext-split \
        data.insert_valid_special=False \
        model.length=1024 \
        model.attn_backend=sdpa \
        block_size=${BLOCK_SIZE} \
        eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
        eval.exact_ll_strategy=block_greedy \
        eval.exact_ll_k=1 \
        +eval.exact_ll_use_kv_cache=true \
        sampling.kv_cache=true \
        wandb.project=duel +wandb.name="${SLURM_JOB_NAME}_block${BLOCK_SIZE}" \
        mode=duel_ppl > $PWD/logs/bd3lm_owt_duel_block_size${BLOCK_SIZE}.log
done
