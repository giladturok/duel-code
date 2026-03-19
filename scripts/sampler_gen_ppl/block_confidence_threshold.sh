#!/bin/bash
#SBATCH -J gen_ppl_bd3lm_block_confidence_threshold                # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu          # Request partition
#SBATCH --exclude=sun-compute-03,snavely-compute-02,portal-compute-01,kuleshov-compute-02,lancer-compute-01 # exclude a problematic node
#SBATCH --constraint="[a6000|a100|h100|h200]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

LENGTH=1024
SEED=2
BLOCK_SIZE=16
sampling_params=(0.03 0.04 0.05 0.1 0.5 0.7 0.8 0.98 0.99)
sampling_params=0.03

for k in "${sampling_params[@]}"; do
    echo "Generating samples with block_confidence_threshold k=$k"
    srun python -u main.py \
        loader.eval_batch_size=1 \
        model=small \
        algo=bd3lm \
        algo.T=5000 \
        algo.backbone=hf_dit \
        data=openwebtext-split \
        model.length=$LENGTH \
        block_size=$BLOCK_SIZE \
        wandb.project=duel +wandb.name=sampler-genppl-block-conf-thresh \
        mode=sample_eval \
        eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
        model.attn_backend=sdpa \
        seed=$SEED \
        sampling.num_sample_batches=1000 \
        sampling.nucleus_p=0.9 \
        sampling.kv_cache=true \
        +sampling.strategy=block_confidence_threshold \
        +sampling.strategy_k=$k \
        sampling.logdir=$PWD/sample_logs/samples_bd3lm_len${LENGTH}_blocksize${BLOCK_SIZE}_block-confidence-threshold_k${k}.txt
done