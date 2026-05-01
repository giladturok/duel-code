#!/bin/bash
#SBATCH -J duel_ppl_sampler             # Job name
#SBATCH -o watch_folder/%x_%j.out       # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err       # log file (out & err)
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH --get-user-env                  # retrieve the users login environment
#SBATCH --mem=64G                       # server memory requested (per node)
#SBATCH -t 960:00:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                 # Request partition
#SBATCH --exclude=sun-compute-03,snavely-compute-02,portal-compute-01,kuleshov-compute-02,lancer-compute-01,sun-compute-01
#SBATCH --constraint="[a6000|a100|h100|h200]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1                    # Type/number of GPUs needed
#SBATCH --open-mode=append              # Do not overwrite logs
#SBATCH --requeue                       # Requeue upon preemption

# Sampler comparison: DUEL PPL on OWT for all unmasking rules.
# Reproduces the DUEL-PPL metric in Figure 5 (appendix), Figure 4 (top), Table 6.
# Model: BD3-LM L'=16 on openwebtext-split (valid=openwebtext-valid-1k).

export TRITON_NUM_STAGES=1

DATA="openwebtext-split"
BLOCK_SIZE=16
EVAL_BATCH_SIZE=64
EVAL_BATCH_SIZE_THRESH=1  # threshold strategy: bs=1 to accurately count model calls
MODEL_LENGTH=1024
CKPT="kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE}"

mkdir -p "${PWD}/logs"

# Integer-k strategies: sweep k ∈ {1, 2, 4, 8}
K_VALUES=(1 2 4 8)
INT_K_STRATEGIES=(block_greedy block_left_to_right block_probability_margin)

# Confidence threshold strategy: thresholds chosen to reproduce Table 6 / Fig 4
# on OWT (NFE 128/256/512/1024 -> k 0.05/0.07/0.15/0.99, matched via wandb
# BD3-LMs project runs against paper perplexities 226.97/116.83/43.48/22.05).
THRESHOLDS=(0.05 0.07 0.15 0.99)

run_exact_ll() {
    local strategy=$1
    local k=$2
    local batch_size=$3
    python -u main.py \
        loader.eval_batch_size=${batch_size} \
        model=small \
        algo=bd3lm \
        algo.backbone=hf_dit \
        data=${DATA} \
        data.valid=openwebtext-valid-1k \
        data.insert_valid_eos=False \
        model.length=${MODEL_LENGTH} \
        block_size=${BLOCK_SIZE} \
        eval.checkpoint_path=${CKPT} \
        eval.exact_ll_strategy=${strategy} \
        eval.exact_ll_k=${k} \
        +eval.exact_ll_use_kv_cache=true \
        sampling.kv_cache=true \
        wandb.project=duel +wandb.name="${SLURM_JOB_NAME}_${strategy}_k${k}" \
        mode=duel_ppl \
        model.attn_backend=sdpa > ${PWD}/logs/bd3lm_${DATA}_block_size${BLOCK_SIZE}_exact_ll_${strategy}_${k}.log
}

for strategy in "${INT_K_STRATEGIES[@]}"; do
    for k in "${K_VALUES[@]}"; do
        run_exact_ll "${strategy}" "${k}" "${EVAL_BATCH_SIZE}"
    done
done

for t in "${THRESHOLDS[@]}"; do
    run_exact_ll "block_confidence_threshold" "${t}" "${EVAL_BATCH_SIZE_THRESH}"
done

# ELBO baseline (uses flex attention backend)
python -u main.py \
    loader.eval_batch_size=16 \
    model=small \
    algo=bd3lm \
    algo.backbone=hf_dit \
    data=${DATA} \
    data.valid=openwebtext-valid-1k \
    data.insert_valid_eos=False \
    model.length=${MODEL_LENGTH} \
    block_size=${BLOCK_SIZE} \
    eval.checkpoint_path=${CKPT} \
    wandb.project=duel +wandb.name="${SLURM_JOB_NAME}_elbo_baseline" \
    mode=elbo_ppl \
    model.attn_backend=flex > ${PWD}/logs/bd3lm_${DATA}_block_size${BLOCK_SIZE}_elbo.log
