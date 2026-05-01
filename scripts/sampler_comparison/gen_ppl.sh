#!/bin/bash
#SBATCH -J sample_eval_sampler          # Job name
#SBATCH -o watch_folder/%x_%j.out       # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err       # log file (out & err)
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH --get-user-env                  # retrieve the users login environment
#SBATCH --mem=32G                       # server memory requested (per node)
#SBATCH -t 960:00:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                 # Request partition
#SBATCH --exclude=sun-compute-03,snavely-compute-02,portal-compute-01,kuleshov-compute-02,lancer-compute-01
#SBATCH --constraint="[a6000|a100|h100|h200]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # Type/number of GPUs needed
#SBATCH --open-mode=append              # Do not overwrite logs
#SBATCH --requeue                       # Requeue upon preemption

# Sampler comparison: sample_eval on OWT for all sampling strategies.
# Each call to mode=sample_eval generates samples and computes, in one pass:
#   - gen PPL (via GPT2-Large sliding window)
#   - token entropy
#   - MAUVE score (vs. OpenWebText reference)
# Reproduces the generative metrics in Figure 5 (appendix) and Figure 4 (bottom).
# Model: BD3-LM L'=16 on openwebtext-split.

MODEL_LENGTH=1024
BLOCK_SIZE=16
SEED=2
NUM_SAMPLE_BATCHES=1000
NUCLEUS_P=0.9
CKPT="kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE}"

SAMPLE_LOGDIR="${PWD}/sample_logs"
mkdir -p "${SAMPLE_LOGDIR}"

# Integer-k strategies: sweep k ∈ {1, 2, 4, 8}
K_VALUES=(1 2 4 8)
INT_K_STRATEGIES=(block_greedy block_left_to_right block_probability_margin)

# Confidence threshold strategy: targets gen-NFE {128, 256, 512, 1024}.
# Gen-PPL thresholds differ from DUEL-PPL because Gen-NFE is averaged over
# sampled text, not the val set.
#
# Derivation (two rounds of interpolation):
#   Round 1 (k 0.05/0.07/0.15/0.99 -> NFE 148/235/488/924):
#     chose 0.035/0.08/0.18/1.0 targeting 128/256/512/1024
#   Round 2 (k 0.035/0.08/0.18/1.0 -> NFE 93/270/565/921):
#     0.035->93 (27% low) -> bump to 0.045; interp: 0.035+(34.9/54.6)*0.015
#     0.08->270 (5.5% high) -> keep
#     0.18->565 (10.4% high) -> lower to 0.16; interp: 0.15+(23.6/76.7)*0.03
#     1.0->921 -> keep; ~921 is the generation-mode ceiling (sequences peak at
#       1023 tokens, not 1024, so NFE ceiling < 1024 regardless of k)
#
# Before re-running changed thresholds (0.045, 0.16), delete their CSVs first
# to avoid the append-mode duplication bug. Keep k=0.08 and k=1.0 CSVs.
THRESHOLDS=(0.045 0.08 0.16 1.0)

for strategy in "${INT_K_STRATEGIES[@]}"; do
    for k in "${K_VALUES[@]}"; do
        slug="${strategy//_/-}"
        logdir="${SAMPLE_LOGDIR}/samples_bd3lm_len${MODEL_LENGTH}_blocksize${BLOCK_SIZE}_${slug}_k${k}.txt"

        python -u main.py \
            loader.eval_batch_size=1 \
            model=small \
            algo=bd3lm \
            algo.T=5000 \
            algo.backbone=hf_dit \
            data=openwebtext-split \
            model.length=${MODEL_LENGTH} \
            block_size=${BLOCK_SIZE} \
            wandb.project=duel +wandb.name="${SLURM_JOB_NAME}_${strategy}_k${k}" \
            mode=sample_eval \
            eval.checkpoint_path=${CKPT} \
            model.attn_backend=sdpa \
            seed=${SEED} \
            sampling.num_sample_batches=${NUM_SAMPLE_BATCHES} \
            sampling.nucleus_p=${NUCLEUS_P} \
            sampling.kv_cache=true \
            +sampling.strategy=${strategy} \
            +sampling.strategy_k=${k} \
            sampling.logdir=${logdir}
    done
done

# Confidence threshold strategy: sweep thresholds chosen to target gen-NFE {128, 256, 512, 1024}.
strategy=block_confidence_threshold
slug="${strategy//_/-}"
for k in "${THRESHOLDS[@]}"; do
    logdir="${SAMPLE_LOGDIR}/samples_bd3lm_len${MODEL_LENGTH}_blocksize${BLOCK_SIZE}_${slug}_k${k}.txt"

    python -u main.py \
        loader.eval_batch_size=1 \
        model=small \
        algo=bd3lm \
        algo.T=5000 \
        algo.backbone=hf_dit \
        data=openwebtext-split \
        model.length=${MODEL_LENGTH} \
        block_size=${BLOCK_SIZE} \
        wandb.project=duel +wandb.name="${SLURM_JOB_NAME}_${strategy}_k${k}" \
        mode=sample_eval \
        eval.checkpoint_path=${CKPT} \
        model.attn_backend=sdpa \
        seed=${SEED} \
        sampling.num_sample_batches=${NUM_SAMPLE_BATCHES} \
        sampling.nucleus_p=${NUCLEUS_P} \
        sampling.kv_cache=true \
        +sampling.strategy=${strategy} \
        +sampling.strategy_k=${k} \
        sampling.logdir=${logdir}
done

# Uniform baseline (no strategy/strategy_k flags)
UNIFORM_LOGDIR="${SAMPLE_LOGDIR}/samples_bd3lm_len${MODEL_LENGTH}_blocksize${BLOCK_SIZE}_uniform.txt"
python -u main.py \
    loader.eval_batch_size=1 \
    model=small \
    algo=bd3lm \
    algo.T=5000 \
    algo.backbone=hf_dit \
    data=openwebtext-split \
    model.length=${MODEL_LENGTH} \
    block_size=${BLOCK_SIZE} \
    wandb.project=duel +wandb.name="${SLURM_JOB_NAME}_uniform" \
    mode=sample_eval \
    eval.checkpoint_path=${CKPT} \
    model.attn_backend=sdpa \
    seed=${SEED} \
    sampling.num_sample_batches=${NUM_SAMPLE_BATCHES} \
    sampling.nucleus_p=${NUCLEUS_P} \
    sampling.kv_cache=true \
    sampling.logdir=${UNIFORM_LOGDIR}
