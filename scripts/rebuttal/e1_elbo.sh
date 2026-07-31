#!/bin/bash
#SBATCH -J e1_elbo                      # Job name
#SBATCH -o watch_folder/%x_%j.out       # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err       # log file (out & err)
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="[a6000]"
#SBATCH --exclude=snavely-compute-02,abdelfattah-compute-02,seo-compute-02,davis-compute-01
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --requeue
#
# E1 / R2 W2-Q1: ELBO cost arms at 110M on OWT.
#
#   ALGO=bd3lm BLOCK=16 SEEDS="1 2 3 4" sbatch scripts/rebuttal/e1_elbo.sh
#   ALGO=mdlm  BLOCK=1024 SEEDS=1 BACKEND=flex sbatch scripts/rebuttal/e1_elbo.sh
#
# "MC=K" is K INDEPENDENT SEEDS = K full validation passes at exactly K x cost.
# diffusion._sample_t:1416-1437 draws _eps_b over (dataloader batch, num_blocks);
# the batch dimension is DATA-parallel, not MC-parallel, so there is no batching
# trick at 110M. Averaging is done offline in NLL space over the SAME full split,
# which is why these arms run the full split rather than a few batches.
#
# MDLM must use BLOCK=1024 (= model.length): _sample_t then yields exactly ONE
# t-draw per sequence per forward, which is the MDLM ELBO as published.

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate bd3lm

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/rebuttal/_e1_common.sh

ALGO="${ALGO:-bd3lm}"
BLOCK="${BLOCK:-16}"
# REPS = independent full evaluations. Each is one MC draw of t per sequence
# (per block for BD3-LM), on the SAME sequences -- seed=null fixes the data
# order but leaves the torch seed fresh. "MC=K" is the mean of the first K.
REPS="${REPS:-1}"
SEEDS="${SEEDS:-$(seq 1 "${REPS}")}"
EVAL_BS="${EVAL_BS:-32}"
LIMIT="${LIMIT:-8}"
BACKEND="${BACKEND:-sdpa}"
TAG="${TAG:-}"

e1_calibrate

mapfile -t COMMON < <(e1_common_args "${ALGO}" "${BLOCK}" "${EVAL_BS}")

BACKENDS="${BACKENDS:-${BACKEND}}"

# seed 1 / sdpa first: that arm carries the reproduction gate, so a preempted
# job still answers the question that matters most.
for BK in ${BACKENDS}; do
  for S in ${SEEDS}; do
    ARM="elbo__${ALGO}${BLOCK}__rep${S}__bs${EVAL_BS}__${BK}${TAG}"
    if [ -s "${E1_ROOT}/${ARM}/run.log" ] && \
       grep -q "^exit_code=0$" "${E1_ROOT}/${ARM}/meta.txt" 2>/dev/null; then
        echo "SKIP ${ARM} (already complete)"; continue
    fi
    # PL_GLOBAL_SEED is the MC knob (see _e1_common.sh:e1_common_args). Same
    # sequences every rep, independent t -- so the spread across reps is pure
    # estimator noise, not data noise.
    export PL_GLOBAL_SEED="${S}"
    run_arm "${ARM}" \
        "${COMMON[@]}" \
        mode=elbo_ppl \
        model.attn_backend="${BK}" \
        trainer.limit_val_batches="${LIMIT}"
    # flex is a best-case reference for the ELBO only; one seed is enough.
    [ "${BK}" = "sdpa" ] || break
  done
done

echo "ALL ARMS DONE for ALGO=${ALGO} BLOCK=${BLOCK} SEEDS='${SEEDS}' BACKENDS='${BACKENDS}'"
