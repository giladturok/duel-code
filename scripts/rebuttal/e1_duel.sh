#!/bin/bash
#SBATCH -J e1_duel                      # Job name
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
# E1 / R2 W2-Q1: DUEL held-out-likelihood cost arms at 110M on OWT.
#
#   ALGO=mdlm  BLOCK=4  SAMPLER=block_greedy KS="1 2 4"      sbatch scripts/rebuttal/e1_duel.sh
#   ALGO=bd3lm BLOCK=16 SAMPLER=block_greedy KS="1 2 4 8 16" sbatch scripts/rebuttal/e1_duel.sh
#
# MDLM takes the UNCACHED path (exact_likelihood.compute_exact_loglikelihood):
# diffusion.forward:583-590 only forwards store_kv/sample_mode to the backbone
# when algo.name == 'bd3lm', and kuleshov-group/mdlm-owt exposes no
# reset_kv_cache, so +eval.exact_ll_use_kv_cache=true would raise. Every MDLM
# DUEL forward is therefore a FULL L=1024 forward that recomputes all L/L'
# blocks to harvest k tokens. BD3-LM takes the cached path: block-length
# forwards against a committed KV prefix.

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate bd3lm

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/rebuttal/_e1_common.sh

ALGO="${ALGO:-bd3lm}"
BLOCK="${BLOCK:-16}"
# Plural: a6000s are scarce, so one job sweeps a whole table rather than
# queueing one job per arm.
SAMPLERS="${SAMPLERS:-${SAMPLER:-block_greedy}}"
KS="${KS:-1}"
EVAL_BS="${EVAL_BS:-32}"
LIMIT="${LIMIT:-8}"
BACKEND="${BACKEND:-sdpa}"
TAG="${TAG:-}"

unset PL_GLOBAL_SEED   # DUEL is deterministic; no MC knob here

e1_calibrate

mapfile -t COMMON < <(e1_common_args "${ALGO}" "${BLOCK}" "${EVAL_BS}")

KV=()
if [ "${ALGO}" = "bd3lm" ]; then
    KV=(+eval.exact_ll_use_kv_cache=true sampling.kv_cache=true)
fi

# k = L' (fully parallel within a block) first, then k = 1 (the paper's
# protocol). Cheapest-first so a preempted job still yields usable rows.
for SAMPLER in ${SAMPLERS}; do
  for K in ${KS}; do
    ARM="duel__${ALGO}${BLOCK}__${SAMPLER}__k${K}__bs${EVAL_BS}__${BACKEND}${TAG}"
    if [ -s "${E1_ROOT}/${ARM}/run.log" ] && \
       grep -q "^exit_code=0$" "${E1_ROOT}/${ARM}/meta.txt" 2>/dev/null; then
        echo "SKIP ${ARM} (already complete)"; continue
    fi
    run_arm "${ARM}" \
        "${COMMON[@]}" \
        mode=duel_ppl \
        eval.exact_ll_strategy="${SAMPLER}" \
        eval.exact_ll_k="${K}" \
        model.attn_backend="${BACKEND}" \
        trainer.limit_val_batches="${LIMIT}" \
        "${KV[@]+"${KV[@]}"}"
  done
done

echo "ALL ARMS DONE for ALGO=${ALGO} BLOCK=${BLOCK} SAMPLERS='${SAMPLERS}' KS='${KS}'"
