#!/bin/bash
#SBATCH -J e1_sweeps
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
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
# E1 side sweeps. BOTH must run inside ONE job: every point is a comparison
# against the other points, so they have to share a node -- two "RTX A6000"s
# have been measured 2.4x apart on identical work.
#
#   A. batch-size sweep, ONE representative arm (BD3-LM L'=16, greedy, k=1 --
#      the paper's protocol). B is pinned at 32 everywhere else; this is the
#      evidence for that choice. _pad_block_logits_to_full (exact_likelihood.py
#      :656-659) allocates [B, 1024, 50258] fp32 = 206 MB/sequence, so the
#      DUEL arm alone needs 6.6 GB at B=32 and 26 GB at B=128.
#   B. attention-backend check. The March MDLM DUEL run used flash_attn (the
#      configs/model/small.yaml default), NOT sdpa. If sdpa is materially
#      slower, every DUEL number here is on a different backend from the
#      paper's and that has to be disclosed rather than silently normalised.

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate bd3lm

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/rebuttal/_e1_common.sh
unset PL_GLOBAL_SEED

e1_calibrate

# ---- A. batch-size sweep -----------------------------------------------------
for BS in 1 8 32 64; do
    mapfile -t C < <(e1_common_args bd3lm 16 "${BS}")
    run_arm "bsweep__bd3lm16__block_greedy__k1__bs${BS}__sdpa" \
        "${C[@]}" mode=duel_ppl \
        eval.exact_ll_strategy=block_greedy eval.exact_ll_k=1 \
        model.attn_backend=sdpa trainer.limit_val_batches=8 \
        +eval.exact_ll_use_kv_cache=true sampling.kv_cache=true
    # ELBO at the same B, so the memory column is comparable arm-to-arm.
    run_arm "bsweep__elbo__bd3lm16__bs${BS}__sdpa" \
        "${C[@]}" mode=elbo_ppl model.attn_backend=sdpa \
        trainer.limit_val_batches=8
done

# ---- B. attention backend ----------------------------------------------------
for BK in sdpa flash_attn flex; do
    mapfile -t C < <(e1_common_args mdlm 4 32)
    run_arm "backend__mdlm4__block_greedy__k1__bs32__${BK}" \
        "${C[@]}" mode=duel_ppl \
        eval.exact_ll_strategy=block_greedy eval.exact_ll_k=1 \
        model.attn_backend="${BK}" trainer.limit_val_batches=8
    mapfile -t C < <(e1_common_args bd3lm 16 32)
    run_arm "backend__bd3lm16__block_greedy__k1__bs32__${BK}" \
        "${C[@]}" mode=duel_ppl \
        eval.exact_ll_strategy=block_greedy eval.exact_ll_k=1 \
        model.attn_backend="${BK}" trainer.limit_val_batches=8 \
        +eval.exact_ll_use_kv_cache=true sampling.kv_cache=true
done

echo "ALL SWEEP ARMS DONE"
