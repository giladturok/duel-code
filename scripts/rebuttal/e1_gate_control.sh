#!/bin/bash
#SBATCH -J e1_gatectl
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="[a6000]"
#SBATCH --exclude=snavely-compute-02,abdelfattah-compute-02,seo-compute-02,davis-compute-01
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --no-requeue
#
# Why the full-split BD3-LM L'=16 ELBO gate does not reproduce.
#
# Measured now : 22.9632  (3453 batches at B=32, seed=null -> UNSHUFFLED loader)
# Archived     : 22.2939  (logs/bd3lm_owt_block_size16.log: 6892 batches at
#                          B=16, seed=1 -> SHUFFLED loader)   -> we are +3.00%
# Paper         : 22.27 (tab:ppl-owt) / 23.52 (tab:rule-comparison, 1000-seq)
#
# 3.00% is ~28 sigma of the MC noise (0.107% at this n), so it is systematic.
# Two candidate causes, and this job separates them:
#
#  A. DATA ORDER x ANTITHETIC STRATIFICATION. diffusion.py:1423-1428 builds
#     offset_b = arange(B*num_blocks)/(B*num_blocks) reshaped to (B, num_blocks),
#     so the sequence sitting at batch position i receives every one of its
#     t-draws inside the narrow band [i/B, (i+1)/B). Position in the batch, not
#     chance, decides a sequence's noise level. With shuffling that pairing is
#     re-randomised; with seed=null it is frozen for the whole pass.
#  B. DIFFERENT VALIDATION SET. The archived BD3-LM run covered 6892x16 =
#     110,272 sequences; the archived MDLM run and both of our full-split runs
#     cover 110,480. The archived BD3-LM evaluation therefore saw 208 fewer
#     sequences than the current code builds.
#
# ARM=order : B=32, seed=1 -> shuffled, everything else identical to our gate.
#             Isolates cause A. If this returns ~22.29, data order is the cause.
# ARM=exact : B=16, seed=1 -> reproduces the archived configuration exactly.
#             If ARM=order disagrees with archived but ARM=exact matches, the
#             residual is batch size (i.e. the stratification grid), not order.
#             Batch count also tells us directly whether cause B is real.

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate bd3lm

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export E1_ROOT="${PWD}/logs/e1_gatectl"
source scripts/rebuttal/_e1_common.sh
unset PL_GLOBAL_SEED

ARM_KIND="${ARM_KIND:-order}"
if [ "${ARM_KIND}" = "exact" ]; then BS=16; else BS=32; fi

# e1_common_args pins seed=null; strip it and pass a real seed so that
# main.py:145 hands valid_seed=1 to the loader and shuffling turns ON.
mapfile -t C < <(e1_common_args bd3lm 16 "${BS}" | grep -v '^seed=null$')

run_arm "gatectl__${ARM_KIND}__bd3lm16__bs${BS}__seed1__sdpa" \
    "${C[@]}" seed=1 mode=elbo_ppl model.attn_backend=sdpa \
    trainer.limit_val_batches=1.0

echo "GATE CONTROL ${ARM_KIND} DONE"
