#!/bin/bash
#SBATCH -J e1_bsweep2
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=96G
#SBATCH -t 4:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="[a6000]"
#SBATCH --exclude=snavely-compute-02,abdelfattah-compute-02,seo-compute-02,davis-compute-01
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --no-requeue
#
# Clean re-run of the batch-size sweep ONLY, into a fresh E1_ROOT.
#
# Why this exists: the first attempt (job 666395, e1_sweeps.sh) was PREEMPTED
# three times on ellis-compute-02 and resumed on fang-compute-01. Because
# run_arm skips arms already at exit_code=0, the surviving sweep was split
# across two nodes -- B=1 and B=8 on fang, B=32 and B=64 on ellis. Every point
# in a batch-size sweep is a comparison against the other points, so a
# node-split sweep is not a valid sweep even if the drift is small (calibration
# puts fang vs ellis at 0.9%, and node spread overall at 1.04x).
#
# --no-requeue is deliberate: a silent mid-sweep node migration is exactly the
# failure being corrected. Better to lose the job and resubmit than to produce
# a second split sweep that looks clean.

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate bd3lm

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export E1_ROOT="${PWD}/logs/e1_bsweep2"
source scripts/rebuttal/_e1_common.sh
unset PL_GLOBAL_SEED

echo "CLEAN B-SWEEP on $(hostname -s), all arms in one job, E1_ROOT=${E1_ROOT}"
e1_calibrate

for BS in 1 8 32 64; do
    mapfile -t C < <(e1_common_args bd3lm 16 "${BS}")
    run_arm "bsweep2__bd3lm16__block_greedy__k1__bs${BS}__sdpa" \
        "${C[@]}" mode=duel_ppl \
        eval.exact_ll_strategy=block_greedy eval.exact_ll_k=1 \
        model.attn_backend=sdpa trainer.limit_val_batches=8 \
        +eval.exact_ll_use_kv_cache=true sampling.kv_cache=true
    run_arm "bsweep2__elbo__bd3lm16__bs${BS}__sdpa" \
        "${C[@]}" mode=elbo_ppl model.attn_backend=sdpa \
        trainer.limit_val_batches=8
done

echo "CLEAN B-SWEEP DONE -- verify one node with:"
echo "  grep -h '^node=' ${E1_ROOT}/*/meta.txt | sort -u"
