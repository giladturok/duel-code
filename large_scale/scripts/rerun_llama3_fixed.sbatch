#!/bin/bash
# Rerun the Llama-3 AR baseline after the off-by-one fix (branch fix-ar-baseline-shift,
# commit 8bc2ecd). See large_scale/tests/test_llama3_scoring.py for what the fix pins.
#
#   $1 = TASKS  (comma-separated lm-eval task list)
#   $2 = TAG    (subdirectory under outputs/llama3_fixed/)
#
# Two arms are launched by the caller:
#   rolling  -- wikitext,lambada_perplexity,ag_news_perplexity
#               CONFIRMATION ONLY. The rolling path was never affected (prompt_len=0 means
#               the scored set is unchanged by the fix; the 2 rolling tests pass on BOTH
#               the pre- and post-fix code). Published Table 4 values to compare against:
#               7.94 / 32.40 / 41.29. These should reproduce.
#   mc       -- arc_challenge,hellaswag,piqa
#               THE NUMBERS THAT ACTUALLY MOVE. The fix restores the first continuation
#               token, which on multiple choice is the first content word.
#               winogrande is deliberately EXCLUDED: its local task definition puts the
#               blanked sentence in the context with fully-resolved sentences as choices,
#               so it is near-solvable by copying and not comparable to published numbers.
#
# Sizing note: 1 GPU / 32G / 2 cpus deliberately. Over-requesting host memory and CPUs is
# what left cells PENDING for ~30h against idle a6000s during the MC sweep -- every node
# with a free GPU was exhausted on RAM/CPU, not GPU.
#SBATCH -J llama3_fixed
#SBATCH --partition=gpu
#SBATCH --constraint="[a6000|a100|h100|h200]"
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

TASKS="$1"
TAG="$2"

export HF_HOME=/share/kuleshov/gt345/.cache/huggingface
export HF_TRUST_REMOTE_CODE=1

echo "=========================================="
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURM_NODELIST}"
echo "Tasks:    ${TASKS}"
echo "Tag:      ${TAG}"
echo "Start:    $(date)"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate dllm

cd /home/gt345/projects/scaling/duel/large_scale

# Record the exact code state the numbers came from -- the tree was previously untracked
# for long stretches, which left earlier numbers unattributable to a SHA.
echo "git branch: $(git rev-parse --abbrev-ref HEAD)"
echo "git commit: $(git rev-parse HEAD)"
echo "git dirty:  $(test -n "$(git status --porcelain -- large_scale)" && echo yes || echo no)"

python -u -m accelerate.commands.launch --num_processes 1 eval_llama3.py \
    --tasks "${TASKS}" \
    --include_path tasks \
    --model llama3 \
    --batch_size 16 \
    --model_args model_path='meta-llama/Meta-Llama-3-8B',dtype='bfloat16' \
    --output_path "outputs/llama3_fixed/${TAG}" \
    --log_samples

echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
