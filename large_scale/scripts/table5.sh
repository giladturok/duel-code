#!/bin/bash
# Reproduces Table 5 of arxiv 2603.01367.
# Large-scale (8B) DUEL evaluation on Wikitext / Lambada / AG News.
#
# Paper numbers:
#   Llama3 Exact   :  7.94 / 32.40 / 41.29
#   LLaDA  ELBO    : ≤15.29 / ≤39.04 / ≤85.17  (mc_num=128)
#   LLaDA  DUEL    : 14.50 / 36.00 / 78.91     (strategy=block_greedy)
#
# Run from this directory:
#   cd /home/gt345/projects/scaling/duel/large_scale
#   bash scripts/table5.sh
#
# Each run is one accelerate-launch via lm-eval-harness; the harness
# loads the task definitions in tasks/ (must pass --include_path tasks).
# Outputs land under outputs/{task}/{model}/{method}/.

set -euo pipefail

TASKS="wikitext,lambada_perplexity,ag_news_perplexity"

# 1) Llama3 Exact
accelerate launch eval_llama3.py \
    --tasks ${TASKS} \
    --include_path tasks \
    --model llama3 \
    --batch_size 16 \
    --model_args model_path='meta-llama/Meta-Llama-3-8B',dtype='bfloat16' \
    --output_path outputs/table5/llama3_exact \
    --log_samples

# 2) LLaDA ELBO (mc_num=128 matches the paper / the existing PERPLEXITY_EXPERIMENTS list)
accelerate launch eval_llada_elbo.py \
    --tasks ${TASKS} \
    --include_path tasks \
    --model llada_elbo \
    --batch_size 16 \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=128 \
    --output_path outputs/table5/llada_elbo \
    --log_samples

# 3) LLaDA DUEL (strategy=block_greedy)
accelerate launch eval_llada_exact.py \
    --tasks ${TASKS} \
    --include_path tasks \
    --model llada_exact \
    --batch_size 16 \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy \
    --output_path outputs/table5/llada_duel \
    --log_samples
