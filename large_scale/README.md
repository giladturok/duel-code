# `large_scale/` — LLaDA-8B + Llama3-8B DUEL

Wraps LLaDA-8B and Llama3-8B with `lm-eval-harness` to compute exact / DUEL / ELBO perplexity on `lm-eval` perplexity benchmarks. Reproduces **Table 5** of [arxiv 2603.01367](https://arxiv.org/abs/2603.01367) (Wikitext, Lambada, AG News at 8B).

A separate harness from the rest of `duel/`: the parent repo uses hydra + pytorch-lightning over its own dataloaders, while this subtree inherits from `lm_eval.api.model.LM` and dispatches through `cli_evaluate`. The two stay decoupled — no shared imports across the boundary.

## Install

In addition to the parent repo's `requirements.txt`:

```bash
pip install lm-eval accelerate
```

The `tasks/` directory contains the lm-eval task YAMLs for the perplexity benchmarks; pass `--include_path tasks` so lm-eval discovers them.

## Reproduce Table 5

```bash
cd large_scale
bash scripts/table5.sh
```

This launches three jobs (Llama3 Exact, LLaDA ELBO, LLaDA DUEL) each on Wikitext / Lambada / AG News.

| Model  | Method | Wikitext | Lambada | AG News |
|--------|--------|---------:|--------:|--------:|
| Llama3 | Exact  |     7.94 |   32.40 |   41.29 |
| LLaDA  | ELBO   |  ≤ 15.29 | ≤ 39.04 | ≤ 85.17 |
| LLaDA  | DUEL   |    14.50 |   36.00 |   78.91 |

## Layout

| Path | Purpose |
|------|---------|
| `eval_llada_exact.py` | LLaDA exact-likelihood evaluator (registers `llada_exact` in lm-eval). Supports strategies: `greedy`, `block_greedy`, `block_left_to_right`, `block_probability_margin`, `block_permutation`. |
| `eval_llada_elbo.py` | LLaDA ELBO evaluator via Monte Carlo sampling (registers `llada_elbo`). |
| `eval_llama3.py` | Standard causal-LM baseline (registers `llama3`). |
| `eval_block_permutation.py` | Older variant focused on the oracle (block_permutation) strategy. Used for GSM8K / generation experiments. |
| `eval_bd3lm_elbo.py` | BD3-LM ELBO evaluator wrapped for lm-eval (used for cross-comparison runs). |
| `generate.py` | LLaDA generation helper (Gumbel noise + low-confidence remasking + semi-AR block decoding). |
| `exact_likelihood/compute.py` | `compute_deterministic_loglikelihood` and `compute_block_permutation_loglikelihood`. Model-agnostic core. |
| `exact_likelihood/strategies.py` | Selection strategies (`BlockGreedyConfidenceStrategy`, `BlockPermutationStrategy`, …). Mirror but distinct from the parent repo's `selection_strategies.py`. |
| `tasks/` | lm-eval task YAMLs for `wikitext`, `lambada_perplexity`, `ag_news_perplexity`, etc. |
| `scripts/launch_*.py` | Python job submitters that fan out the per-task accelerate-launches as SLURM jobs. |
| `scripts/run_job.sbatch` | Generic SLURM wrapper. |
| `scripts/table5.sh` | One-shot launcher for Table 5 (the three rows × three datasets). |

## Why two harnesses?

- The parent (`duel/`) needs hydra + lightning to interoperate with the BD3-LM training and the per-strategy `mode=duel_ppl` eval loop — that's how Tables 4/6/7 are reproduced.
- This subtree needs lm-eval-harness because the 8B benchmarks (and the Llama3 baseline) are easiest to do via lm-eval's standardized task definitions and result reporting.
- The two evaluation paths produce comparable perplexity numbers but via genuinely different code paths. We keep them separate rather than abstracting over both.
