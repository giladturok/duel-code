#!/bin/bash
# Run all DUEL experiments by calling individual scripts.
#
# Usage:
#   bash scripts/run_all.sh            # Run locally (requires GPU)
#   sbatch scripts/run_all.sh          # Submit as SLURM job
#
# To run a subset, comment out sections below.
# Individual scripts use `srun` for SLURM compatibility.
# When running locally (not via sbatch), srun falls back to local execution.

set -e

echo "========================================"
echo "Table 1: In-Domain Perplexity (OWT)"
echo "========================================"

# ELBO
bash scripts/1_owt_perplexity/bd3lm_elbo.sh
bash scripts/1_owt_perplexity/mdlm_elbo.sh
bash scripts/1_owt_perplexity/sedd_elbo.sh
bash scripts/1_owt_perplexity/ar.sh

# DUEL
bash scripts/1_owt_perplexity/bd3lm_duel.sh
bash scripts/1_owt_perplexity/mdlm_duel.sh
bash scripts/1_owt_perplexity/sedd_duel.sh

echo "========================================"
echo "Table 2: In-Domain Perplexity (LM1B)"
echo "========================================"

# ELBO
bash scripts/2_lm1b_perplexity/bd3lm_elbo.sh
bash scripts/2_lm1b_perplexity/mdlm_elbo.sh
bash scripts/2_lm1b_perplexity/sedd_elbo.sh
bash scripts/2_lm1b_perplexity/ar.sh

# DUEL
bash scripts/2_lm1b_perplexity/bd3lm_duel.sh
bash scripts/2_lm1b_perplexity/mdlm_duel.sh
bash scripts/2_lm1b_perplexity/sedd_duel.sh

echo "========================================"
echo "Table 3: Zero-Shot Perplexity"
echo "========================================"

# ELBO
bash scripts/3_zeroshot_perplexity/bd3lm_elbo.sh
bash scripts/3_zeroshot_perplexity/mdlm_elbo.sh
bash scripts/3_zeroshot_perplexity/sedd_elbo.sh
bash scripts/3_zeroshot_perplexity/ar.sh

# DUEL
bash scripts/3_zeroshot_perplexity/bd3lm_duel.sh
bash scripts/3_zeroshot_perplexity/mdlm_duel.sh
bash scripts/3_zeroshot_perplexity/sedd_duel.sh

echo "========================================"
echo "Table 4: Sampler Comparison"
echo "========================================"

# DUEL exact PPL
bash scripts/duel_strategy/block_greedy.sh
bash scripts/duel_strategy/block_left_to_right.sh
bash scripts/duel_strategy/block_probability_margin.sh
bash scripts/duel_strategy/block_conf_thresh.sh
bash scripts/duel_strategy/elbo_baseline.sh

# Generative PPL
bash scripts/gen_strategy/block_greedy.sh
bash scripts/gen_strategy/block_left_to_right.sh
bash scripts/gen_strategy/block_probability_margin.sh
bash scripts/gen_strategy/block_confidence_threshold.sh
bash scripts/gen_strategy/uniform.sh

echo "========================================"
echo "All experiments complete!"
echo "========================================"
