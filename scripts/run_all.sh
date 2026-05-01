#!/bin/bash
# Run all DUEL paper experiments by submitting individual scripts.
#
# Usage:
#   bash scripts/run_all.sh            # Submit all sbatch jobs
#
# To run a subset, comment out sections below.

set -e

echo "========================================"
echo "Tables 2 & 4: In-Domain Perplexity (OWT)"
echo "========================================"

# ELBO
bash scripts/owt_perplexity/bd3lm_elbo.sh
bash scripts/owt_perplexity/mdlm_elbo.sh
bash scripts/owt_perplexity/sedd_elbo.sh
bash scripts/owt_perplexity/ar.sh

# DUEL
bash scripts/owt_perplexity/bd3lm_duel.sh
bash scripts/owt_perplexity/mdlm_duel.sh
bash scripts/owt_perplexity/sedd_duel.sh

echo "========================================"
echo "Table 3: In-Domain Perplexity (LM1B)"
echo "========================================"

# ELBO
bash scripts/lm1b_perplexity/bd3lm_elbo.sh
bash scripts/lm1b_perplexity/mdlm_elbo.sh
bash scripts/lm1b_perplexity/sedd_elbo.sh
bash scripts/lm1b_perplexity/ar.sh

# DUEL
bash scripts/lm1b_perplexity/bd3lm_duel.sh
bash scripts/lm1b_perplexity/mdlm_duel.sh
bash scripts/lm1b_perplexity/sedd_duel.sh

echo "========================================"
echo "Tables 4 & 8: Zero-Shot Perplexity"
echo "========================================"

# ELBO
bash scripts/zeroshot_perplexity/bd3lm_elbo.sh
bash scripts/zeroshot_perplexity/mdlm_elbo.sh
bash scripts/zeroshot_perplexity/sedd_elbo.sh
bash scripts/zeroshot_perplexity/ar.sh

# DUEL
bash scripts/zeroshot_perplexity/bd3lm_duel.sh
bash scripts/zeroshot_perplexity/mdlm_duel.sh
bash scripts/zeroshot_perplexity/sedd_duel.sh

echo "========================================"
echo "Table 6 / Figure 4 / Figure 5: Sampler Comparison"
echo "========================================"

# DUEL PPL for all unmasking rules + ELBO baseline
bash scripts/sampler_comparison/duel_ppl.sh

# Gen PPL + entropy + MAUVE for all sampling strategies + uniform baseline
bash scripts/sampler_comparison/sample_eval.sh

echo "========================================"
echo "All experiments submitted!"
echo "========================================"
