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
for BS in 4 8 16; do
    sed -i "s/^BLOCK_SIZE=.*/BLOCK_SIZE=${BS}/" scripts/elbo_ppl/ppl_owt_bd3lm.sh
    bash scripts/elbo_ppl/ppl_owt_bd3lm.sh
done
bash scripts/elbo_ppl/ppl_owt_mdlm.sh
bash scripts/elbo_ppl/ppl_owt_sedd.sh
bash scripts/elbo_ppl/ppl_owt_ar.sh

# DUEL
for BS in 4 8 16; do
    sed -i "s/^BLOCK_SIZE=.*/BLOCK_SIZE=${BS}/" scripts/duel_ppl/bd3lm_owt.sh
    bash scripts/duel_ppl/bd3lm_owt.sh
done
bash scripts/duel_ppl/mdlm_owt.sh
bash scripts/duel_ppl/sedd_owt.sh

echo "========================================"
echo "Table 2: In-Domain Perplexity (LM1B)"
echo "========================================"

# ELBO
for BS in 4 8 16; do
    sed -i "s/^BLOCK_SIZE=.*/BLOCK_SIZE=${BS}/" scripts/elbo_ppl/ppl_lm1b_bd3lm.sh
    bash scripts/elbo_ppl/ppl_lm1b_bd3lm.sh
done
bash scripts/elbo_ppl/ppl_lm1b_mdlm.sh
bash scripts/elbo_ppl/ppl_lm1b_sedd.sh
bash scripts/elbo_ppl/ppl_lm1b_ar.sh

# DUEL
for BS in 4 8 16; do
    sed -i "s/^BLOCK_SIZE=.*/BLOCK_SIZE=${BS}/" scripts/duel_ppl/bd3lm_lm1b.sh
    bash scripts/duel_ppl/bd3lm_lm1b.sh
done
bash scripts/duel_ppl/mdlm_lm1b.sh
bash scripts/duel_ppl/sedd_lm1b.sh

echo "========================================"
echo "Table 3: Zero-Shot Perplexity"
echo "========================================"

# ELBO
bash scripts/elbo_zs_ppl/ppl_zs_owt_bd3lm.sh
bash scripts/elbo_zs_ppl/ppl_zs_owt_mdlm.sh
bash scripts/elbo_zs_ppl/ppl_zs_owt_sedd.sh
bash scripts/elbo_zs_ppl/ppl_zs_owt_ar.sh

# DUEL
bash scripts/duel_zs_ppl/owt_bd3lm_block_greedy.sh
bash scripts/duel_zs_ppl/owt_mdlm_block_greedy.sh
bash scripts/duel_zs_ppl/owt_sedd_block_greedy.sh

echo "========================================"
echo "Table 4: Sampler Comparison"
echo "========================================"

# DUEL exact PPL
bash scripts/sampler_duel_ppl/block_greedy.sh
bash scripts/sampler_duel_ppl/block_left_to_right.sh
bash scripts/sampler_duel_ppl/block_probability_margin.sh
bash scripts/sampler_duel_ppl/block_conf_thresh.sh
bash scripts/sampler_duel_ppl/elbo.sh

# Generative PPL
bash scripts/sampler_gen_ppl/block_greedy.sh
bash scripts/sampler_gen_ppl/block_left_to_right.sh
bash scripts/sampler_gen_ppl/block_probability_margin.sh
bash scripts/sampler_gen_ppl/block_confidence_threshold.sh
bash scripts/sampler_gen_ppl/uniform.sh

echo "========================================"
echo "All experiments complete!"
echo "========================================"
