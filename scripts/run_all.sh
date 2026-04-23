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
    sed -i "s/^BLOCK_SIZE=.*/BLOCK_SIZE=${BS}/" scripts/elbo/bd3lm_owt.sh
    bash scripts/elbo/bd3lm_owt.sh
done
bash scripts/elbo/mdlm_owt.sh
bash scripts/elbo/sedd_owt.sh
bash scripts/ar/elbo_owt.sh

# DUEL
for BS in 4 8 16; do
    sed -i "s/^BLOCK_SIZE=.*/BLOCK_SIZE=${BS}/" scripts/duel/bd3lm_owt.sh
    bash scripts/duel/bd3lm_owt.sh
done
bash scripts/duel/mdlm_owt.sh
bash scripts/duel/sedd_owt.sh

echo "========================================"
echo "Table 2: In-Domain Perplexity (LM1B)"
echo "========================================"

# ELBO
for BS in 4 8 16; do
    sed -i "s/^BLOCK_SIZE=.*/BLOCK_SIZE=${BS}/" scripts/elbo/bd3lm_lm1b.sh
    bash scripts/elbo/bd3lm_lm1b.sh
done
bash scripts/elbo/mdlm_lm1b.sh
bash scripts/elbo/sedd_lm1b.sh
bash scripts/ar/elbo_lm1b.sh

# DUEL
for BS in 4 8 16; do
    sed -i "s/^BLOCK_SIZE=.*/BLOCK_SIZE=${BS}/" scripts/duel/bd3lm_lm1b.sh
    bash scripts/duel/bd3lm_lm1b.sh
done
bash scripts/duel/mdlm_lm1b.sh
bash scripts/duel/sedd_lm1b.sh

echo "========================================"
echo "Table 3: Zero-Shot Perplexity"
echo "========================================"

# ELBO
for data in ag_news lambada ptb wikitext103 scientific_papers_pubmed scientific_papers_arxiv lm1b-gpt2; do
    bash scripts/elbo_zs/bd3lm_${data}.sh
    bash scripts/elbo_zs/mdlm_${data}.sh
    bash scripts/elbo_zs/sedd_${data}.sh
    bash scripts/ar_zs/elbo_${data}.sh
done

# DUEL
for data in ag_news lambada ptb wikitext103 scientific_papers_pubmed scientific_papers_arxiv lm1b-gpt2; do
    bash scripts/duel_zs/bd3lm_${data}.sh
    bash scripts/duel_zs/mdlm_${data}.sh
    bash scripts/duel_zs/sedd_${data}.sh
done

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
