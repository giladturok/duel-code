#!/bin/bash
# Submit the whole E1 matrix (R2 W2/Q1, 110M held-out likelihood cost tables).
# Run from the repo root:  bash scripts/rebuttal/e1_launch_all.sh
#
# One job per TABLE, not one per arm: a6000s are scarce, and -- more
# importantly -- every arm within a table must land on the SAME node for its
# wall-clock column to mean anything. Each job also runs an identical
# calibration arm first (e1_calibrate) so cross-table node drift is visible.
set -u
cd "$(dirname "$0")/../.."
S="scripts/rebuttal"
ALL3="block_greedy block_left_to_right block_probability_margin"

# ---- DUEL: 8 val batches, interior frames only (first is warm-up dominated) --
ALGO=mdlm  BLOCK=4  SAMPLERS="${ALL3}" KS="4 2 1"         sbatch -J e1d_mdlm4  ${S}/e1_duel.sh
ALGO=bd3lm BLOCK=4  SAMPLERS="${ALL3}" KS="4 2 1"         sbatch -J e1d_bd4    ${S}/e1_duel.sh
ALGO=bd3lm BLOCK=8  SAMPLERS="${ALL3}" KS="8 4 2 1"       sbatch -J e1d_bd8    ${S}/e1_duel.sh
ALGO=bd3lm BLOCK=16 SAMPLERS="${ALL3}" KS="16 8 4 2 1"    sbatch -J e1d_bd16   ${S}/e1_duel.sh

# ---- ELBO: 4 MC repeats on the SAME sequences + one flex row per table ------
ALGO=mdlm  BLOCK=1024 REPS=4 BACKENDS="sdpa flex" sbatch -J e1e_mdlm  ${S}/e1_elbo.sh
ALGO=bd3lm BLOCK=4    REPS=4 BACKENDS="sdpa flex" sbatch -J e1e_bd4   ${S}/e1_elbo.sh
ALGO=bd3lm BLOCK=8    REPS=4 BACKENDS="sdpa flex" sbatch -J e1e_bd8   ${S}/e1_elbo.sh
ALGO=bd3lm BLOCK=16   REPS=4 BACKENDS="sdpa flex" sbatch -J e1e_bd16  ${S}/e1_elbo.sh

# ---- Reproduction gate: FULL 110,480-sequence split ------------------------
# The published "ELBO <=23.52" (sections/experiments.tex:152) cannot be checked
# on 8 batches. These two arms are the only full-split runs; they also give the
# measured (not extrapolated) full-split ELBO wall-clock for the cost tables.
ALGO=bd3lm BLOCK=16   REPS=1 LIMIT=1.0 TAG="_fullsplit" sbatch -J e1g_bd16 -t 8:00:00 ${S}/e1_elbo.sh
ALGO=mdlm  BLOCK=1024 REPS=1 LIMIT=1.0 TAG="_fullsplit" sbatch -J e1g_mdlm -t 8:00:00 ${S}/e1_elbo.sh

# ---- Batch-size sweep + attention-backend risk check ------------------------
sbatch -J e1_sweeps ${S}/e1_sweeps.sh

squeue -u "${USER}" -n e1d_mdlm4,e1d_bd4,e1d_bd8,e1d_bd16,e1e_mdlm,e1e_bd4,e1e_bd8,e1e_bd16,e1g_bd16,e1g_mdlm,e1_sweeps \
    -o "%.10i %.12j %.9T %R"
