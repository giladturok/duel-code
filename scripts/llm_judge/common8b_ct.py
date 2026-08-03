"""Constants for the confidence-threshold (CT) add-on to the nucleus study.

Adds 4 CT arms (tau sweep, seed 1250, nucleus p=0.98) to the completed
16-config nucleus study in out8b_nuc/. CT is a variable-NFE sampler: each
arm is mapped to the budget cell whose nominal NFE is closest to the arm's
achieved mean generation NFE. The mapping is APPROXIMATE and per-sample
NFE varies widely (see CT_NFE_MEAN / CT_NFE_MEDIAN); every table that
shows CT next to a fixed-k config must carry that disclosure.

Blinding: 4 NEW opaque ids n16..n19 appended to out8b_nuc/blinding.json
(existing n00..n15 entries untouched), shuffle seed CT_BLIND_SEED.

Requests: absolute = 4 x 128; pairwise = per budget cell, CT vs each of
the 4 existing samplers of that cell x N_CT_PAIR_INDICES prefix-matched
pairs x 2 presentation orders. Existing samplers appear ONLY as
comparison sides in pairs (their texts come from out8b_nuc/prep.json,
i.e. the same prep pipeline); they are not re-judged standalone.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common8b_nuc as N

OUT_DIR = N.OUT_DIR
DATA_DIR = N.DATA_DIR
SEED = N.SEED
CT_BLIND_SEED = N.BLIND_SEED + 1   # 20260804 — fresh shuffle for n16..n19
CT_ID_START = 16                   # ids n16..n19
CT_FILE_SEED = 1250
N_CT_PAIR_INDICES = 30             # first 30 of the prep.json shuffled order

# budget cell (nominal NFE) -> tau, ordered high budget -> low.
CT_TAU_OF_BUDGET = {256: "0.99", 128: "0.075", 64: "0.041", 32: "0.028"}
CT_CONFIGS = [f"ct_tau{t}" for t in CT_TAU_OF_BUDGET.values()]
CT_BUDGET_OF = {f"ct_tau{t}": b for b, t in CT_TAU_OF_BUDGET.items()}

# Achieved generation NFE over the 128 shard0-128 samples (nfe_measured).
# Basis for the budget-cell mapping above; medians show the per-sample
# spread (tau0.028 median 14 vs mean 35).
CT_NFE_MEAN = {"ct_tau0.99": 241.4, "ct_tau0.075": 147.2,
               "ct_tau0.041": 80.0, "ct_tau0.028": 35.4}
CT_NFE_MEDIAN = {"ct_tau0.99": 255.0, "ct_tau0.075": 161.5,
                 "ct_tau0.041": 66.5, "ct_tau0.028": 14.0}

CT_NOTE = (
    "CT budgets are approximate: confidence-threshold decoding spends a "
    "variable number of forward passes per sample, so each tau arm is "
    "assigned to the budget cell nearest its achieved MEAN generation NFE "
    "(241/147/80/35 for tau 0.99/0.075/0.041/0.028 -> cells 256/128/64/32); "
    "per-sample NFE varies widely (medians 255/162/67/14).")


def ct_data_path(tau):
    return os.path.join(
        DATA_DIR, f"ct_bs32_tau{tau}_seed{CT_FILE_SEED}_shard0-128.jsonl")


# DUEL conditional ppl for the 4 CT arms — computed by a SEPARATE scoring
# job. Leave None until it lands; analyze8b skips None entries gracefully.
DUEL_PPL_CT = {name: None for name in CT_CONFIGS}
