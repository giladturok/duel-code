"""Constants for the nucleus p=0.98 8B prefix-grid judge study.

Same 16 configs, prompts, schemas, and settings as the base 8B study
(common8b), but: data from the nuc098_full5000 grid (phase-a shard0-128,
prefix/sample indices 0-127), ALL 128 samples judged per config, 50
prefix-matched pairs per config pair, fresh blinding (ids n00..n15,
shuffle seed SEED+1), outputs in out8b_nuc/.

Select this variant by running prep8b/build_batches8b/analyze8b with
JUDGE8B_VARIANT=nuc in the environment.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import common8b as B

DATA_DIR = ("/home/gt345/projects/scaling/exact_ll/outputs/"
            "sampling_grid_llada8b_prefix_nuc098_full5000")
OUT_DIR = os.path.join(C.HERE, "out8b_nuc")

SEED = B.SEED                 # 20260802 (index order)
BLIND_SEED = B.SEED + 1       # fresh blinding, distinct from out8b's mapping
ID_PREFIX = "n"
N_ABS_INDICES = 128           # ALL phase-a samples (0..127, shuffled order)
N_PAIR_INDICES = 50           # first 50 of the shuffled order
MAX_WORDS = B.MAX_WORDS

FILE_SEED = B.FILE_SEED
STRATEGIES = B.STRATEGIES
KS = B.KS
BUDGET_OF_K = B.BUDGET_OF_K
CONFIGS = B.CONFIGS

# The full5000 prefix set is all-full-length by construction; prep asserts
# it against this file (first 128 entries all have n_tokens == 256).
TRUE_CONT_PATH = os.path.join(
    "/home/gt345/projects/scaling/exact_ll/outputs/"
    "sampling_grid_llada8b_prefix", "true_continuations_full5000.json")
EXPECT_NUCLEUS_P = 0.98
ASSERT_ALL_FULL = True

STUDY_LABEL = "LLaDA-8B prefix grid, nucleus p=0.98, full-length prefixes"
STUDY_NOTE = (
    "All 128 phase-a samples per config judged; every prefix's true "
    "continuation is 256 tokens by construction (asserted). DUEL ppl "
    "below is measured on the BASE (non-nucleus) 8B grid samples "
    "(scores/*_duel.jsonl, 109-slice) — same (strategy, k) configs, "
    "different sampler temperature setting.")


def data_path(strategy, k):
    return os.path.join(
        DATA_DIR,
        f"{strategy}_bs32_k{k}_seed{FILE_SEED[strategy][k]}"
        "_shard0-128.jsonl")


ABS_SYSTEM_8B = B.ABS_SYSTEM_8B    # "research language model" wording
PAIR_SYSTEM_8B = B.PAIR_SYSTEM_8B

# Per-config DUEL conditional ppl on the base 8B grid: exp(sum nll_nats /
# sum n_tokens) over the full_length==True rows (109 of 128 per config) of
# outputs/sampling_grid_llada8b_prefix/scores/{config}_duel.jsonl.
# Computed 2026-08-02.
DUEL_PPL_8B = {
    "greedy_k1": 12.484, "greedy_k2": 19.578,
    "greedy_k4": 35.366, "greedy_k8": 70.708,
    "left_to_right_k1": 12.372, "left_to_right_k2": 24.987,
    "left_to_right_k4": 63.393, "left_to_right_k8": 153.740,
    "margin_k1": 12.773, "margin_k2": 16.586,
    "margin_k4": 25.217, "margin_k8": 48.272,
    "random_k1": 13.062, "random_k2": 14.339,
    "random_k4": 17.177, "random_k8": 25.067,
}
assert sorted(DUEL_PPL_8B) == sorted(CONFIGS)
