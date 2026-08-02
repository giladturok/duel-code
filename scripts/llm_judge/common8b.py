"""Shared constants for the 8B prefix-grid judge study.

Reuses the 110M study's prompts and schemas (common.py) with one approved
edit: the absolute-rating system prompt says "research language model"
instead of "small (110M-parameter) research language model" (the 8B samples
are less degraded at k=1; the never-refuse framing is kept). All outputs go
to out8b/ with a fresh blinding (opaque ids b00..b15).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

DATA_DIR = ("/home/gt345/projects/scaling/exact_ll/outputs/"
            "sampling_grid_llada8b_prefix")
OUT_DIR = os.path.join(C.HERE, "out8b")

SEED = C.SEED                  # 20260802
BLIND_SEED = C.SEED            # blinding shuffle seed (out8b mapping)
ID_PREFIX = "b"                # opaque config ids b00..b15
N_ABS_INDICES = 100            # prefix indices sampled from 0..127
N_PAIR_INDICES = 30            # first 30 of those for pairwise
MAX_WORDS = C.MAX_WORDS        # 400-word cap (rarely binds at ~256 tokens)

TRUE_CONT_PATH = None          # set below after DATA_DIR helpers
EXPECT_NUCLEUS_P = None        # base grid has no nucleus sampling
ASSERT_ALL_FULL = False        # 19/128 true continuations are short

STUDY_LABEL = "LLaDA-8B prefix grid (16 configs)"
STUDY_NOTE = ("Judge text = continuation truncated at first "
              "<|endoftext|>. 19/128 true continuations are short "
              "(full_length_continuation flag recorded per row).")

# strategy -> per-config seed suffix in the canonical filenames
FILE_SEED = {
    "left_to_right": {1: 1234, 2: 1235, 4: 1236, 8: 1237},
    "greedy": {1: 1238, 2: 1239, 4: 1240, 8: 1241},
    "margin": {1: 1242, 2: 1243, 4: 1244, 8: 1245},
    "random": {1: 1246, 2: 1247, 4: 1248, 8: 1249},
}
STRATEGIES = sorted(FILE_SEED)
KS = [1, 2, 4, 8]
BUDGET_OF_K = {1: 256, 2: 128, 4: 64, 8: 32}  # nominal NFE

CONFIGS = [f"{s}_k{k}" for s in STRATEGIES for k in KS]


def data_path(strategy, k):
    return os.path.join(
        DATA_DIR, f"{strategy}_bs32_k{k}_seed{FILE_SEED[strategy][k]}.jsonl")


TRUE_CONT_PATH = os.path.join(DATA_DIR, "true_continuations.json")


ABS_SYSTEM_8B = C.ABS_SYSTEM.replace(
    "small (110M-parameter) research language model",
    "research language model")
assert ABS_SYSTEM_8B != C.ABS_SYSTEM, "prompt edit did not apply"
PAIR_SYSTEM_8B = C.PAIR_SYSTEM  # unchanged (only the one approved edit)

# Per-config DUEL conditional ppl on this grid's own samples:
# exp(sum nll_nats / sum n_tokens) over the full_length==True rows (109 of
# 128 per config) of sampling_grid_llada8b_prefix/scores/{config}_duel.jsonl.
# Computed 2026-08-02 (previously a TODO hook; GPU job landed).
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
