"""Sensitivity of the within-budget CoLA-vs-DUEL conclusion to every analysis choice.

There are three defensible per-passage summary statistics (token-weighted mean,
unweighted mean, fraction of units above 0.5), two defensible segmentation schemes
(sentence, fixed window) and two gate-validated checkpoints. That is 12 combinations.
With only 3 integer-k cells per NFE budget a Spearman coefficient can only take the
values {-1, -0.5, +0.5, +1}, so a conclusion that holds for one combination and not
another is not a conclusion. This script prints the whole matrix so the reader can see
which budgets give a stable answer and which do not.
"""
import glob
import itertools
import os
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cola_data as cd  # noqa: E402
from cola_analyze import BUDGETS, DUEL_PPL, passage_stats  # noqa: E402

MODELS = ["yiiino/deberta-v3-large-cola", "textattack/roberta-base-CoLA"]
SCHEMES = ["sentence", "window"]
STATS = ["wmean", "mean", "frac"]


def main():
    scores = sys.argv[1] if len(sys.argv) > 1 else "out/scores"
    print("Spearman(CoLA, -DUEL ppl) within each NFE budget, n=3 cells per budget.")
    print("+1 = CoLA reproduces the DUEL ranking exactly; -1 = exactly inverts it.")
    print("Possible values with n=3: {-1, -0.5, +0.5, +1}.\n")
    hdr = f"{'model':<14} {'scheme':<9} {'stat':<6}" + "".join(
        f"{'NFE'+str(B):>9}" for B in BUDGETS) + f"   {'best@each budget'}"
    print(hdr)
    print("-" * len(hdr))
    table = {}
    for model, scheme, stat in itertools.product(MODELS, SCHEMES, STATS):
        mtag = model.replace("/", "_")
        cola = {}
        for f in glob.glob(os.path.join(scores, f"*__{mtag}__{scheme}.npz")):
            z = np.load(f, allow_pickle=True)
            cola[str(z["cell"])] = np.nanmean(passage_stats(z)[stat])
        row, best = [], []
        for B in BUDGETS:
            cells = [c for c, (r, _, b) in cd.CELLS.items()
                     if b == B and r in DUEL_PPL and c in cola]
            if len(cells) < 3:
                row.append(np.nan)
                best.append("?")
                continue
            v = np.array([cola[c] for c in cells])
            p = np.array([DUEL_PPL[cd.CELLS[c][0]][B] for c in cells])
            row.append(sps.spearmanr(v, -p).correlation)
            best.append(cd.CELLS[max(cells, key=lambda c: cola[c])][0][:4])
        table[(model, scheme, stat)] = row
        short = model.split("/")[-1][:13]
        print(f"{short:<14} {scheme:<9} {stat:<6}" +
              "".join(f"{x:>+9.2f}" if not np.isnan(x) else f"{'--':>9}"
                      for x in row) + "   " + " ".join(best))

    print("\n=== STABILITY PER BUDGET ===")
    for i, B in enumerate(BUDGETS):
        vals = [v[i] for v in table.values() if not np.isnan(v[i])]
        if not vals:
            continue
        uniq = sorted(set(vals))
        verdict = ("STABLE" if len(uniq) == 1 and uniq[0] > 0 else
                   "STABLE (inverts)" if len(uniq) == 1 and uniq[0] < 0 else
                   "consistently negative" if max(uniq) < 0 else
                   "consistently positive" if min(uniq) > 0 else
                   "UNSTABLE - sign flips with analysis choice")
        print(f"  NFE {B:>4}: values {uniq}  over {len(vals)} combinations -> {verdict}")


if __name__ == "__main__":
    main()
