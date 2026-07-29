"""Is the within-budget CoLA ranking driven by repetition rather than by fluency?

Motivation. The `anchor-owt-repeat` probe scores *above* real held-out OWT on CoLA
acceptability, so the metric is blind to repetitive degeneracy. That is tolerable as a
stated limitation only if it does not actually change the ranking on the real cells. But
`block-left-to-right_k2` is measured as the most repetitive sampler cell of all
(rep_4 = 0.264 vs 0.111-0.136 for its NFE-512 peers), and left-to-right is exactly the
rule generative perplexity wrongly prefers within-budget. So the blindness could be
producing the same inversion, by the same mechanism, in the metric that was supposed to
fix it.

This script tests that directly, per NFE budget:

  1. Spearman/Pearson correlation between a passage's CoLA score and its rep_4, computed
     *within* each cell (so the compute axis cannot contribute).
  2. The within-budget cell ranking recomputed on the subset of passages whose rep_4 is
     below a threshold, i.e. with the repetitive passages removed. If the ranking flips
     back to agree with DUEL once repetition is controlled for, the inversion is a
     repetition artifact and the CoLA score must be reported paired with rep_4.
  3. The same ranking on a rep_4-matched subsample, as a check that (2) is not an artifact
     of throwing away different fractions of different cells.

Requires cola_score.sh output plus rep_stats.py --npz-out.
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cola_data as cd  # noqa: E402
from cola_analyze import BUDGETS, DUEL_PPL, passage_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--rep-npz", required=True)
    ap.add_argument("--model", default="yiiino/deberta-v3-large-cola")
    ap.add_argument("--scheme", default="sentence")
    ap.add_argument("--stat", default="wmean")
    ap.add_argument("--rep-thresh", type=float, default=0.10,
                    help="keep passages with rep_4 below this (real OWT mean is 0.034)")
    args = ap.parse_args()

    rep = np.load(args.rep_npz)
    mtag = args.model.replace("/", "_")
    cola, rep4 = {}, {}
    for f in sorted(glob.glob(os.path.join(
            args.scores, f"*__{mtag}__{args.scheme}.npz"))):
        z = np.load(f, allow_pickle=True)
        cell = str(z["cell"])
        key = f"{cell}::rep_4"
        if key not in rep:
            print(f"  (no rep_4 for {cell}, skipping)")
            continue
        c = passage_stats(z)[args.stat]
        r = rep[key]
        n = min(len(c), len(r))
        cola[cell], rep4[cell] = c[:n], r[:n]

    print(f"model={args.model} scheme={args.scheme} stat={args.stat} "
          f"rep_thresh={args.rep_thresh}")
    print(f"cells: {len(cola)}")

    print("\n=== 1. WITHIN-CELL correlation between CoLA score and rep_4 ===")
    print("A positive value means: inside a single cell, the more repetitive passages")
    print("get the HIGHER acceptability score. That is the blindness biting on real data.")
    print(f"{'cell':<32} {'spearman':>9} {'pearson':>9} {'n':>6}")
    for cell in sorted(cola):
        c, r = cola[cell], rep4[cell]
        m = ~(np.isnan(c) | np.isnan(r))
        if m.sum() < 20:
            continue
        rho = sps.spearmanr(c[m], r[m]).correlation
        pea = sps.pearsonr(c[m], r[m])[0]
        print(f"{cell:<32} {rho:>+9.3f} {pea:>+9.3f} {m.sum():>6}")

    print("\n=== 2. WITHIN-BUDGET RANKING, ALL vs LOW-REPETITION PASSAGES ===")
    for B in BUDGETS:
        cells = [c for c, (rule, _, b) in cd.CELLS.items()
                 if b == B and rule in DUEL_PPL and c in cola]
        if len(cells) < 2:
            continue
        print(f"\n--- NFE ~= {B} ---")
        print(f"{'cell':<32} {'all':>8} {'lowrep':>8} {'kept%':>7} "
              f"{'rep_4':>7} {'DUELppl':>8}")
        allv, lowv = {}, {}
        for c in cells:
            x, r = cola[c], rep4[c]
            m = ~(np.isnan(x) | np.isnan(r))
            keep = m & (r < args.rep_thresh)
            allv[c] = np.nanmean(x[m])
            lowv[c] = np.nanmean(x[keep]) if keep.sum() > 20 else np.nan
            print(f"{c:<32} {allv[c]:>8.4f} {lowv[c]:>8.4f} "
                  f"{100*keep.sum()/max(m.sum(),1):>7.1f} {np.nanmean(r):>7.4f} "
                  f"{DUEL_PPL[cd.CELLS[c][0]][B]:>8.2f}")
        for label, vals in (("all passages", allv), ("low-rep only", lowv)):
            order = sorted(cells, key=lambda c: -vals[c])
            duel = sorted(cells, key=lambda c: DUEL_PPL[cd.CELLS[c][0]][B])
            v = np.array([vals[c] for c in cells])
            p = np.array([DUEL_PPL[cd.CELLS[c][0]][B] for c in cells])
            ok = ~np.isnan(v)
            rho = (sps.spearmanr(v[ok], -p[ok]).correlation
                   if ok.sum() >= 3 else np.nan)
            print(f"  {label:<14} CoLA best->worst: "
                  f"{[cd.CELLS[c][0] for c in order]}  "
                  f"Spearman vs -DUELppl = {rho:+.3f}")
            print(f"  {'':<14} DUEL best->worst: {[cd.CELLS[c][0] for c in duel]}")

    print("\n=== 3. REP_4-MATCHED SUBSAMPLE ===")
    print("Rather than a hard threshold, resample passages in each cell to a common")
    print("rep_4 distribution (deciles of the pooled within-budget distribution), so")
    print("every cell contributes the same repetition profile.")
    rng = np.random.default_rng(0)
    for B in BUDGETS:
        cells = [c for c, (rule, _, b) in cd.CELLS.items()
                 if b == B and rule in DUEL_PPL and c in cola]
        if len(cells) < 2:
            continue
        pooled = np.concatenate([rep4[c] for c in cells])
        edges = np.nanpercentile(pooled, np.arange(0, 101, 10))
        edges = np.unique(edges)
        matched = {}
        for c in cells:
            x, r = cola[c], rep4[c]
            bins = np.digitize(r, edges[1:-1])
            # target count per bin = min available across the cells in this budget
            per = []
            for b_i in range(len(edges) - 1):
                idx = np.where((bins == b_i) & ~np.isnan(x))[0]
                nmin = min(
                    int(np.sum((np.digitize(rep4[c2], edges[1:-1]) == b_i)
                               & ~np.isnan(cola[c2]))) for c2 in cells)
                if nmin > 0 and len(idx) >= nmin:
                    per.append(x[rng.choice(idx, nmin, replace=False)])
            matched[c] = float(np.mean(np.concatenate(per))) if per else np.nan
        order = sorted(cells, key=lambda c: -matched[c])
        duel = sorted(cells, key=lambda c: DUEL_PPL[cd.CELLS[c][0]][B])
        v = np.array([matched[c] for c in cells])
        p = np.array([DUEL_PPL[cd.CELLS[c][0]][B] for c in cells])
        ok = ~np.isnan(v)
        rho = sps.spearmanr(v[ok], -p[ok]).correlation if ok.sum() >= 3 else np.nan
        print(f"\n  NFE={B}: " + "  ".join(f"{c.split('_')[0][6:]}={matched[c]:.4f}"
                                           for c in cells))
        print(f"    matched CoLA best->worst: {[cd.CELLS[c][0] for c in order]}  "
              f"Spearman vs -DUELppl = {rho:+.3f}")
        print(f"    DUEL          best->worst: {[cd.CELLS[c][0] for c in duel]}")


if __name__ == "__main__":
    main()
