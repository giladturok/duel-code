"""Aggregate CoLA unit scores to per-cell acceptability, with error bars and tests.

Aggregation
-----------
Three per-passage summaries are computed from the per-unit P(acceptable):

  `mean`      unweighted mean over the passage's units
  `wmean`     mean weighted by each unit's GPT-2 token count  <-- headline statistic
  `frac`      fraction of units with P(acceptable) > 0.5

`wmean` is the headline number. Reason: a token-weighted mean is a property of the
*text*, not of the segmentation. Degraded cells over-segment (measured: 42 vs 9
sentence-terminal marks per 1k chars at NFE 128 vs 1024), so they contain many short
units; an unweighted mean would let a degraded passage be flattered by a large number of
tiny locally-plausible fragments, whereas token weighting gives each fragment influence
proportional to how much of the passage it actually is. `frac` is reported too because it
is the most interpretable ("what share of this text is acceptable English?") but it
discards the classifier's confidence, and `mean` is reported as the naive baseline. All
three are given so the ranking can be checked for robustness to the choice.

Statistics
----------
The unit of resampling is the **passage** (1000 per cell), because units within a passage
are not independent -- they come from one 1024-token generation. Bootstrap: resample
passages with replacement, recompute the cell mean, take percentile CIs. Within-budget
pairwise tests resample the *same* bootstrap indices for both cells (paired on passage
index, valid because all cells share seed 2 and passage i of every cell is generated from
the same noise draw), and report the two-sided bootstrap p-value for a difference in
means plus the paired mean difference.

Reference values
----------------
DUEL held-out perplexity per (rule, NFE) is taken from the paper's Table 5. It is *not*
recomputed here.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cola_data as cd  # noqa: E402

# Paper Table 5: DUEL held-out perplexity at NFE 128/256/512/1024.
DUEL_PPL = {
    "left-to-right": {128: 240.27, 256: 109.00, 512: 44.51, 1024: 21.46},
    "greedy": {128: 165.48, 256: 67.15, 512: 34.74, 1024: 22.04},
    "probability-margin": {128: 141.31, 256: 57.66, 512: 32.33, 1024: 22.35},
}

BUDGETS = [128, 256, 512, 1024]
ANCHORS = ["anchor-owt-real", "anchor-owt-repeat", "anchor-owt-word-shuffled",
           "anchor-owt-token-shuffled"]


def passage_stats(npz):
    """Per-passage mean / token-weighted mean / fraction>0.5, plus retention."""
    off = npz["offsets"]
    p = npz["unit_prob"].astype(np.float64)
    w = npz["unit_tok"].astype(np.float64)
    n = len(off) - 1
    out = {k: np.full(n, np.nan) for k in ("mean", "wmean", "frac")}
    for i in range(n):
        a, b = off[i], off[i + 1]
        if b <= a:
            continue
        pi, wi = p[a:b], w[a:b]
        out["mean"][i] = pi.mean()
        out["wmean"][i] = (pi * wi).sum() / wi.sum() if wi.sum() > 0 else pi.mean()
        out["frac"][i] = (pi > 0.5).mean()
    out["n_units"] = np.diff(off)
    for k in ("n_tok_in", "n_tok_scored", "n_nws_in", "n_nws_scored",
              "n_units_short"):
        out[k] = npz[k].astype(np.float64)
    return out


def bootstrap_ci(x, n_boot, rng, alpha=0.05):
    x = x[~np.isnan(x)]
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi), float(boots.std(ddof=1))


def paired_test(a, b, n_boot, rng):
    """Paired bootstrap over passage index. Returns (diff, ci_lo, ci_hi, p)."""
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided bootstrap p: fraction of resamples on the other side of 0, x2
    frac = min((boots <= 0).mean(), (boots >= 0).mean())
    p = min(1.0, 2 * frac)
    return float(d.mean()), float(lo), float(hi), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--model", default="yiiino/deberta-v3-large-cola")
    ap.add_argument("--scheme", default="sentence")
    ap.add_argument("--stat", default="wmean", choices=["mean", "wmean", "frac"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    mtag = args.model.replace("/", "_")
    files = sorted(glob.glob(os.path.join(
        args.scores, f"*__{mtag}__{args.scheme}.npz")))
    if not files:
        raise SystemExit(f"no score files for {args.model} / {args.scheme} "
                         f"in {args.scores}")

    data = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        cell = str(z["cell"])
        data[cell] = passage_stats(z)

    print(f"model={args.model}  scheme={args.scheme}  stat={args.stat}  "
          f"n_boot={args.n_boot}")
    print(f"cells present: {len(data)}  "
          f"(missing: {sorted(set(cd.CELLS) | set(ANCHORS) - set(data)) or 'none'})")
    rng = np.random.default_rng(args.seed)

    # ---------- per-cell table ----------
    rows = {}
    for cell, d in data.items():
        x = d[args.stat]
        mean, lo, hi, se = bootstrap_ci(x, args.n_boot, rng)
        rows[cell] = {
            "n_passages": int(np.sum(~np.isnan(x))),
            "stat": mean, "ci_lo": lo, "ci_hi": hi, "boot_se": se,
            "median": float(np.nanmedian(x)),
            "p10": float(np.nanpercentile(x, 10)),
            "p90": float(np.nanpercentile(x, 90)),
            "units_per_passage": float(d["n_units"].mean()),
            "tok_per_unit": float(d["n_tok_scored"].sum() /
                                  max(d["n_units"].sum(), 1)),
            "tok_retention": float(d["n_tok_scored"].sum() /
                                   max(d["n_tok_in"].sum(), 1)),
            "content_retention": float(d["n_nws_scored"].sum() /
                                       max(d["n_nws_in"].sum(), 1)),
            "short_unit_frac": float(d["n_units_short"].sum() /
                                     max(d["n_units"].sum(), 1)),
            "empty_passages": int(np.sum(np.isnan(x))),
        }
        for alt in ("mean", "wmean", "frac"):
            rows[cell][f"alt_{alt}"] = float(np.nanmean(d[alt]))

    print("\n=== ANCHORS ===")
    hdr = (f"{'cell':<30} {'stat':>7} {'95% CI':>17} {'median':>7} "
           f"{'p10-p90':>15} {'u/psg':>6} {'tok/u':>6} {'contRet':>8} {'drop':>6}")
    print(hdr)
    for cell in ANCHORS:
        if cell not in rows:
            continue
        r = rows[cell]
        print(f"{cell:<30} {r['stat']:>7.4f} "
              f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] {r['median']:>7.4f} "
              f"[{r['p10']:.3f},{r['p90']:.3f}] {r['units_per_passage']:>6.1f} "
              f"{r['tok_per_unit']:>6.1f} {r['content_retention']:>8.5f} "
              f"{1-r['content_retention']:>6.4f}")

    print("\n=== SAMPLER CELLS, GROUPED BY NFE BUDGET (all comparisons within budget) ===")
    for B in BUDGETS:
        cells = [c for c, (_, _, b) in cd.CELLS.items() if b == B and c in rows]
        cells.sort(key=lambda c: -rows[c]["stat"])
        print(f"\n--- NFE ~= {B} ---")
        print(hdr)
        for c in cells:
            r = rows[c]
            print(f"{c:<30} {r['stat']:>7.4f} "
                  f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] {r['median']:>7.4f} "
                  f"[{r['p10']:.3f},{r['p90']:.3f}] {r['units_per_passage']:>6.1f} "
                  f"{r['tok_per_unit']:>6.1f} {r['content_retention']:>8.5f} "
                  f"{1-r['content_retention']:>6.4f}")
        # pairwise within-budget tests, integer-k cells only
        ints = [c for c in cells if cd.CELLS[c][0] in DUEL_PPL]
        if len(ints) > 1:
            print(f"  pairwise (paired bootstrap over passages, integer-k only):")
            for i in range(len(ints)):
                for j in range(i + 1, len(ints)):
                    a, b = ints[i], ints[j]
                    d, lo, hi, p = paired_test(
                        data[a][args.stat], data[b][args.stat], args.n_boot, rng)
                    sig = "*" if p < 0.05 else " "
                    print(f"    {a:<30} - {b:<30} "
                          f"diff={d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f} {sig}")

    # ---------- mean generative perplexity per cell, for contrast ----------
    gen_ppl = {}
    for cell in cd.CELLS:
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "sample_logs",
            cd.PREFIX + cell + ".txt")
        if os.path.exists(path):
            gen_ppl[cell] = float(np.mean(
                cd.read_numeric_columns(path)["gen_ppl"]))

    # ---------- within-budget rank correlation vs DUEL ppl ----------
    print("\n=== WITHIN-BUDGET RANK CORRELATION vs DUEL PPL (Table 5) ===")
    print("Only the 12 integer-k cells are used. The confidence-threshold cells are")
    print("EXCLUDED: sample_logs used tau in {0.045,0.08,0.16,1.0} but duel_ppl.sh:38")
    print("used {0.05,0.07,0.15,0.99} -- different systems, not pairable across runs.")
    corr = {}
    for B in BUDGETS:
        pairs = []
        for cell, (rule, k, b) in cd.CELLS.items():
            if b == B and rule in DUEL_PPL and cell in rows:
                pairs.append((cell, rows[cell]["stat"], DUEL_PPL[rule][B]))
        if len(pairs) < 3:
            continue
        acc = np.array([p[1] for p in pairs])
        ppl = np.array([p[2] for p in pairs])
        # negate ppl so that "higher is better" for both axes
        rho, prho = sps.spearmanr(acc, -ppl)
        pear, ppear = sps.pearsonr(acc, -ppl)
        gp = np.array([gen_ppl.get(p[0], np.nan) for p in pairs])
        rho_g = sps.spearmanr(gp, ppl).correlation if not np.isnan(gp).any() else np.nan
        corr[B] = {"n": len(pairs), "spearman": float(rho), "spearman_p": float(prho),
                   "pearson": float(pear), "pearson_p": float(ppear),
                   "genppl_spearman_vs_duel": float(rho_g),
                   "cells": [(p[0], p[1], p[2], float(gp[i]))
                             for i, p in enumerate(pairs)]}
        print(f"\n  NFE={B}  (n={len(pairs)} cells)")
        for i, (cell, a, pp) in enumerate(
                sorted(pairs, key=lambda t: -t[1])):
            print(f"    {cell:<32} CoLA={a:.4f}  DUEL-ppl={pp:7.2f}  "
                  f"gen-ppl={gen_ppl.get(cell, float('nan')):7.1f}")
        print(f"    Spearman(CoLA, -DUELppl) = {rho:+.3f} (p={prho:.3f});  "
              f"Pearson = {pear:+.3f}")
        print(f"    for contrast, Spearman(gen-ppl, DUEL-ppl) = {rho_g:+.3f} "
              f"(both 'lower is better', so +1 = agrees, -1 = inverts)")
        print(f"    CoLA    ranking best->worst: "
              f"{[cd.CELLS[c][0] for c, _, _ in sorted(pairs, key=lambda t:-t[1])]}")
        print(f"    DUEL    ranking best->worst: "
              f"{[cd.CELLS[c][0] for c, _, _ in sorted(pairs, key=lambda t: t[2])]}")
        print(f"    gen-ppl ranking best->worst: "
              f"{[cd.CELLS[c][0] for c in sorted([p[0] for p in pairs], key=lambda c: gen_ppl.get(c, 9e9))]}")

    # pooled, reported only to show it is the artifact the brief warns about
    allp = [(c, rows[c]["stat"], DUEL_PPL[cd.CELLS[c][0]][cd.CELLS[c][2]])
            for c in rows if c in cd.CELLS and cd.CELLS[c][0] in DUEL_PPL]
    if len(allp) >= 4:
        rho, prho = sps.spearmanr([p[1] for p in allp], [-p[2] for p in allp])
        print(f"\n  POOLED over all {len(allp)} integer-k cells: "
              f"Spearman = {rho:+.3f} (p={prho:.4f})")
        print("  ^ NOT a valid comparison: the NFE axis (128->1024) dominates the")
        print("    sampler axis, so every metric correlates with every other. Reported")
        print("    only for contrast with the within-budget numbers above.")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"model": args.model, "scheme": args.scheme,
                       "stat": args.stat, "cells": rows,
                       "within_budget_corr": corr}, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
