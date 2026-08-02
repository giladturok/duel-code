"""Unblind and analyze judge results; emit out/summary.md.

Inputs:  out/blinding.json, out/prep.json, out/abs_results.jsonl,
         out/pair_results.jsonl
Outputs: out/summary.md (+ printed to stdout)

Analyses
--------
- Per-config means with 95% bootstrap CIs (10k resamples) for the 5
  absolute dimensions.
- Spearman(judge overall, DUEL ppl) across the 16 configs, with a
  10k-permutation two-sided p; comparison correlations for gen-ppl (mean
  over the sampled 100 rows) and MAUVE (file-level) against DUEL ppl.
- Within-budget 4x4 order-averaged win-rate matrices (tie = 0.5; unit =
  sample pair averaged over its AB/BA presentations) with Wilson CIs
  (computed on the summed unit scores — approximate, since units take
  values in {0, .25, .5, .75, 1}).
- Flip-consistency rate per budget (both presentation orders agree).
- Length-bias check: Spearman(orig_words, overall) within each config.
"""
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

N_BOOT = 10_000
N_PERM = 10_000
RNG = np.random.default_rng(C.SEED)

SHORT = {
    "block-greedy": "greedy", "block-left-to-right": "l2r",
    "block-probability-margin": "margin",
    "block-confidence-threshold": "conf-thr",
}


def short_name(name):
    stem, k = name.rsplit("_k", 1)
    return f"{SHORT[stem]} k{k}"


def _rank(x):
    """Average ranks (ties averaged)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def perm_p(x, y, n=N_PERM):
    obs = spearman(x, y)
    y = np.asarray(y, dtype=float)
    count = 0
    for _ in range(n):
        if abs(spearman(x, RNG.permutation(y))) >= abs(obs) - 1e-12:
            count += 1
    return obs, (count + 1) / (n + 1)


def boot_ci(values, n=N_BOOT):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def wilson(successes, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


def main():
    blinding = C.load_json(os.path.join(C.OUT_DIR, "blinding.json"))
    mapping = blinding["mapping"]                # cid -> real name
    prep = C.load_json(os.path.join(C.OUT_DIR, "prep.json"))
    abs_results = C.load_jsonl(os.path.join(C.OUT_DIR, "abs_results.jsonl"))
    pair_results = C.load_jsonl(os.path.join(C.OUT_DIR, "pair_results.jsonl"))

    lines = []

    def emit(s=""):
        lines.append(s)

    emit("# LLM-judge study: BD3-LM L'=16 OWT sampler grid")
    emit()
    emit(f"Judge: `{C.MODEL}` (effort={C.EFFORT}, structured outputs), "
         f"blinded configs, seed {C.SEED}.")
    emit()
    emit("**Note: ELBO is constant (<=23.52) across all 16 configs — zero "
         "discriminative signal by construction.** DUEL ppl values are from "
         "paper Table 6 (BD3-LM L'=16, OWT); confidence-threshold taus "
         "0.045/0.08/0.16/1.0 map to eval-side NFE ~140/242/514/1003.")
    emit()

    # ---------------- absolute ratings, unblinded per config
    abs_by_cfg = defaultdict(lambda: defaultdict(dict))  # name->dim->{idx:score}
    n_missing_abs = 0
    for r in abs_results:
        if r["status"] != "ok":
            n_missing_abs += 1
            continue
        _, cid, idx = r["custom_id"].split("_")
        name = mapping[cid]
        for dim in C.ABS_DIMS:
            abs_by_cfg[name][dim][int(idx)] = float(r["parsed"][dim])

    emit("## Absolute ratings (mean [95% bootstrap CI], n<=100 per config)")
    emit()
    header = ("| config | NFE | DUEL ppl | " +
              " | ".join(C.ABS_DIMS) + " | n |")
    emit(header)
    emit("|" + "---|" * (len(C.ABS_DIMS) + 4))
    order = sorted(C.CONFIGS, key=lambda nm: (-C.BUDGET_OF[nm], nm))
    stats = {}
    for name in order:
        dims = abs_by_cfg[name]
        cells = []
        for dim in C.ABS_DIMS:
            m, lo, hi = boot_ci(list(dims[dim].values()))
            stats.setdefault(name, {})[dim] = m
            cells.append(f"{m:.2f} [{lo:.2f},{hi:.2f}]")
        n = len(dims["overall"])
        emit(f"| {short_name(name)} | {C.BUDGET_OF[name]} | "
             f"{C.DUEL_PPL[name]:.2f} | " + " | ".join(cells) + f" | {n} |")
    emit()
    if n_missing_abs:
        emit(f"Missing/failed absolute ratings: {n_missing_abs}.")
        emit()

    # ---------------- metric correlations across the 16 configs
    names = list(order)
    duel = [C.DUEL_PPL[nm] for nm in names]
    judge_overall = [stats[nm]["overall"] for nm in names]
    inv = {v: k for k, v in mapping.items()}
    gen_ppl_mean, mauve = [], []
    for nm in names:
        cfg = prep["configs"][inv[nm]]
        gen_ppl_mean.append(
            float(np.mean([row["gen_ppl"] for row in cfg["rows"].values()])))
        mauve.append(cfg["mauve"])

    emit("## Metric correlations across the 16 configs (Spearman)")
    emit()
    emit("| pair | rho | perm. p (10k) |")
    emit("|---|---|---|")
    for label, a, b in [
        ("judge overall vs DUEL ppl", judge_overall, duel),
        ("gen-ppl (sampled-100 mean) vs DUEL ppl", gen_ppl_mean, duel),
        ("MAUVE vs DUEL ppl", mauve, duel),
        ("judge overall vs gen-ppl", judge_overall, gen_ppl_mean),
        ("judge overall vs MAUVE", judge_overall, mauve),
    ]:
        rho, p = perm_p(a, b)
        emit(f"| {label} | {rho:+.3f} | {p:.4f} |")
    emit()

    # ---------------- pairwise, within budget
    # unit[(budget, cidA, cidB, idx)][order] = winner cid or "tie"
    units = defaultdict(dict)
    n_missing_pair = 0
    for r in pair_results:
        if r["status"] != "ok":
            n_missing_pair += 1
            continue
        _, budget, cid_a, cid_b, idx, present = r["custom_id"].split("_")
        w = r["parsed"]["winner"]
        if w == "tie":
            winner = "tie"
        elif present == "AB":
            winner = cid_a if w == "A" else cid_b
        else:  # BA: sample A shown was cid_b
            winner = cid_b if w == "A" else cid_a
        units[(int(budget), cid_a, cid_b, int(idx))][present] = winner

    emit("## Within-budget pairwise win rates "
         "(row beats column; order-averaged, tie=0.5)")
    emit()
    fam_order = ["block-greedy", "block-left-to-right",
                 "block-probability-margin", "block-confidence-threshold"]
    for budget in sorted({b for b, *_ in units}, reverse=True):
        cfgs = sorted(
            [nm for nm in C.CONFIGS if C.BUDGET_OF[nm] == budget],
            key=lambda nm: fam_order.index(nm.rsplit("_k", 1)[0]))
        cids = [inv[nm] for nm in cfgs]
        # unit scores per unordered pair
        score = {}
        nunits = {}
        flips_agree, flips_total = 0, 0
        for (b, ca, cb, idx), verdicts in units.items():
            if b != budget:
                continue
            vals = []
            for pres, winner in verdicts.items():
                vals.append(1.0 if winner == ca
                            else 0.5 if winner == "tie" else 0.0)
            if len(verdicts) == 2:
                flips_total += 1
                v = list(verdicts.values())
                if v[0] == v[1]:
                    flips_agree += 1
            u = float(np.mean(vals))
            score[(ca, cb)] = score.get((ca, cb), 0.0) + u
            nunits[(ca, cb)] = nunits.get((ca, cb), 0) + 1
        emit(f"### Budget ~{budget} NFE")
        emit()
        emit("| | " + " | ".join(short_name(nm) for nm in cfgs) + " |")
        emit("|" + "---|" * (len(cfgs) + 1))
        for i, (ri, rnm) in enumerate(zip(cids, cfgs)):
            row = [f"**{short_name(rnm)}**"]
            for j, cj in enumerate(cids):
                if i == j:
                    row.append("—")
                    continue
                key = (ri, cj) if (ri, cj) in score else (cj, ri)
                if key not in score:
                    row.append("n/a")
                    continue
                n = nunits[key]
                s = score[key] if key == (ri, cj) else n - score[key]
                lo, hi = wilson(s, n)
                row.append(f"{s / n:.2f} [{lo:.2f},{hi:.2f}]")
            emit("| " + " | ".join(row) + " |")
        emit()
        if flips_total:
            emit(f"Flip-consistency (AB/BA verdicts agree): "
                 f"{flips_agree}/{flips_total} = "
                 f"{flips_agree / flips_total:.2f}")
            emit()
    if n_missing_pair:
        emit(f"Missing/failed pairwise judgments: {n_missing_pair}.")
        emit()

    # ---------------- length-bias check
    emit("## Length-bias check: Spearman(orig_words, overall) within config")
    emit()
    emit("| config | rho | n |")
    emit("|---|---|---|")
    for name in order:
        cfg = prep["configs"][inv[name]]
        xs, ys = [], []
        for idx, sc in abs_by_cfg[name]["overall"].items():
            row = cfg["rows"].get(str(idx))
            if row is not None:
                xs.append(row["orig_words"])
                ys.append(sc)
        rho = spearman(xs, ys) if len(xs) > 2 else float("nan")
        emit(f"| {short_name(name)} | {rho:+.3f} | {len(xs)} |")
    emit()

    text = "\n".join(lines) + "\n"
    out_path = os.path.join(C.OUT_DIR, "summary.md")
    with open(out_path, "w") as fh:
        fh.write(text)
    print(text)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
