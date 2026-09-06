"""Merged (original + ext) judge analysis for the n=1000 expansion.

Recomputes, per scale: fixed-policy pair win-rates + Wilson CIs (~500
units/pair), decided classification, metric agreement scorecard, and
primary_reason shares over decided-pair verdicts. Prints markdown-ready
blocks for per_nfe_tables.md. Run with --scale 110m|8b|both.
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict
from itertools import combinations

BASE = "/home/gt345/projects/scaling/duel/scripts/llm_judge"
EXACT = "/home/gt345/projects/scaling/exact_ll/outputs"


def wilson(s, n, z=1.96):
    p = s / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def load_results(out_dir, arm):
    rows = []
    for name in (f"{arm}_results.jsonl", f"ext_{arm}_results.jsonl"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p)]
    return rows


def name110(n):
    m = {"block-greedy": "greedy", "block-left-to-right": "l2r",
         "block-probability-margin": "margin"}
    return m.get(n.rsplit("_k", 1)[0])


def name8b(n):
    m = {"greedy": "greedy", "left_to_right": "l2r", "margin": "margin"}
    return m.get(n.rsplit("_k", 1)[0])


def pair_units(out_dir, namer):
    blinding = json.load(open(f"{out_dir}/blinding.json"))["mapping"]
    units = defaultdict(dict)          # (budget, ca, cb, idx) -> {pres: winner}
    reasons = defaultdict(lambda: defaultdict(int))  # (budget, fs) -> reason counts
    for r in load_results(out_dir, "pair"):
        if r["status"] != "ok":
            continue
        _, budget, ca, cb, idx, pres = r["custom_id"].split("_")
        na, nb = namer(blinding[ca]), namer(blinding[cb])
        if na is None or nb is None:
            continue
        w = r["parsed"]["winner"]
        winner = ("tie" if w == "tie"
                  else (ca if w == "A" else cb) if pres == "AB"
                  else (cb if w == "A" else ca))
        units[(int(budget), ca, cb, int(idx))][pres] = winner
        if w != "tie":
            reasons[(int(budget), frozenset((na, nb)))][
                r["parsed"].get("primary_reason", "other")] += 1
    score, nunits = defaultdict(float), defaultdict(int)
    for (b, ca, cb, idx), verdicts in units.items():
        na, nb = namer(blinding[ca]), namer(blinding[cb])
        vals = [1.0 if w == ca else 0.5 if w == "tie" else 0.0
                for w in verdicts.values()]
        key = (b, na, nb)
        score[key] += sum(vals) / len(vals)
        nunits[key] += 1
    return score, nunits, reasons


def metric_verdict(vals, a, b, i, hb):
    va, vb = vals[a][i], vals[b][i]
    if va == vb:
        return None
    return (va > vb) if hb else (va < vb)


def duel_ppl_8b_n1000():
    out = {}
    stems = {}
    for f in glob.glob(f"{EXACT}/sampling_grid_llada8b_prefix/scores/"
                       "*_duel*.jsonl"):
        stem = os.path.basename(f).split("_duel")[0]
        stems.setdefault(stem, []).append(f)
    for stem, files in stems.items():
        rows = {}
        for f in files:
            for line in open(f):
                if not line.endswith("\n"):
                    continue
                r = json.loads(line)
                if r["full_length"]:
                    rows[r["prefix_index"]] = r
        tot = sum(r["nll_nats"] for r in rows.values())
        cnt = sum(r["n_tokens"] for r in rows.values())
        strat, k = stem.rsplit("_bs32_k", 1)
        k = k.split("_")[0]
        out[(strat.replace("left_to_right", "l2r")
             .replace("greedy", "greedy").replace("margin", "margin"),
             int(k))] = (math.exp(tot / cnt), len(rows))
    return out


def genppl_8b_n1000():
    out = {}
    for f in glob.glob(f"{EXACT}/sampling_grid_llada8b_prefix_nuc098_full5000/"
                       "scores_nuc098/*_gpt2_n1000.jsonl"):
        rows = [json.loads(l) for l in open(f)]
        tot = sum(r["nll_sum"] for r in rows)
        cnt = sum(r["n_tokens"] for r in rows)
        stem = os.path.basename(f).replace("_gpt2_n1000.jsonl", "")
        strat, k = stem.rsplit("_bs32_k", 1)
        k = k.split("_")[0]
        out[(strat.replace("left_to_right", "l2r"), int(k))] = \
            math.exp(tot / cnt)
    return out


# 110M metric values (unchanged: already n=1000 / paper Table 6)
M110 = {
    1024: {"greedy": (22.04, 14.4, 0.257), "margin": (22.35, 16.1, 0.203),
           "l2r": (21.46, 15.4, 0.249)},
    512: {"margin": (32.3, 76.8, 0.035), "greedy": (34.7, 78.3, 0.030),
          "l2r": (44.5, 53.7, 0.009)},
    256: {"greedy": (67.2, 192.7, 0.0085), "margin": (57.7, 202.0, 0.0078),
          "l2r": (109.0, 137.7, 0.0055)},
    128: {"margin": (141.3, 370.5, 0.005), "greedy": (165.5, 380.2, 0.005),
          "l2r": (240.3, 346.8, 0.005)},
}
MAUVE_8B = {  # n=1000 column, summary_mauve_full.md
    (256, "l2r"): 0.3979, (128, "l2r"): 0.0145, (64, "l2r"): 0.0083,
    (32, "l2r"): 0.0081,
    (256, "greedy"): 0.4017, (128, "greedy"): 0.0263, (64, "greedy"): 0.0127,
    (32, "greedy"): 0.0098,
    (256, "margin"): 0.4274, (128, "margin"): 0.0425, (64, "margin"): 0.0122,
    (32, "margin"): 0.0097,
}
K_OF_BUDGET_8B = {256: 1, 128: 2, 64: 4, 32: 8}


def run_scale(scale):
    if scale == "110m":
        out_dir, namer = f"{BASE}/out", name110
        metrics = {b: {s: v for s, v in d.items()} for b, d in M110.items()}
    else:
        out_dir, namer = f"{BASE}/out8b_nuc", name8b
        duel = duel_ppl_8b_n1000()
        gen = genppl_8b_n1000()
        metrics = {}
        for b, k in K_OF_BUDGET_8B.items():
            metrics[b] = {}
            for s in ("greedy", "l2r", "margin"):
                key = (s if s != "l2r" else "left_to_right", k)
                d = duel.get((s, k)) or duel.get(key)
                g = gen.get((s, k)) or gen.get(key)
                metrics[b][s] = (d[0] if d else float("nan"),
                                 g if g else float("nan"),
                                 MAUVE_8B[(b, s)])
        print("8B DUEL n per config:",
              {k: v[1] for k, v in sorted(duel.items())})

    score, nunits, reasons = pair_units(out_dir, namer)
    tally = {"DUEL": [0, 0], "gen-ppl": [0, 0], "MAUVE": [0, 0]}
    lines = []
    decided_keys = []
    for (b, a, c) in sorted(score, key=lambda k: (-k[0], k[1], k[2])):
        n = nunits[(b, a, c)]
        p = score[(b, a, c)] / n
        lo, hi = wilson(score[(b, a, c)], n)
        decided = lo > 0.5 or hi < 0.5
        winner, loser = (a, c) if p > 0.5 else (c, a)
        marks = []
        for mi, (mn, hb) in enumerate([("DUEL", False), ("gen-ppl", False),
                                       ("MAUVE", True)]):
            ok = metric_verdict(metrics[b], winner, loser, mi, hb)
            if decided:
                tally[mn][1] += 1
                if ok:
                    tally[mn][0] += 1
            marks.append("✓" if ok else ("=" if ok is None else "✗"))
        if decided:
            decided_keys.append((b, frozenset((a, c))))
        status = "**decided**" if decided else "tie"
        w = f"**{winner} > {loser}**" if decided else f"{winner} > {loser}"
        mm = [m if decided else f"({m})" for m in marks]
        lines.append(f"| {b} | {w} | {p:.2f} [{lo:.3f}, {hi:.3f}] n={n} "
                     f"| {status} | {mm[0]} | {mm[1]} | {mm[2]} |")
    print(f"\n===== {scale} pair table =====")
    print("| NFE | Pair (judge direction) | Win-rate [95% CI] | Status "
          "| DUEL | gen-ppl | MAUVE |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(lines))
    print(f"\n{scale} decided scorecard: " + "  ".join(
        f"{k} {v[0]}/{v[1]}" for k, v in tally.items()))

    rsum = defaultdict(int)
    tot = 0
    for key in decided_keys:
        for reason, cnt in reasons.get(key, {}).items():
            rsum[reason] += cnt
            tot += cnt
    print(f"{scale} decided-verdict primary_reason (n={tot}): " + ", ".join(
        f"{k} {100 * v / tot:.0f}%" for k, v in
        sorted(rsum.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="both", choices=["110m", "8b", "both"])
    a = ap.parse_args()
    for s in (["110m", "8b"] if a.scale == "both" else [a.scale]):
        run_scale(s)
