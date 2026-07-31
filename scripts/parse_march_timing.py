#!/usr/bin/env python3
"""Parse the BD3-LM OpenWebText block_size=16 timing sweep (E0.1-prelim).

Reads ``logs/bd3lm_openwebtext-split_block_size16_*.log``, extracts wall time,
per-iteration time series, config, and final perplexity for each arm, and emits
a normalized seconds-per-sequence table.

Pure log parsing: no GPU, no network, no jobs submitted.

Usage:
    python scripts/parse_march_timing.py [--logdir LOGS] [--csv OUT.csv]
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import statistics
import sys

# --------------------------------------------------------------------------
# Regexes
# --------------------------------------------------------------------------

# tqdm progress frame: "<done>/<total> [<h:>?<mm>:<ss><" (elapsed side of the
# "elapsed<remaining" pair).
TQDM_RE = re.compile(r"(\d+)/(\d+) \[(\d+:)?(\d+):(\d+)<")

# Hydra config dump lines are box-drawn; strip the decoration and match "key: value".
CFG_RES = {
    "eval_batch_size": re.compile(r"eval_batch_size:\s*(\d+)"),
    "devices": re.compile(r"devices:\s*(\d+)"),
    "attn_backend": re.compile(r"attn_backend:\s*(\S+)"),
    "strategy": re.compile(r"exact_ll_strategy:\s*(\S+)"),
    "k": re.compile(r"exact_ll_k:\s*([0-9.]+)"),
    "block_size": re.compile(r"^\s*│?\s*└──\s*block_size\s*$"),  # handled separately
    "length": re.compile(r"\blength:\s*(\d+)\s*$", re.M),
    "mode": re.compile(r"└──\s*(elbo_ppl|exact_ll|ppl_eval|\S*ppl\S*)\s*$", re.M),
    "cross_attn": re.compile(r"cross_attn:\s*(\S+)"),
    "kv_cache": re.compile(r"exact_ll_use_kv_cache:\s*(\S+)"),
}

# Final metric table values.
PPL_RES = [
    ("val/exact_ppl", re.compile(r"val/exact_ppl\s*│\s*([0-9.eE+-]+)")),
    ("val/ppl", re.compile(r"val/ppl\s*│\s*([0-9.eE+-]+)")),
]
NLL_RES = [
    ("val/exact_nll", re.compile(r"val/exact_nll\s*│\s*([0-9.eE+-]+)")),
    ("val/nll", re.compile(r"val/nll\s*│\s*([0-9.eE+-]+)")),
]
DECODE_RE = re.compile(r"val/num_decoding_steps\s*│\s*([0-9.eE+-]+)")

# tau is only recoverable from the filename for the confidence-threshold arms
# (the Hydra dump reuses exact_ll_k, which is set to 0 for those runs).
TAU_RE = re.compile(r"confidence_threshold_([0-9.]+)\.log$")

# Published Figure 3/4 perplexities for the confidence-threshold arms.
PUBLISHED_PPL = {
    ("block_confidence_threshold", "0.05"): 226.97,
    ("block_confidence_threshold", "0.07"): 116.83,
    ("block_confidence_threshold", "0.15"): 43.48,
    ("block_confidence_threshold", "0.99"): 22.05,
}
GATE_TOL = 1.0

# Values handed to us to verify independently (arm_key -> claimed wall seconds).
CLAIMED_WALL = {
    "elbo": 159,
    "block_greedy k=8": 333,
    "block_greedy k=4": 565,
    "block_greedy k=2": 1026,
    "block_greedy k=1": 1950,
    "block_left_to_right k=1": 1827,
    "block_probability_margin k=1": 3019,
    "block_confidence_threshold tau=0.99": 11950,
}


def _seconds(h, mm, ss) -> int:
    return int(h[:-1]) * 3600 + int(mm) * 60 + int(ss) if h else int(mm) * 60 + int(ss)


def parse_tqdm_series(text: str):
    """Return (total_iters, [(iter, cumulative_seconds), ...]) deduped, monotone.

    tqdm rewrites the same line with \\r, so the raw text holds every frame.
    We keep the *last* frame emitted for each iteration index (tqdm sometimes
    re-prints the final frame) and require the iteration index to be monotone.
    """
    seen = {}
    order = []
    total = None
    for m in TQDM_RE.finditer(text):
        it = int(m.group(1))
        total = int(m.group(2))
        secs = _seconds(m.group(3), m.group(4), m.group(5))
        if it not in seen:
            order.append(it)
        seen[it] = secs
    series = [(it, seen[it]) for it in sorted(order)]
    return total, series


def per_iteration_times(series):
    """Differences of the cumulative-elapsed series -> per-iteration seconds.

    Returns ``(warmup_s, first_batch_s, deltas, last_delta)``.

    Two frames are deliberately NOT per-batch costs and are excluded from
    ``deltas``:

      * frame 0 -> frame 1. tqdm emits a "0/N [00:00<?" frame at t=0 before the
        first batch runs, so this delta is (warm-up + CUDA graph/compile +
        first batch), not a steady-state batch. Reported as ``first_batch_s``.
      * the final frame. Lightning's epoch-end metric reduction (and, for
        `flex` attention, a recompile) lands inside the last tqdm update, and
        the last batch is usually *partial* -- the validation set is 1075
        sequences, so at bs=64 the 17th batch holds 51, not 64. Reported as
        ``last_delta``.
    """
    if len(series) < 3:
        return None, None, [], None
    warmup = series[0][1]  # elapsed at frame 0; ~0 by construction
    first_batch = series[1][1] - series[0][1]
    deltas = []
    for (i0, t0), (i1, t1) in zip(series[1:], series[2:]):
        n = i1 - i0
        if n <= 0:
            continue
        deltas.extend([(t1 - t0) / n] * n)
    last = deltas.pop() if deltas else None
    return warmup, first_batch, deltas, last


def find_first(text, patterns):
    for name, rx in patterns:
        m = rx.search(text)
        if m:
            return name, float(m.group(1))
    return None, None


def arm_key(cfg):
    if cfg["is_elbo"]:
        return "elbo"
    if cfg["strategy"] == "block_confidence_threshold":
        return f"block_confidence_threshold tau={cfg['tau']}"
    return f"{cfg['strategy']} k={cfg['k']}"


def analytic_forwards_per_seq(cfg):
    """Forwards per sequence, where determinate from the config alone.

    BD3-LM block_size=16, L=1024 -> 64 blocks.
      * ELBO (mode=elbo_ppl, K=1 MC sample): one network forward per sequence.
      * DUEL with a fixed unmask width k: 16/k steps per block over 64 blocks
        = 1024/k forwards per sequence.
      * DUEL confidence-threshold: the number of unmasked tokens per step is
        data-dependent, so forwards/seq is NOT determinate from the config. It
        is logged as `val/num_decoding_steps`, but that metric goes to W&B only
        (duel/diffusion.py:838-847) and never reaches stdout -- see
        scripts/extract_duel_ppl.py for the W&B-API path.
    """
    if cfg["is_elbo"]:
        return 1.0
    if cfg["strategy"] == "block_confidence_threshold":
        return None
    k = cfg["k"]
    if not k:
        return None
    return cfg["length"] / k


def parse_log(path):
    with open(path, "r", errors="replace") as fh:
        raw = fh.read()
    if not raw.strip():
        return {"path": path, "empty": True}

    text = raw.replace("\r", "\n")

    def grab(name, cast=str, default=None):
        m = CFG_RES[name].search(text)
        return cast(m.group(1)) if m else default

    tau_m = TAU_RE.search(os.path.basename(path))
    is_elbo = "elbo_ppl" in text and "exact_ppl" not in text

    cfg = {
        "eval_batch_size": grab("eval_batch_size", int, 1),
        "devices": grab("devices", int, 1),
        "attn_backend": grab("attn_backend", str, "?"),
        "strategy": grab("strategy", str, "?"),
        "k": grab("k", float, None),
        "length": grab("length", int, 1024),
        "kv_cache": grab("kv_cache", str, "?"),
        "tau": tau_m.group(1) if tau_m else None,
        "is_elbo": is_elbo,
    }
    if is_elbo:
        cfg["strategy"] = "elbo"
        cfg["k"] = 1.0
    elif cfg["k"] is not None:
        cfg["k"] = int(cfg["k"])

    total, series = parse_tqdm_series(text)
    if not series:
        return {"path": path, "empty": True, "reason": "no tqdm frames"}

    iters_done, wall = series[-1]
    complete = (total is not None and iters_done == total)

    warmup, first_batch, deltas, last_delta = per_iteration_times(series)
    mean_it = statistics.mean(deltas) if deltas else float("nan")
    sd_it = statistics.pstdev(deltas) if len(deltas) > 1 else float("nan")

    ppl_key, ppl = find_first(text, PPL_RES)
    nll_key, nll = find_first(text, NLL_RES)
    dm = DECODE_RE.search(text)

    nseq = iters_done * cfg["eval_batch_size"] * cfg["devices"]

    return {
        "path": path,
        "empty": False,
        "cfg": cfg,
        "arm": arm_key(cfg),
        "iters_done": iters_done,
        "iters_total": total,
        "complete": complete,
        "wall": wall,
        "nseq": nseq,
        "s_per_seq": wall / nseq if nseq else float("nan"),
        "warmup_frame_s": warmup,
        "first_batch_s": first_batch,
        "last_delta_s": last_delta,
        "n_deltas": len(deltas),
        "mean_iter": mean_it,
        "sd_iter": sd_it,
        "min_iter": min(deltas) if deltas else float("nan"),
        "max_iter": max(deltas) if deltas else float("nan"),
        "ppl": ppl,
        "ppl_key": ppl_key,
        "nll": nll,
        "nll_key": nll_key,
        "num_decoding_steps": float(dm.group(1)) if dm else None,
        "fwd_per_seq": analytic_forwards_per_seq(cfg),
        "fwd_len": cfg["length"],
    }


def fmt(x, nd=2, width=None, na="n/a"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        s = na
    else:
        s = f"{x:,.{nd}f}" if nd else f"{x:,.0f}"
    return s.rjust(width) if width else s


def main():
    ap = argparse.ArgumentParser()
    default_logs = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    ap.add_argument("--logdir", default=default_logs)
    ap.add_argument("--pattern",
                    default="bd3lm_openwebtext-split_block_size16_*.log")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--extra-logs", nargs="*",
                    default=["mdlm_owt_block_size4-exact_ll.log"],
                    help="extra logs to report per-iteration stats for only")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.logdir, args.pattern)))
    if not paths:
        sys.exit(f"no logs matched {args.pattern} under {args.logdir}")

    rows, skipped = [], []
    for p in paths:
        r = parse_log(p)
        if r.get("empty"):
            skipped.append((p, r.get("reason", "zero-byte / no output")))
        else:
            rows.append(r)

    # ELBO baseline for the x-ELBO column.
    elbo = next((r for r in rows if r["cfg"]["is_elbo"]), None)
    base = elbo["s_per_seq"] if elbo else None

    def sort_key(r):
        order = {"elbo": 0, "block_greedy": 1, "block_left_to_right": 2,
                 "block_probability_margin": 3, "block_confidence_threshold": 4}
        s = r["cfg"]["strategy"]
        sec = float(r["cfg"]["tau"]) if r["cfg"]["tau"] else -(r["cfg"]["k"] or 0)
        return (order.get(s, 9), sec)

    rows.sort(key=sort_key)

    print("=" * 116)
    print("E0.1-prelim -- BD3-LM OpenWebText, block_size=16, L=1024, single GPU")
    print(f"logs: {args.logdir}/{args.pattern}   ({len(rows)} parsed, "
          f"{len(skipped)} skipped)")
    print("=" * 116)
    if skipped:
        print("\nSKIPPED LOGS")
        for p, why in skipped:
            print(f"  {os.path.basename(p):<70} {why}")

    hdr = (f"\n{'arm':<38}{'bs':>4}{'dev':>4}{'attn':>7}{'iters':>8}"
           f"{'fwd/seq':>9}{'fwdlen':>8}{'wall(s)':>10}{'nseq':>7}"
           f"{'s/seq':>9}{'ppl':>10}{'xELBO':>8}")
    print(hdr)
    print("-" * 116)
    for r in rows:
        c = r["cfg"]
        x = r["s_per_seq"] / base if base else None
        it = f"{r['iters_done']}/{r['iters_total']}"
        if not r["complete"]:
            it += "!"
        print(f"{r['arm']:<38}{c['eval_batch_size']:>4}{c['devices']:>4}"
              f"{c['attn_backend']:>7}{it:>8}"
              f"{fmt(r['fwd_per_seq'], 0, 9)}{r['fwd_len']:>8}"
              f"{fmt(r['wall'], 0, 10)}{r['nseq']:>7}"
              f"{fmt(r['s_per_seq'], 3, 9)}{fmt(r['ppl'], 2, 10)}"
              f"{fmt(x, 1, 8)}")

    # ---------------- per-iteration stability ----------------
    print("\n" + "=" * 116)
    print("PER-ITERATION TIME SERIES  (consecutive tqdm cumulative-elapsed "
          "frames; 1 s clock resolution)")
    print("  Steady-state batches only: the first batch (warm-up/compile) and "
          "the last batch (partial batch +")
    print("  epoch-end metric reduction) are reported separately, not folded "
          "into mean/sd.")
    print("=" * 116)
    print(f"{'arm':<38}{'n':>5}{'batch1':>9}{'mean':>10}{'sd':>9}{'cv%':>8}"
          f"{'min':>8}{'max':>8}{'lastbatch':>11}")
    print("-" * 116)
    for r in rows:
        cv = 100 * r["sd_iter"] / r["mean_iter"] if r["mean_iter"] else float("nan")
        print(f"{r['arm']:<38}{r['n_deltas']:>5}{fmt(r['first_batch_s'],1,9)}"
              f"{fmt(r['mean_iter'],2,10)}{fmt(r['sd_iter'],2,9)}"
              f"{fmt(cv,1,8)}{fmt(r['min_iter'],1,8)}{fmt(r['max_iter'],1,8)}"
              f"{fmt(r['last_delta_s'],1,11)}")

    # ---------------- gate ----------------
    print("\n" + "=" * 116)
    print("GATE: confidence-threshold perplexities vs published Figure 3/4 "
          f"(tolerance +/-{GATE_TOL} ppl)")
    print("=" * 116)
    print(f"{'arm':<38}{'measured':>12}{'published':>12}{'delta':>10}  verdict")
    print("-" * 116)
    any_fail = False
    for r in rows:
        key = (r["cfg"]["strategy"], r["cfg"]["tau"])
        if key not in PUBLISHED_PPL:
            continue
        pub = PUBLISHED_PPL[key]
        d = r["ppl"] - pub
        ok = abs(d) <= GATE_TOL
        any_fail |= not ok
        print(f"{r['arm']:<38}{fmt(r['ppl'],2,12)}{pub:>12.2f}{d:>+10.2f}  "
              f"{'PASS' if ok else '*** FAIL -- OFF BY MORE THAN 1 PPL ***'}")
    missing = [k for k in PUBLISHED_PPL
               if not any((r["cfg"]["strategy"], r["cfg"]["tau"]) == k for r in rows)]
    for k in missing:
        any_fail = True
        print(f"{k[0]+' tau='+k[1]:<38}{'MISSING':>12}{PUBLISHED_PPL[k]:>12.2f}"
              f"{'--':>10}  *** NO USABLE LOG ***")
    print("\nGATE RESULT: " + ("FAILED" if any_fail else "PASSED -- all "
                               "confidence-threshold arms within +/-1 ppl"))

    # ---------------- claimed-value verification ----------------
    print("\n" + "=" * 116)
    print("VERIFICATION of the wall-clock values supplied to this analysis")
    print("=" * 116)
    print(f"{'arm':<38}{'claimed(s)':>12}{'measured(s)':>13}{'delta':>10}  verdict")
    print("-" * 116)
    by_arm = {r["arm"]: r for r in rows}
    for arm, claimed in CLAIMED_WALL.items():
        r = by_arm.get(arm)
        if r is None:
            print(f"{arm:<38}{claimed:>12}{'--':>13}{'--':>10}  NO LOG")
            continue
        d = r["wall"] - claimed
        print(f"{arm:<38}{claimed:>12}{r['wall']:>13}{d:>+10}  "
              f"{'CONFIRMED' if d == 0 else 'MISMATCH'}")

    # ------- cross-check: the "235 s/iter over 115 iterations" claim -------
    print("\n" + "=" * 116)
    print("CROSS-CHECK: the reported '235 s/iter +/- <1 s over 115 iterations' "
          "figure")
    print("=" * 116)
    for name in args.extra_logs:
        p = os.path.join(args.logdir, name)
        if not os.path.exists(p):
            print(f"  {name}: NOT FOUND")
            continue
        r = parse_log(p)
        if r.get("empty"):
            print(f"  {name}: empty")
            continue
        c = r["cfg"]
        print(f"  {name}")
        print(f"    config      : bs={c['eval_batch_size']} devices="
              f"{c['devices']} attn={c['attn_backend']} "
              f"strategy={c['strategy']} k={c['k']} L={c['length']}")
        print(f"    progress    : {r['iters_done']}/{r['iters_total']} "
              f"iterations -- "
              f"{'COMPLETE' if r['complete'] else 'INCOMPLETE (run did not finish)'}")
        print(f"    wall        : {r['wall']:,} s "
              f"({r['wall']/3600:.2f} h)")
        print(f"    naive s/iter: {r['wall']/r['iters_done']:.1f} "
              "(wall / iters_done)")
        print(f"    steady-state: mean {r['mean_iter']:.2f} s  sd "
              f"{r['sd_iter']:.2f} s  (n={r['n_deltas']}, "
              f"cv={100*r['sd_iter']/r['mean_iter']:.1f}%)")
        print(f"    min/max     : {r['min_iter']:.1f} / {r['max_iter']:.1f} s")
        print(f"    first batch : {r['first_batch_s']:.1f} s")

    # ---------------- caveats ----------------
    print("\n" + "=" * 116)
    print("CAVEATS")
    print("=" * 116)
    nseqs = sorted({r["nseq"] for r in rows})
    print(f"  * nseq = iters * eval_batch_size * devices gives {nseqs}. The bs=1 "
          "arms run 1075 iterations,")
    print("    so the true validation set is 1075 sequences; the batched arms' "
          "final batch is partial and")
    print("    the formula over-counts them by 13 sequences (+1.2% on nseq, "
          "-1.2% on s/seq). s/seq for the")
    print("    bs=1 arms is exact; for bs=16/64 arms it is a 1.2% "
          "under-estimate.")
    print("  * val/num_decoding_steps is logged to W&B only "
          "(duel/diffusion.py:838-847) and never printed,")
    print("    so forwards/seq for the confidence-threshold arms is not "
          "recoverable from these logs.")
    print("  * tqdm elapsed has 1 s resolution, so sd on sub-second iterations "
          "is quantization-dominated.")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["arm", "strategy", "k", "tau", "eval_batch_size",
                        "devices", "attn_backend", "iters_done", "iters_total",
                        "fwd_per_seq", "fwd_len", "wall_s", "nseq", "s_per_seq",
                        "ppl", "nll", "x_elbo", "n_iter_samples",
                        "mean_iter_s", "sd_iter_s", "first_batch_s",
                        "last_batch_s", "log"])
            for r in rows:
                c = r["cfg"]
                w.writerow([r["arm"], c["strategy"], c["k"], c["tau"],
                            c["eval_batch_size"], c["devices"],
                            c["attn_backend"], r["iters_done"],
                            r["iters_total"], r["fwd_per_seq"], r["fwd_len"],
                            r["wall"], r["nseq"], r["s_per_seq"], r["ppl"],
                            r["nll"],
                            r["s_per_seq"] / base if base else "",
                            r["n_deltas"], r["mean_iter"], r["sd_iter"],
                            r["first_batch_s"], r["last_delta_s"],
                            os.path.basename(r["path"])])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
