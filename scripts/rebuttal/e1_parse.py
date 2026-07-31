#!/usr/bin/env python3
"""Collect E1 (R2 W2/Q1) cost-arm results into one TSV/JSON.

Reads ${E1_ROOT}/<arm>/{run.log,mem.csv,meta.txt} written by
scripts/rebuttal/_e1_common.sh:run_arm and emits, per arm:

  wall_s        wall-clock of the validate() loop only (first->last tqdm tick)
  s_per_batch   mean per-batch time, sigma, and relative spread
  peak_mem_mib  max of the 2 Hz nvidia-smi poll, minus the pre-run baseline
  ppl           val/exact_ppl (DUEL) or val/ppl (ELBO) from Lightning's table
  fwd_reported  val/num_decoding_steps as logged (UNDERCOUNTS -- see below)

Forward-count correction. exact_likelihood.compute_exact_loglikelihood_cached
increments `steps` only inside the within-block unmask loop (:174); the
block-commit forward at :180 sits OUTSIDE that loop and is never counted. The
true per-sequence forward count on the cached (BD3-LM) path is therefore
    num_blocks * (ceil(L'/k) + 1)     not     num_blocks * ceil(L'/k).
The uncached (MDLM) path has no commit forward, so there `steps` is exact.

Usage: python scripts/rebuttal/e1_parse.py [E1_ROOT] > results.tsv
"""
import glob
import json
import math
import os
import re
import subprocess
import sys

TQDM = re.compile(r"Validation DataLoader 0:.*?\|\s*(\d+)/(\d+)")
METRIC = re.compile(r"^\s*│\s*([A-Za-z0-9_/.\- ]+?)\s*│\s*([-\d.eE+]+)\s*│")
KV = re.compile(r"^([a-z_]+)=(.*)$")


NESTED = re.compile(r'nested_key: "([^"]+)"\s*\n\s*value_json: "([^"]*)"')


def _wandb_history(d):
    """{metric: [values]} decoded from the offline wandb datastore.

    val/num_decoding_steps is logged on_step=True, on_epoch=False,
    prog_bar=False (diffusion.py:841-847), so it appears in NO console stream:
    not the progress bar, not Lightning's end-of-run table. This is the only
    place it survives.
    """
    cache = os.path.join(d, "wandb_view.txt")
    if not os.path.exists(cache):
        runs = sorted(glob.glob(os.path.join(d, "hydra", "wandb", "*run-*")))
        runs = [r for r in runs if os.path.isdir(r) and "latest" not in r]
        if not runs:
            return {}
        try:
            txt = subprocess.run(
                ["wandb", "sync", "--view", "--verbose", runs[-1]],
                capture_output=True, text=True, timeout=600).stdout
        except Exception:
            return {}
        with open(cache, "w") as f:
            f.write(txt)
    txt = open(cache, errors="replace").read()
    hist = {}
    for key, val in NESTED.findall(txt):
        try:
            hist.setdefault(key, []).append(float(val))
        except ValueError:
            pass
    return hist


def parse_arm(d):
    out = {"arm": os.path.basename(d)}
    meta = os.path.join(d, "meta.txt")
    if os.path.exists(meta):
        for line in open(meta):
            m = KV.match(line.strip())
            if m:
                out[m.group(1)] = m.group(2)

    # --- peak memory -------------------------------------------------------
    mem = os.path.join(d, "mem.csv")
    if os.path.exists(mem):
        vals = [int(x) for x in open(mem).read().split() if x.strip().isdigit()]
        if vals:
            out["peak_mem_mib"] = max(vals)
            out["n_mem_samples"] = len(vals)
    try:
        out["baseline_mem_mib"] = int(open(os.path.join(d, "mem_baseline.txt")).read().strip())
    except Exception:
        out["baseline_mem_mib"] = None
    if out.get("peak_mem_mib") is not None and out.get("baseline_mem_mib") is not None:
        out["peak_mem_mib_net"] = out["peak_mem_mib"] - out["baseline_mem_mib"]

    # --- timing + metrics --------------------------------------------------
    # val/num_decoding_steps is logged on_step only and reaches no console
    # stream; offline wandb's summary is the only place it survives.
    # hydra chdirs into ${arm}/hydra, and WandbLogger's save_dir='.' wins over
    # WANDB_DIR, so the run lands at ${arm}/hydra/wandb/offline-run-*/files/.
    # main.py never calls run.finish(), so wandb-summary.json is never written;
    # the history only exists inside the binary .wandb datastore. `wandb sync
    # --view --verbose` decodes it, and the result is cached per arm because it
    # is slow.
    hist = _wandb_history(d)
    for k in ("val/num_decoding_steps", "val/exact_ppl_step", "val/ppl_step"):
        if k in hist:
            vals = hist[k]
            out["wb_" + k.replace("/", "_")] = vals[-1]
            if k == "val/num_decoding_steps" and len(set(vals)) > 1:
                out.setdefault("flags", []).append(
                    f"num_decoding_steps varies across batches: {sorted(set(vals))}")

    log = os.path.join(d, "run.log")
    first_ts = {}          # tqdm counter value -> earliest wall-clock seen
    table_ts = None        # first line of Lightning's final metric table
    total = None
    metrics = {}
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            sp = line.find(" ")
            if sp <= 0:
                continue
            try:
                ts = float(line[:sp])
            except ValueError:
                continue
            rest = line[sp + 1:]
            m = TQDM.search(rest)
            if m:
                i, n = int(m.group(1)), int(m.group(2))
                total = n
                if i not in first_ts:
                    first_ts[i] = ts
            if table_ts is None and rest.startswith("┏"):
                table_ts = ts
            m2 = METRIC.match(rest)
            if m2:
                try:
                    metrics[m2.group(1).strip()] = float(m2.group(2))
                except ValueError:
                    pass
        txt = open(log, errors="replace").read()
        for pat in ("Traceback", "CUDA out of memory", "ncclCommInit", "Error"):
            if pat in txt:
                out.setdefault("flags", []).append(pat)

    if len(first_ts) >= 2:
        ks = sorted(first_ts)
        deltas = [first_ts[ks[j + 1]] - first_ts[ks[j]] for j in range(len(ks) - 1)]
        # Drop the FIRST and LAST frame. The first carries dataloader spin-up,
        # cuDNN autotune and lazy CUDA context init -- on the ELBO it has been
        # measured at 72 s against a 0.67 s steady state, a 100x outlier that
        # would be ~25% of a 4-batch mean and inflates sigma ~20x. The last
        # frame absorbs the epoch-end sync. Everything reported as s/seq or
        # s/batch comes from the INTERIOR frames only; the full series is kept
        # in batch_times_s so the exclusion is auditable.
        out["batch_times_s"] = [round(x, 3) for x in deltas]
        # Uniform warm-up rule, applied identically to every arm:
        #   1. drop the LAST frame (it only holds the epoch-end sync, ~0.01 s);
        #   2. then drop LEADING frames exceeding 1.25x the median of what is
        #      left, repeatedly.
        # A fixed "drop frame 0" is not enough. sdpa warms up in one frame
        # ([3.75, 1.81, 1.82, ...]) but FlexAttention needs two, because the
        # first pays a ~55 s torch.compile and the second is still cold
        # ([54.69, 3.21, 1.62, 1.62, ...]). Dropping one frame each would score
        # flex at 1.88 +- 0.65 (cv 34%) and make it look slower than sdpa, when
        # its steady state is actually ~11% FASTER. The criterion is stated
        # rather than tuned per arm, and batch_times_s keeps the raw series so
        # the exclusion can be audited.
        body = deltas[:-1] if len(deltas) >= 3 else deltas
        dropped = 0
        while len(body) > 2:
            med = sorted(body[1:])[len(body[1:]) // 2]
            if body[0] > 1.25 * med:
                body = body[1:]
                dropped += 1
            else:
                break
        core = body
        mean = sum(core) / len(core)
        var = sum((x - mean) ** 2 for x in core) / max(len(core) - 1, 1)
        out["n_batches_timed"] = len(core)
        out["warmup_frames_dropped"] = dropped
        out["timed_frames"] = f"{dropped}..{len(deltas) - 2}"
        out["first_batch_s"] = round(deltas[0], 3)
        out["n_val_batches"] = total

        # AUTHORITATIVE wall-clock. The tqdm counter advances when the Python
        # loop returns, but CUDA is async: on cheap arms (ELBO) the CPU races
        # several batches ahead and the per-batch deltas collapse to ~10 ms,
        # which is not a measurement of anything. Bracketing from the "0/N" tick
        # to Lightning's metric table -- which forces the epoch-end reduction and
        # therefore a full device sync -- captures all GPU work exactly once.
        # DUEL arms sync every step anyway (exact_likelihood.py:141,159 call
        # .item()), so there the per-batch deltas ARE real and the sigma gate is
        # meaningful; on ELBO arms treat cpu_spread_pct as void.
        if table_ts is not None:
            out["wall_val_s"] = round(table_ts - first_ts[ks[0]], 3)
            if total:
                out["s_per_batch_incl_warmup"] = round(
                    (table_ts - first_ts[ks[0]]) / total, 3)
        # Steady-state per-batch cost = the interior-frame mean. This, not the
        # total, is what gets extrapolated to the full 110,480-sequence split.
        out["s_per_batch"] = round(mean, 3)
        out["s_per_batch_sigma"] = round(math.sqrt(var), 3)
        out["cv_pct"] = round(100 * math.sqrt(var) / mean, 2) if mean else None

    out["ppl"] = metrics.get("val/exact_ppl", metrics.get("val/ppl"))
    out["val_ppl"] = metrics.get("val/ppl")
    out["val_exact_ppl"] = metrics.get("val/exact_ppl")
    out["val_nll"] = metrics.get("val/nll")
    out["fwd_reported"] = metrics.get("val/num_decoding_steps")
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "E1_ROOT", os.path.join(os.getcwd(), "logs", "e1_cost"))
    arms = sorted(
        os.path.join(root, x) for x in os.listdir(root)
        if os.path.isdir(os.path.join(root, x)))
    rows = [parse_arm(a) for a in arms]
    cols = ["arm", "exit_code", "host", "cotenants", "n_val_batches", "timed_frames",
            "n_batches_timed", "first_batch_s", "s_per_batch", "s_per_batch_sigma",
            "cv_pct", "s_per_batch_incl_warmup", "wall_val_s", "peak_mem_mib",
            "baseline_mem_mib", "peak_mem_mib_net", "wb_val_num_decoding_steps", "wb_val_exact_ppl_step",
            "ppl", "val_ppl", "val_exact_ppl", "val_nll", "batch_times_s", "flags"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in cols))
    with open(os.path.join(root, "e1_results.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)


if __name__ == "__main__":
    main()
