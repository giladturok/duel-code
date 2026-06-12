"""Parse DUEL-PPL eval logs and emit (strategy, k, NFE, val/exact_ppl) rows.

Reads `<logs_dir>/bd3lm_openwebtext-split_block_size16_exact_ll_<strategy>_<k>.log`
files produced by `scripts/sampler_comparison/duel_ppl.sh` and extracts the
final `val/exact_ppl` printed at the end of validation.

For integer-k strategies, NFE is constant per batch (1024/k). For
`block_confidence_threshold`, NFE varies per batch and is logged to W&B as
`val/num_decoding_steps` (one value per validation batch). The mean across the
validation pass is fetched from W&B; the run URL is parsed out of the log file.

Usage:
    python scripts/extract_duel_ppl.py [logs_dir]      # uses W&B for thresholds
    python scripts/extract_duel_ppl.py --no-wandb      # skip W&B, leave NFE blank
"""
import argparse
import re
import sys
from pathlib import Path

PPL_RE = re.compile(r"val/exact_ppl\s*│\s*(?P<ppl>[0-9.]+)")
# wandb prints the run URL with ANSI color codes; tolerate them.
WANDB_URL_RE = re.compile(
    r"https://wandb\.ai/(?P<entity>[^/\s]+)/(?P<project>[^/\s]+)/runs/(?P<run_id>[^\s\x1b]+)"
)

LOG_TEMPLATE = "bd3lm_openwebtext-split_block_size16_exact_ll_{strategy}_{k}.log"

# Integer-k strategies: sweep k ∈ {1, 2, 4, 8}
K_VALUES = [1, 2, 4, 8]
INT_K_STRATEGIES = ["block_greedy", "block_left_to_right", "block_probability_margin"]
INTEGER_K_NFE = {1: 1024, 2: 512, 4: 256, 8: 128}

# Confidence threshold strategy: thresholds match duel_ppl.sh (NFE 128/256/512/1024).
THRESHOLDS = [0.05, 0.07, 0.15, 0.99]


def parse_log(path: Path):
    if not path.exists():
        return None
    matches = PPL_RE.findall(path.read_text(errors="ignore"))
    return float(matches[-1]) if matches else None


def wandb_run_path(path: Path):
    """Return 'entity/project/run_id' parsed from the wandb URL in the log."""
    if not path.exists():
        return None
    m = WANDB_URL_RE.search(path.read_text(errors="ignore"))
    if not m:
        return None
    return f"{m['entity']}/{m['project']}/{m['run_id']}"


def mean_nfe_from_wandb(run_path: str):
    """Mean of `val/num_decoding_steps` across the run's validation batches."""
    import wandb  # imported lazily so --no-wandb works without the dep
    api = wandb.Api()
    run = api.run(run_path)
    hist = run.history(keys=["val/num_decoding_steps"], pandas=True)
    if hist.empty or "val/num_decoding_steps" not in hist:
        return None
    return float(hist["val/num_decoding_steps"].mean())


def make_row(logs_dir: Path, strategy: str, k, nfe, *, fetch_nfe_from_wandb=False):
    name = LOG_TEMPLATE.format(strategy=strategy, k=k)
    log_path = logs_dir / name
    row = {
        "strategy": strategy,
        "k": k,
        "nfe": nfe,
        "duel_ppl": parse_log(log_path),
        "log": name,
    }
    if fetch_nfe_from_wandb:
        run_path = wandb_run_path(log_path)
        if run_path is not None:
            try:
                row["nfe"] = mean_nfe_from_wandb(run_path)
            except Exception as e:
                print(f"  [wandb fetch failed for {run_path}: {e}]", file=sys.stderr)
    return row


def main(logs_dir: Path, use_wandb: bool = True):
    rows = []
    for strategy in INT_K_STRATEGIES:
        for k in K_VALUES:
            rows.append(make_row(logs_dir, strategy, k, INTEGER_K_NFE[k]))
    for k in THRESHOLDS:
        rows.append(make_row(
            logs_dir, "block_confidence_threshold", k, None,
            fetch_nfe_from_wandb=use_wandb,
        ))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs_dir", nargs="?", default="logs", type=Path)
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Skip W&B fetch; leave confidence-threshold NFE blank.",
    )
    args = parser.parse_args()

    rows = main(args.logs_dir, use_wandb=not args.no_wandb)
    width = max(len(r["log"]) for r in rows) if rows else 0
    print(f"{'log':<{width}}  strategy                  k       NFE       exact_ppl")
    for r in rows:
        if r["nfe"] is None:
            nfe_str = "-"
        elif isinstance(r["nfe"], float):
            nfe_str = f"{r['nfe']:.1f}"
        else:
            nfe_str = str(r["nfe"])
        ppl = f"{r['duel_ppl']:.2f}" if r["duel_ppl"] is not None else "MISSING"
        print(f"{r['log']:<{width}}  {r['strategy']:<25} {str(r['k']):<7} {nfe_str:<8}  {ppl}")
