"""Score one cell (or one anchor) for linguistic acceptability with a CoLA classifier.

Emits a per-passage NPZ so that aggregation, error bars and the choice of summary
statistic are all decided downstream from the same raw scores.

Per unit we store P(acceptable) and the unit's GPT-2 token count. Per passage we store
the segmentation accounting (tokens in / tokens scored / non-whitespace chars in /
scored / units below the minimum length) so the drop rate is auditable per cell.

Nothing is dropped: over-long units are hard-split, under-length units are merged
forward. Silently discarding unparseable units would penalise exactly the degraded cells
we hypothesise are worse -- a confound in favour of our own conclusion -- so retention is
reported rather than assumed.

Usage:
  python cola_score.py --cell block-greedy_k8 --model yiiino/deberta-v3-large-cola \
      --scheme sentence --out out/
  python cola_score.py --anchors out/anchors.jsonl --cell anchor-owt-real ...
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cola_data as cd  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))

# Positive-class index per checkpoint, established by the validation gate in
# cola_gate.py -- NOT assumed. cointegrated/roberta-large-cola-krishna2020 is
# label-flipped relative to the GLUE convention, which is why this is a table.
POS_INDEX = {
    "yiiino/deberta-v3-large-cola": 1,
    "textattack/roberta-base-CoLA": 1,
    "JeremiahZ/roberta-base-cola": 1,
    "mrm8488/deberta-v3-small-finetuned-cola": 1,
    "textattack/bert-base-uncased-CoLA": 1,
    "cointegrated/roberta-large-cola-krishna2020": 0,
}


def load_passages(cell, anchors_path):
    """Return list of (index, text) for a sampler cell or an anchor."""
    if cell.startswith("anchor-"):
        if not anchors_path:
            raise ValueError("--anchors required for anchor cells")
        out = []
        with open(anchors_path) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["cell"] == cell:
                    out.append((rec["index"], rec["text"]))
        if not out:
            raise ValueError(f"no rows for {cell} in {anchors_path}")
        return out
    path = os.path.join(REPO, "sample_logs", cd.PREFIX + cell + ".txt")
    return [(r["index"], r["text"]) for r in cd.read_sample_log(path)]


@torch.no_grad()
def score_units(units, model, tok, device, batch_size, max_length):
    """P(acceptable) per unit. Batches length-sorted for throughput, then unsorts."""
    order = np.argsort([len(u) for u in units])
    probs = np.empty(len(units), dtype=np.float32)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        batch = [units[j] for j in idx]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(device)
        logits = model(**enc).logits.float()
        p = torch.softmax(logits, dim=-1)[:, model._pos_index].cpu().numpy()
        probs[idx] = p
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--model", default="yiiino/deberta-v3-large-cola")
    ap.add_argument("--scheme", default="sentence", choices=["sentence", "window"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--min-tok", type=int, default=4)
    ap.add_argument("--max-tok", type=int, default=96)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N passages")
    args = ap.parse_args()

    if args.model not in POS_INDEX:
        raise SystemExit(
            f"{args.model} has not been through cola_gate.py; refusing to score with "
            f"a checkpoint whose label orientation and MCC are unverified")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.cell}__{args.model.replace('/', '_')}__{args.scheme}"
    out_path = os.path.join(args.out, tag + ".npz")

    gpt2 = AutoTokenizer.from_pretrained("gpt2", model_max_length=int(1e9))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model._pos_index = POS_INDEX[args.model]
    model.eval().to(device)

    passages = load_passages(args.cell, args.anchors)
    if args.limit:
        passages = passages[: args.limit]
    print(f"{args.cell}: {len(passages)} passages, scheme={args.scheme}, "
          f"model={args.model}, device={device}", flush=True)

    all_units, offsets, stats_rows = [], [0], []
    t0 = time.time()
    for _, text in passages:
        units, st = cd.segment_passage(
            text, gpt2, scheme=args.scheme, min_tok=args.min_tok,
            max_tok=args.max_tok, window=args.window)
        all_units.extend(units)
        offsets.append(len(all_units))
        stats_rows.append(st)
    print(f"segmented {len(all_units)} units in {time.time()-t0:.1f}s", flush=True)

    unit_tok = np.concatenate(
        [np.asarray(s["unit_tok"], dtype=np.int32) if s["unit_tok"]
         else np.zeros(0, np.int32) for s in stats_rows]) \
        if stats_rows else np.zeros(0, np.int32)

    t0 = time.time()
    probs = score_units(all_units, model, tok, device, args.batch_size,
                        args.max_length)
    dt = time.time() - t0
    print(f"scored {len(all_units)} units in {dt:.1f}s "
          f"({len(all_units)/max(dt,1e-9):.0f} units/s)", flush=True)

    np.savez_compressed(
        out_path,
        cell=args.cell, model=args.model, scheme=args.scheme,
        passage_index=np.asarray([i for i, _ in passages], dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        unit_prob=probs,
        unit_tok=unit_tok,
        n_tok_in=np.asarray([s["n_tok_in"] for s in stats_rows], np.int32),
        n_tok_scored=np.asarray([s["n_tok_scored"] for s in stats_rows], np.int32),
        n_nws_in=np.asarray([s["n_nws_in"] for s in stats_rows], np.int32),
        n_nws_scored=np.asarray([s["n_nws_scored"] for s in stats_rows], np.int32),
        n_units_short=np.asarray([s["n_units_short"] for s in stats_rows], np.int32),
        empty_passage=np.asarray([s["empty_passage"] for s in stats_rows], bool),
    )
    print(f"wrote {out_path}", flush=True)
    print(f"mean unit P(acceptable) = {probs.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
