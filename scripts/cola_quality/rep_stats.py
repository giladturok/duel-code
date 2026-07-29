"""Repetition statistics per cell -- the complement to the CoLA acceptability score.

Why this exists. The `anchor-owt-repeat` probe (one real sentence repeated to fill 1024
tokens) scores *higher* on CoLA acceptability than genuine held-out OpenWebText, because
every unit in it is a perfectly grammatical sentence. So a per-sentence acceptability
score is blind to repetitive degeneracy in the same direction generative perplexity is
-- it does not *reward* it the way gen-ppl does, but it does not penalise it either.
CoLA acceptability is therefore necessary but not sufficient, and needs a
passage-level degeneracy statistic alongside it.

Reported per cell, over GPT-2 tokens (the tokenizer the samples were generated with):

  `rep_n`      1 - (# distinct n-grams / # n-grams), averaged over passages. 0 = no
               n-gram ever repeats; -> 1 = the passage is one phrase over and over.
               This is the `rep-n` of Welleck et al. 2020 / Holtzman et al. 2020.
  `distinct_1` type-token ratio over unigrams.
  `max_ngram_frac` share of the passage taken by its single most frequent 4-gram.
               Sensitive to one dominant repeated phrase in a way rep_n averages away.

Runs on CPU; no GPU needed.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cola_data as cd  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))


def rep_n(ids, n):
    if len(ids) < n + 1:
        return np.nan
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def max_ngram_frac(ids, n=4):
    if len(ids) < n + 1:
        return np.nan
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return Counter(grams).most_common(1)[0][1] / len(grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--npz-out", default=None,
                    help="also dump per-passage rep_4 arrays, for joining against "
                         "the per-passage CoLA scores")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2", model_max_length=int(1e9))

    jobs = []
    for cell in cd.CELLS:
        path = os.path.join(REPO, "sample_logs", cd.PREFIX + cell + ".txt")
        if os.path.exists(path):
            jobs.append((cell, [r["text"] for r in cd.read_sample_log(path)]))
    if args.anchors:
        by_cell = {}
        with open(args.anchors) as fh:
            for line in fh:
                rec = json.loads(line)
                by_cell.setdefault(rec["cell"], []).append(rec["text"])
        jobs.extend(sorted(by_cell.items()))

    out = {}
    per_passage = {}
    print(f"{'cell':<32} {'rep_1':>7} {'rep_4':>7} {'rep_8':>7} "
          f"{'distinct_1':>10} {'max4gram':>9}")
    for cell, texts in jobs:
        if args.limit:
            texts = texts[: args.limit]
        acc = {"rep_1": [], "rep_4": [], "rep_8": [], "distinct_1": [],
               "max_ngram_frac": []}
        for t in texts:
            ids = tok(t.replace(cd.EOT, " "), add_special_tokens=False)["input_ids"]
            acc["rep_1"].append(rep_n(ids, 1))
            acc["rep_4"].append(rep_n(ids, 4))
            acc["rep_8"].append(rep_n(ids, 8))
            acc["distinct_1"].append(
                len(set(ids)) / len(ids) if ids else np.nan)
            acc["max_ngram_frac"].append(max_ngram_frac(ids, 4))
        out[cell] = {k: float(np.nanmean(v)) for k, v in acc.items()}
        out[cell]["n"] = len(texts)
        per_passage[cell] = {k: np.asarray(v, dtype=np.float32)
                             for k, v in acc.items()}
        o = out[cell]
        print(f"{cell:<32} {o['rep_1']:>7.4f} {o['rep_4']:>7.4f} {o['rep_8']:>7.4f} "
              f"{o['distinct_1']:>10.4f} {o['max_ngram_frac']:>9.5f}", flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.out}")
    if args.npz_out:
        flat = {f"{cell}::{k}": v
                for cell, d in per_passage.items() for k, v in d.items()}
        np.savez_compressed(args.npz_out, **flat)
        print(f"wrote {args.npz_out}")


if __name__ == "__main__":
    main()
