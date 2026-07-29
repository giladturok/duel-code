"""Mandatory gate: reproduce published CoLA validation MCC for candidate checkpoints.

We must not score BD3-LM samples with a checkpoint whose published metric we cannot
reproduce, because the whole point of the metric is that it is a *trustworthy*
discriminative alternative to generative perplexity.

For each candidate we report, on the GLUE CoLA validation split (1043 sentences):
  - accuracy
  - Matthews correlation coefficient (MCC), the official CoLA metric
  - the inferred orientation of the positive ("acceptable") class

Label orientation is inferred, not assumed: several community checkpoints
(textattack in particular) ship id2label = {LABEL_0, LABEL_1} with no semantics, and
some are trained with the CoLA labels flipped. We compute MCC for both orientations
and report both; a checkpoint is only usable if one orientation clearly reproduces
the published number.

Usage:
  python cola_gate.py --out gate_results.json
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# (model_id, published metric description, published value or None)
CANDIDATES = [
    ("yiiino/deberta-v3-large-cola", "mcc", 0.7193201130196331),
    ("JeremiahZ/roberta-base-cola", "mcc", 0.6232164195970928),
    ("mrm8488/deberta-v3-small-finetuned-cola", "mcc", 0.6333205721749096),
    ("textattack/roberta-base-CoLA", "acc", 0.850431447746884),
    ("textattack/bert-base-uncased-CoLA", "acc", None),
    ("cointegrated/roberta-large-cola-krishna2020", "acc", None),
]


def load_cola_validation():
    import datasets
    ds = datasets.load_dataset("nyu-mll/glue", "cola", split="validation")
    return list(ds["sentence"]), np.asarray(ds["label"])


@torch.no_grad()
def predict(model_id, sentences, device, batch_size=64):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval().to(device)
    id2label = dict(getattr(model.config, "id2label", {}) or {})
    probs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(device)
        logits = model(**enc).logits.float()
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(probs, axis=0), tok.name_or_path, id2label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gate_results.json")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}", flush=True)

    sentences, labels = load_cola_validation()
    print(f"CoLA validation: {len(sentences)} sentences, "
          f"{labels.sum()} acceptable ({labels.mean():.3f})", flush=True)
    assert len(sentences) == 1043, f"unexpected CoLA val size {len(sentences)}"

    results = {}
    for model_id, kind, published in CANDIDATES:
        print(f"\n=== {model_id} ===", flush=True)
        try:
            probs, _, id2label = predict(model_id, sentences, device, args.batch_size)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            results[model_id] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        pred_argmax = probs.argmax(axis=1)
        # orientation A: class index 1 == acceptable (the GLUE convention)
        acc_a = float((pred_argmax == labels).mean())
        mcc_a = float(matthews_corrcoef(labels, pred_argmax))
        # orientation B: class index 0 == acceptable (labels flipped)
        acc_b = float((1 - pred_argmax == labels).mean())
        mcc_b = float(matthews_corrcoef(labels, 1 - pred_argmax))

        best = "index1_is_acceptable" if mcc_a >= mcc_b else "index0_is_acceptable"
        entry = {
            "id2label": {str(k): v for k, v in (id2label or {}).items()},
            "published_kind": kind,
            "published_value": published,
            "acc_index1_acceptable": acc_a,
            "mcc_index1_acceptable": mcc_a,
            "acc_index0_acceptable": acc_b,
            "mcc_index0_acceptable": mcc_b,
            "chosen_orientation": best,
            "mcc": max(mcc_a, mcc_b),
            "acc": acc_a if mcc_a >= mcc_b else acc_b,
        }
        if published is not None:
            got = entry["mcc"] if kind == "mcc" else entry["acc"]
            entry["reproduced_value"] = got
            entry["abs_delta_vs_published"] = abs(got - published)
            entry["reproduces"] = bool(abs(got - published) < 0.01)
        results[model_id] = entry
        print(json.dumps(entry, indent=2), flush=True)

    with open(args.out, "w") as fh:
        json.dump({"n_val": len(sentences), "results": results}, fh, indent=2)
    print(f"\nwrote {args.out}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'model':<48} {'MCC':>7} {'acc':>7} {'orientation':>24} {'repro':>7}")
    for mid, e in results.items():
        if "error" in e:
            print(f"{mid:<48} {'ERROR':>7}")
            continue
        rep = e.get("reproduces")
        rep_s = "-" if rep is None else ("YES" if rep else "NO")
        print(f"{mid:<48} {e['mcc']:>7.4f} {e['acc']:>7.4f} "
              f"{e['chosen_orientation']:>24} {rep_s:>7}")


if __name__ == "__main__":
    main()
