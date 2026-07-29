"""Build anchor passages so the absolute CoLA numbers are interpretable.

CoLA is curated sentences from the linguistics literature; BD3-LM samples are
machine-generated web text. A raw P(acceptable) of, say, 0.55 means nothing on its own,
so every cell is reported against fixed reference points built from the *same* held-out
data the model was evaluated on (`openwebtext-valid-1k` = `openwebtext` `train[-1000:]`,
per `dataloader.py:383-390`).

Anchors, all 1000 passages, all truncated to the same 1024 GPT-2 tokens as the samples:

`anchor-owt-real`
    Real held-out OpenWebText. **Upper anchor**: the ceiling any sampler could reach,
    and the honest answer to "how acceptable does genuine web text look to a CoLA model?"

`anchor-owt-token-shuffled`
    The same passages with GPT-2 tokens permuted uniformly at random within the passage.
    **Floor**: unigram statistics identical to real text, all syntax destroyed. This is the
    lower anchor the `uniform` sampler cell cannot provide -- `uniform` turns out to be a
    full-budget NFE=1023 sampler producing fluent text (mean gen_ppl 33.8), not degenerate
    output.

`anchor-owt-word-shuffled`
    Words permuted within each sentence, sentence boundaries and punctuation preserved.
    A milder floor: local word identity and sentence length are kept, only word order is
    destroyed. Separates "wrong words" from "wrong order".

`anchor-owt-repeat`
    **The failure-mode probe.** One well-formed real sentence, repeated until the passage
    is ~1024 tokens. Every unit is a perfectly grammatical English sentence, but the
    passage as a whole is exactly the repetitive degeneracy that generative perplexity
    *rewards*. If this anchor scores high, then a per-sentence acceptability score is
    necessary but not sufficient, and cannot by itself replace gen-ppl -- it would share
    gen-ppl's blind spot rather than fix it. This anchor is the direct test of that.

Output: JSONL with {"cell": ..., "index": ..., "text": ...}.
"""
import argparse
import json
import os
import random
import re

MAXTOK = 1024

# The prepared `openwebtext` arrow shards already in the shared cache. We read the
# final shard directly rather than calling load_dataset(): the cached builder is a
# script-based dataset, and load_dataset() tries to re-extract from the original
# source tarballs (which are not present), so it starts a 12 GB re-download instead
# of using the cache. Reading the last shard and taking its final `n` rows is
# *identical* to `train[-n:]`, i.e. `openwebtext-valid-1k` from dataloader.py:383-390,
# because the split is a plain concatenation of the shards in order.
OWT_DIR = ("/share/kuleshov/gt345/.cache/huggingface/datasets/openwebtext/"
           "plain_text/0.0.0/b4325f019c648b1641a1784748667e8b74e5e064")


def load_owt_tail(n):
    import datasets
    info = json.load(open(os.path.join(OWT_DIR, "dataset_info.json")))
    split = info["splits"]["train"]
    lengths = split["shard_lengths"]
    total = split["num_examples"]
    assert sum(lengths) == total, (sum(lengths), total)
    last = len(lengths) - 1
    assert lengths[last] >= n, (
        f"last shard has only {lengths[last]} rows, need {n}")
    shard = os.path.join(OWT_DIR,
                         f"openwebtext-train-{last:05d}-of-{len(lengths):05d}.arrow")
    ds = datasets.Dataset.from_file(shard)
    assert len(ds) == lengths[last], (len(ds), lengths[last])
    print(f"OWT train has {total} examples across {len(lengths)} shards; "
          f"reading last shard {os.path.basename(shard)} ({len(ds)} rows) and "
          f"taking its final {n} == global train[-{n}:]", flush=True)
    return list(ds[len(ds) - n:]["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2)  # same seed as the sample logs
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2", model_max_length=int(1e9))
    docs = load_owt_tail(args.n)
    print(f"loaded {len(docs)} held-out OWT documents", flush=True)

    rng = random.Random(args.seed)
    sent_re = re.compile(r'(?<=[.!?])\s+')

    out = []
    for i, doc in enumerate(docs):
        ids = tok(doc, add_special_tokens=False)["input_ids"][:MAXTOK]
        real = tok.decode(ids)
        out.append({"cell": "anchor-owt-real", "index": i, "text": real})

        # token-shuffled floor
        perm = list(ids)
        rng.shuffle(perm)
        out.append({"cell": "anchor-owt-token-shuffled", "index": i,
                    "text": tok.decode(perm)})

        # word-shuffled within sentence (order destroyed, sentences preserved)
        parts = []
        for sent in sent_re.split(real):
            w = sent.split()
            if len(w) > 1:
                rng.shuffle(w)
            parts.append(" ".join(w))
        out.append({"cell": "anchor-owt-word-shuffled", "index": i,
                    "text": " ".join(parts)})

        # repetition-degeneracy probe: pick the first sentence of >= 8 words that
        # ends in terminal punctuation, then repeat it to ~MAXTOK tokens
        cands = [s.strip() for s in sent_re.split(real)
                 if len(s.split()) >= 8 and s.strip()[-1:] in ".!?"]
        if not cands:
            # no clean sentence in this document; fall back to the whole passage so
            # this anchor keeps exactly n entries rather than silently shrinking
            rep_text = real
            rep_src = None
        else:
            rep_src = cands[0]
            k = len(tok(rep_src, add_special_tokens=False)["input_ids"])
            reps = max(2, MAXTOK // max(k, 1))
            rep_text = " ".join([rep_src] * reps)
            rep_ids = tok(rep_text, add_special_tokens=False)["input_ids"][:MAXTOK]
            rep_text = tok.decode(rep_ids)
        out.append({"cell": "anchor-owt-repeat", "index": i, "text": rep_text,
                    "repeated_sentence": rep_src})

    with open(args.out, "w") as fh:
        for rec in out:
            fh.write(json.dumps(rec) + "\n")

    from collections import Counter
    print(Counter(r["cell"] for r in out), flush=True)
    n_fallback = sum(1 for r in out if r["cell"] == "anchor-owt-repeat"
                     and r.get("repeated_sentence") is None)
    print(f"anchor-owt-repeat fallbacks (no clean sentence found): {n_fallback}",
          flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
