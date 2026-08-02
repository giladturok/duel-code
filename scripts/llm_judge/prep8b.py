"""Prepare the 8B prefix-grid samples for judging.

Outputs
-------
out8b/blinding.json  {"seed": ..., "mapping": {"b00": "<strategy>_k<k>", ...}}
out8b/prep.json      Blinded per-config data: 100 sampled prefix indices
                     (fixed seed, from 0..127), cleaned text, and per-row
                     metadata for analysis slicing:
                       truncated_at_doc_break   text contained <|endoftext|>
                                                and was cut at its first
                                                occurrence
                       full_length_continuation the TRUE continuation for
                                                this prefix_index has 256
                                                tokens (true_continuations
                                                .json; 19/128 are short)

Judge text = the `text` field (continuation only), truncated at the first
<|endoftext|>, then capped at MAX_WORDS whitespace words (same cap as the
110M study; rarely binds at ~256 generated tokens).
"""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import common8b as C8

EOT = "<|endoftext|>"
_WORD = re.compile(r"\S+")


def clean_text(text):
    truncated_at_doc_break = EOT in text
    t = text.split(EOT, 1)[0].strip()
    matches = list(_WORD.finditer(t))
    orig_words = len(matches)
    word_capped = orig_words > C8.MAX_WORDS
    if word_capped:
        t = t[: matches[C8.MAX_WORDS - 1].end()]
    return t, truncated_at_doc_break, word_capped, orig_words


def main():
    with open(os.path.join(C8.DATA_DIR, "true_continuations.json")) as fh:
        true_cont = json.load(fh)
    true_ntok = {e["prefix_index"]: e["n_tokens"] for e in true_cont
                 if e["prefix_index"] < 128}
    assert len(true_ntok) == 128

    rng = random.Random(C8.SEED)
    indices = rng.sample(range(128), C8.N_ABS_INDICES)

    names = list(C8.CONFIGS)
    random.Random(C8.SEED).shuffle(names)
    mapping = {f"b{i:02d}": name for i, name in enumerate(names)}
    C.dump_json({"seed": C8.SEED, "mapping": mapping},
                os.path.join(C8.OUT_DIR, "blinding.json"))

    out = {"seed": C8.SEED, "indices": indices, "configs": {}}
    for cid, name in sorted(mapping.items()):
        strategy, k = name.rsplit("_k", 1)
        rows = {}
        skipped = 0
        with open(C8.data_path(strategy, int(k))) as fh:
            for line in fh:
                rec = json.loads(line)
                assert rec["strategy"].replace("-", "_") == strategy, rec["strategy"]
                assert int(rec["k"]) == int(k)
                rows[rec["prefix_index"]] = rec
        assert sorted(rows) == list(range(128)), name

        sampled, missing = {}, []
        for idx in indices:
            rec = rows[idx]
            text, at_break, word_capped, orig_words = clean_text(rec["text"])
            if not text:
                missing.append(idx)
                skipped += 1
                continue
            sampled[str(idx)] = {
                "text": text,
                "truncated_at_doc_break": at_break,
                "full_length_continuation": true_ntok[idx] == 256,
                "word_capped": word_capped,
                "orig_words": orig_words,
            }
        texts = [r["text"] for r in sampled.values()]
        assert len(set(texts)) == len(texts), (
            f"duplicate cleaned texts within config {cid}")
        out["configs"][cid] = {
            "n_rows_parsed": len(rows),
            "n_rows_skipped": skipped,
            "missing_sampled_indices": missing,
            "rows": sampled,
        }
        n_break = sum(r["truncated_at_doc_break"] for r in sampled.values())
        print(f"{cid}: sampled={len(sampled)} skipped_empty={skipped} "
              f"doc_break={n_break}")

    C.dump_json(out, os.path.join(C8.OUT_DIR, "prep.json"))
    print(f"wrote {os.path.join(C8.OUT_DIR, 'prep.json')}")


if __name__ == "__main__":
    main()
