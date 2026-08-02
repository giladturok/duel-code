"""Parse the 16 sample-log configs, sample rows, clean text, and blind.

Outputs
-------
out/blinding.json   {"seed": ..., "mapping": {"c00": "<real config name>", ...}}
out/prep.json       Blinded per-config data: sampled rows with cleaned text,
                    truncation metadata, per-row gen_ppl, file-level MAUVE,
                    and skipped-row counts.

Parsing follows scripts/cola_quality/cola_data.py: headerless CSV with
columns [gen_ppl, gen_nfes, gen_entropy, gen_lengths, mauve_score, samples,
seed]; `samples` is repr of a one-element python list with newlines escaped,
recovered via ast.literal_eval. Rows whose samples field fails to parse as a
one-element list of str (e.g. list-repr scalars like `[nan]`) are skipped and
counted per config — never silently.
"""
import ast
import csv
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

COLUMNS = ["gen_ppl", "gen_nfes", "gen_entropy", "gen_lengths",
           "mauve_score", "samples", "seed"]

_WORD = re.compile(r"\S+")


def clean_text(text):
    """Strip leading EOT, replace internal EOT with a dashed separator,
    truncate to the first MAX_WORDS whitespace words (format-preserving)."""
    t = text.lstrip()
    while t.startswith(C.EOT):
        t = t[len(C.EOT):].lstrip()
    t = t.replace(C.EOT, "\n\n-----\n\n").strip()
    matches = list(_WORD.finditer(t))
    orig_words = len(matches)
    truncated = orig_words > C.MAX_WORDS
    if truncated:
        t = t[: matches[C.MAX_WORDS - 1].end()]
    return t, truncated, orig_words


def parse_config(path):
    """Return (rows: {index: record}, skipped: int, mauve: float|None)."""
    rows, skipped, mauve = {}, 0, None
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if len(row) != len(COLUMNS):
                skipped += 1
                continue
            rec = dict(zip(COLUMNS, row))
            try:
                samples = ast.literal_eval(rec["samples"])
            except (ValueError, SyntaxError):
                skipped += 1
                continue
            if (not isinstance(samples, list) or len(samples) != 1
                    or not isinstance(samples[0], str)):
                skipped += 1
                continue
            try:
                gen_ppl = float(rec["gen_ppl"])
                mv = float(rec["mauve_score"])
            except ValueError:
                skipped += 1
                continue
            if mauve is None:
                mauve = mv
            rows[i] = {"text": samples[0], "gen_ppl": gen_ppl}
    return rows, skipped, mauve


def main():
    rng = random.Random(C.SEED)
    indices = rng.sample(range(1000), C.N_ABS_INDICES)

    # Blinding: shuffle the sorted config-name list with the same seed.
    names = list(C.CONFIGS)
    random.Random(C.SEED).shuffle(names)
    mapping = {f"c{i:02d}": name for i, name in enumerate(names)}
    C.dump_json({"seed": C.SEED, "mapping": mapping},
                os.path.join(C.OUT_DIR, "blinding.json"))

    out = {"seed": C.SEED, "indices": indices, "configs": {}}
    for cid, name in sorted(mapping.items()):
        path = os.path.join(C.SAMPLE_LOG_DIR, C.PREFIX + name + ".txt")
        rows, skipped, mauve = parse_config(path)
        sampled, missing = {}, []
        for idx in indices:
            if idx not in rows:
                missing.append(idx)
                continue
            text, truncated, orig_words = clean_text(rows[idx]["text"])
            sampled[str(idx)] = {
                "text": text,
                "truncated": truncated,
                "orig_words": orig_words,
                "gen_ppl": rows[idx]["gen_ppl"],
            }
        texts = [r["text"] for r in sampled.values()]
        assert len(set(texts)) == len(texts), (
            f"duplicate cleaned texts within config {cid}")
        out["configs"][cid] = {
            "n_rows_parsed": len(rows),
            "n_rows_skipped": skipped,
            "missing_sampled_indices": missing,
            "mauve": mauve,
            "rows": sampled,
        }
        print(f"{cid}: parsed={len(rows)} skipped={skipped} "
              f"sampled={len(sampled)} missing={len(missing)}")

    C.dump_json(out, os.path.join(C.OUT_DIR, "prep.json"))
    print(f"wrote {os.path.join(C.OUT_DIR, 'prep.json')}")


if __name__ == "__main__":
    main()
