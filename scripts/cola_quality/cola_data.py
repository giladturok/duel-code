"""Parse BD3-LM sample logs and segment passages into acceptability-scoring units.

Parsing
-------
`sample_logs/*.txt` are headerless CSVs written by `utils.update_and_save_csv`
(the `writeheader()` branch is unreachable because the file is opened in append
mode and already exists by then). Column order comes from `main.py:105-111`:

    gen_ppl, gen_nfes, gen_entropy, gen_lengths, mauve_score, samples, seed

`samples` was built as `[[t] for t in text_samples]`, so `csv.writer` stored
`repr(['...'])`. That means the field is a Python list literal and, importantly,
newlines inside the text are stored *escaped* (`\\n`), so every record occupies
exactly one physical line. `ast.literal_eval` recovers the text verbatim.

Segmentation
------------
CoLA classifiers are trained on single short sentences; our inputs are ~1024-token
passages. Two segmentation schemes are provided:

`sentence` (primary)
    Split on `<|endoftext|>` document boundaries, then on blank lines, then on
    sentence-terminal punctuation. Over-long units are *hard-split*, never
    truncated. Under-length units are *merged forward*, never dropped.

`window` (control)
    Fixed-width windows of GPT-2 tokens, ignoring punctuation entirely. Because
    degraded cells contain far more punctuation-like noise than clean cells
    (measured: 42 `[.!?]` per 1k chars at NFE 128 vs 9 at NFE 1024), a
    punctuation-driven segmentation could differ systematically across cells.
    The window scheme has identical unit length in every cell, so it isolates the
    text-quality signal from the segmentation.

Nothing is ever silently dropped. `segment_passage` returns a per-passage
accounting of tokens in, tokens scored and units below the minimum length, so the
retention rate can be reported per cell. Silently discarding unparseable units
would bias the metric *against* the degraded cells -- i.e. in the direction of our
own hypothesis -- so retention is a first-class output, not a diagnostic.
"""
import ast
import csv
import os
import re
import sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

EOT = "<|endoftext|>"

COLUMNS = ["gen_ppl", "gen_nfes", "gen_entropy", "gen_lengths",
           "mauve_score", "samples", "seed"]

# Cell name -> (rule, k, nominal NFE budget). Nominal budget is the measured mean
# NFE rounded to the 128/256/512/1024 grid.
CELLS = {
    "block-greedy_k1": ("greedy", 1, 1024),
    "block-greedy_k2": ("greedy", 2, 512),
    "block-greedy_k4": ("greedy", 4, 256),
    "block-greedy_k8": ("greedy", 8, 128),
    "block-left-to-right_k1": ("left-to-right", 1, 1024),
    "block-left-to-right_k2": ("left-to-right", 2, 512),
    "block-left-to-right_k4": ("left-to-right", 4, 256),
    "block-left-to-right_k8": ("left-to-right", 8, 128),
    "block-probability-margin_k1": ("probability-margin", 1, 1024),
    "block-probability-margin_k2": ("probability-margin", 2, 512),
    "block-probability-margin_k4": ("probability-margin", 4, 256),
    "block-probability-margin_k8": ("probability-margin", 8, 128),
    "block-confidence-threshold_k0.045": ("confidence-threshold", 0.045, 128),
    "block-confidence-threshold_k0.08": ("confidence-threshold", 0.08, 256),
    "block-confidence-threshold_k0.16": ("confidence-threshold", 0.16, 512),
    "block-confidence-threshold_k1.0": ("confidence-threshold", 1.0, 1024),
    "uniform": ("uniform", None, 1024),
}

PREFIX = "samples_bd3lm_len1024_blocksize16_"


def cell_name_from_path(path):
    return os.path.basename(path)[len(PREFIX):-len(".txt")]


def read_sample_log(path):
    """Yield dicts, one per row. Raises on malformed column counts."""
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if len(row) != len(COLUMNS):
                raise ValueError(
                    f"{path}: row {i} has {len(row)} fields, expected "
                    f"{len(COLUMNS)}")
            rec = dict(zip(COLUMNS, row))
            samples = ast.literal_eval(rec["samples"])
            if not isinstance(samples, list) or len(samples) != 1:
                raise ValueError(
                    f"{path}: row {i} samples field is not a 1-element list "
                    f"(got {type(samples).__name__} of len "
                    f"{len(samples) if hasattr(samples, '__len__') else '?'})")
            yield {
                "index": i,
                "gen_ppl": float(rec["gen_ppl"]),
                "gen_nfes": float(rec["gen_nfes"]),
                "gen_entropy": float(rec["gen_entropy"]),
                "gen_lengths": float(rec["gen_lengths"]),
                "mauve_score": float(rec["mauve_score"]),
                "seed": int(rec["seed"]),
                "text": samples[0],
            }


# A sentence boundary: terminal punctuation (optionally followed by closing
# quotes/brackets) then whitespace. Deliberately a plain regex rather than a
# trained splitter: on symbol salad a trained model's behaviour is unpredictable
# and unauditable, and it would differ systematically between clean and degraded
# cells, which is precisely the confound we are trying to avoid.
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]”’]*\s+')
_PARA_BOUNDARY = re.compile(r"\n\s*\n+")
_LINE_BOUNDARY = re.compile(r"\n+")


def _raw_units_sentence(text):
    """Split into candidate units on doc -> paragraph -> line -> sentence."""
    units = []
    for doc in text.split(EOT):
        if not doc.strip():
            continue
        for para in _PARA_BOUNDARY.split(doc):
            if not para.strip():
                continue
            for line in _LINE_BOUNDARY.split(para):
                if not line.strip():
                    continue
                for sent in _SENT_BOUNDARY.split(line):
                    if sent.strip():
                        units.append(sent.strip())
    return units


def segment_passage(text, tokenizer, scheme="sentence", min_tok=4, max_tok=96,
                    window=32):
    """Segment one passage.

    Returns (units, stats) where `units` is a list of text strings and `stats`
    accounts for every token so the retention rate can be audited.

    Token counts use the GPT-2 tokenizer the samples were generated with, so they
    are comparable to the `gen_lengths` column.
    """
    # tokens present in the passage once document markers are removed
    body = text.replace(EOT, " ")
    n_tok_in = len(tokenizer(body, add_special_tokens=False)["input_ids"])

    if scheme == "window":
        raw = []
        for doc in text.split(EOT):
            if not doc.strip():
                continue
            ids = tokenizer(doc, add_special_tokens=False)["input_ids"]
            for s in range(0, len(ids), window):
                chunk = tokenizer.decode(ids[s : s + window]).strip()
                if chunk:
                    raw.append(chunk)
        units = raw
        n_short = sum(
            1 for u in units
            if len(tokenizer(u, add_special_tokens=False)["input_ids"]) < min_tok)
    elif scheme == "sentence":
        raw = _raw_units_sentence(text)
        # hard-split over-long units (never truncate: keep all the text)
        split = []
        for u in raw:
            ids = tokenizer(u, add_special_tokens=False)["input_ids"]
            if len(ids) <= max_tok:
                split.append((u, len(ids)))
                continue
            for s in range(0, len(ids), max_tok):
                piece = tokenizer.decode(ids[s : s + max_tok]).strip()
                if piece:
                    split.append(
                        (piece,
                         len(tokenizer(piece, add_special_tokens=False)["input_ids"])))
        # merge under-length units forward into the previous unit, so degraded
        # cells do not turn into a mass of 1-token fragments and so that nothing
        # has to be dropped for being too short
        merged = []
        for piece, ntok in split:
            if merged and ntok < min_tok and merged[-1][1] + ntok <= max_tok:
                prev, pn = merged.pop()
                merged.append((prev + " " + piece, pn + ntok))
            else:
                merged.append((piece, ntok))
        units = [u for u, _ in merged]
        n_short = sum(1 for _, n in merged if n < min_tok)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    n_tok_scored = sum(
        len(tokenizer(u, add_special_tokens=False)["input_ids"]) for u in units)
    stats = {
        "n_tok_in": n_tok_in,
        "n_tok_scored": n_tok_scored,
        "n_units": len(units),
        "n_units_short": n_short,
        "empty_passage": len(units) == 0,
    }
    return units, stats
