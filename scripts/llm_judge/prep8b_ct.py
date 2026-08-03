"""Prep the 4 confidence-threshold (CT) arms for the nucleus add-on study.

Reads the CT shard0-128 files, applies the SAME cleaning as prep8b
(truncate at first <|endoftext|>, MAX_WORDS cap, per-row metadata), and:

1. Extends out8b_nuc/blinding.json with 4 NEW opaque ids n16..n19
   (existing n00..n15 entries are asserted untouched; idempotent).
2. Writes out8b_nuc/ct_prep.json in the prep.json format (same shuffled
   `indices` list as prep.json, so pairwise stays prefix-matched), plus
   per-row nfe_measured and per-config achieved-NFE stats.

Requires out8b_nuc/prep.json and blinding.json (the completed nucleus
study) to exist.
"""
import json
import os
import random
import sys

os.environ.setdefault("JUDGE8B_VARIANT", "nuc")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import common8b_ct as CT
from prep8b import clean_text  # identical cleaning (JUDGE8B_VARIANT=nuc)


def extend_blinding():
    path = os.path.join(CT.OUT_DIR, "blinding.json")
    blinding = C.load_json(path)
    before = dict(blinding["mapping"])
    names = list(CT.CT_CONFIGS)
    random.Random(CT.CT_BLIND_SEED).shuffle(names)
    new = {f"n{CT.CT_ID_START + i:02d}": name for i, name in enumerate(names)}
    for cid, name in new.items():
        if cid in blinding["mapping"]:
            assert blinding["mapping"][cid] == name, (cid, name)
        else:
            blinding["mapping"][cid] = name
    # never touch pre-existing entries
    assert all(blinding["mapping"][k] == v for k, v in before.items())
    blinding["ct_blind_seed"] = CT.CT_BLIND_SEED
    C.dump_json(blinding, path)
    print(f"blinding: {len(blinding['mapping'])} ids "
          f"(ct: {sorted(new)}) -> {path}")
    return {cid: name for cid, name in blinding["mapping"].items()
            if name in CT.CT_CONFIGS}


def main():
    prep = C.load_json(os.path.join(CT.OUT_DIR, "prep.json"))
    indices = prep["indices"]
    assert sorted(indices) == list(range(128))

    ct_mapping = extend_blinding()

    out = {"seed": CT.SEED, "indices": indices, "note": CT.CT_NOTE,
           "configs": {}}
    for cid, name in sorted(ct_mapping.items()):
        tau = name[len("ct_tau"):]
        rows, n_lines = {}, 0
        # Dedup by sample_index keep-LAST (refill jobs can append a full
        # regeneration pass on top of a partial one).
        with open(CT.ct_data_path(tau)) as fh:
            for line in fh:
                rec = json.loads(line)
                n_lines += 1
                assert rec["strategy"] == "confidence_threshold", rec["strategy"]
                assert rec["tau"] == float(tau), rec["tau"]
                assert rec["nucleus_p"] == 0.98, rec["nucleus_p"]
                key = rec.get("sample_index", rec["prefix_index"])
                if key < 128:
                    rows[key] = rec
        assert sorted(rows) == list(range(128)), (
            f"{name}: expected exactly indices 0-127 after dedup, got "
            f"{len(rows)} unique of {n_lines} lines")

        sampled, missing, skipped = {}, [], 0
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
                "full_length_continuation": True,  # asserted by nuc prep
                "word_capped": word_capped,
                "orig_words": orig_words,
                "nfe_measured": rec["nfe_measured"],
            }
        texts = [r["text"] for r in sampled.values()]
        assert len(set(texts)) == len(texts), (
            f"duplicate cleaned texts within config {cid}")
        nfes = sorted(r["nfe_measured"] for r in sampled.values())
        out["configs"][cid] = {
            "n_rows_parsed": len(rows),
            "n_rows_skipped": skipped,
            "missing_sampled_indices": missing,
            "nfe_mean": sum(nfes) / len(nfes),
            "nfe_median": (nfes[len(nfes) // 2 - 1]
                           + nfes[len(nfes) // 2]) / 2,
            "rows": sampled,
        }
        n_break = sum(r["truncated_at_doc_break"] for r in sampled.values())
        print(f"{cid} ({name} -> cell {CT.CT_BUDGET_OF[name]}): "
              f"sampled={len(sampled)} skipped_empty={skipped} "
              f"doc_break={n_break} nfe mean={out['configs'][cid]['nfe_mean']:.1f} "
              f"median={out['configs'][cid]['nfe_median']:.1f}")

    C.dump_json(out, os.path.join(CT.OUT_DIR, "ct_prep.json"))
    print(f"wrote {os.path.join(CT.OUT_DIR, 'ct_prep.json')}")


if __name__ == "__main__":
    main()
