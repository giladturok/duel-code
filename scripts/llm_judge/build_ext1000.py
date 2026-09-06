"""Build the n=1000 judge-expansion request files for BOTH scales.

Extends the existing studies (out/ = 110M BD3-LM, out8b_nuc/ = LLaDA-8B
nucleus) from 100/128 judged samples per config to 1000, holding every
element of the protocol fixed: same prompts, schemas, model/effort, blinding
mappings, cleaning functions, and custom_id formats. Only NEW sample/prefix
indices are requested — no existing custom_id is ever re-issued.

Design (budget-trimmed to fit $161 of API credit):
- Absolute arm: expanded only at the top TWO budgets per scale (110M NFE
  1024/512; 8B k=1/k=2). The lower budgets are absolute-floor (every sample
  scores 1.00 in the existing data) — 900 more requests each would carry
  zero information.
- Pairwise arm: expanded at ALL budgets, from 50 to 500 index-matched units
  per config pair (450 new x 2 presentation orders). New pair indices =
  original indices[50:] + a seeded shuffle of the never-sampled indices.

Outputs (per out dir): ext_prep.json, ext_abs_requests.jsonl,
ext_pair_requests.jsonl. Submit with run_batches.py via
LLM_JUDGE_OUT=<dir> LLM_JUDGE_REQ_PREFIX=ext_.

Validation: for every index shared with the original prep.json, the freshly
cleaned text must be byte-identical to the stored one (protocol identity);
custom_ids checked disjoint from the existing request files; total cost
estimated and asserted under budget.
"""
import json
import os
import random
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
os.environ["JUDGE8B_VARIANT"] = "nuc"
import common8b_nuc as C8
from prep import parse_config, clean_text as clean_110m
from prep8b import clean_text as clean_8b

EXT_SEED_OFFSET = 7          # rest-of-pool shuffle seed = SEED + 7
N_PAIR_TARGET = 500          # total units per config pair after expansion
ABS_EXT_BUDGETS_110M = {1024, 512}
ABS_EXT_KS_8B = {1, 2}
BUDGET_USD = 140.0           # hard ceiling for the cost assertion
IN_PRICE, OUT_PRICE = 1.0, 5.0   # $/MTok, Sonnet-5 batch intro
EST_OUT_TOK = 115


def write_jsonl(rows, path):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows):6d} requests -> {path}")


def existing_ids(out_dir, prefixes=("",)):
    ids = set()
    for p in prefixes:
        for arm in ("abs", "pair"):
            path = os.path.join(out_dir, f"{p}{arm}_requests.jsonl")
            if os.path.exists(path):
                for line in open(path):
                    ids.add(json.loads(line)["custom_id"])
    return ids


def check_overlap_texts(scale, prep_rows, new_rows, cid):
    """Every index present in both must have byte-identical text."""
    n = 0
    for idx, row in prep_rows.items():
        if idx in new_rows:
            assert new_rows[idx]["text"] == row["text"], (
                f"{scale} {cid} idx {idx}: cleaned text mismatch vs prep.json")
            n += 1
    return n


def build_pairs(mapping, budget_of, texts, pair_ext_indices, system, out):
    """Shared pairwise builder. budget_of: cid -> budget label."""
    budget_groups = {}
    for cid in mapping:
        budget_groups.setdefault(budget_of(cid), []).append(cid)
    reqs = []
    for budget in sorted(budget_groups):
        cids = sorted(budget_groups[budget])
        assert len(cids) == 4, (budget, cids)
        for cid_a, cid_b in combinations(cids, 2):
            for idx in pair_ext_indices:
                row_a = texts[cid_a].get(str(idx))
                row_b = texts[cid_b].get(str(idx))
                if row_a is None or row_b is None:
                    continue
                for order in ("AB", "BA"):
                    first, second = ((row_a, row_b) if order == "AB"
                                     else (row_b, row_a))
                    reqs.append({
                        "custom_id":
                            f"pair_{budget}_{cid_a}_{cid_b}_{idx}_{order}",
                        "params": C.message_params(
                            system,
                            C.PAIR_USER_TEMPLATE.format(
                                a=first["text"], b=second["text"]),
                            C.PAIR_SCHEMA),
                    })
    write_jsonl(reqs, out)
    return reqs


def build_110m():
    prep = C.load_json(os.path.join(C.OUT_DIR, "prep.json"))
    blinding = C.load_json(os.path.join(C.OUT_DIR, "blinding.json"))
    mapping = blinding["mapping"]
    indices = prep["indices"]
    assert len(indices) == C.N_ABS_INDICES == 100

    rest = [i for i in range(1000) if i not in set(indices)]
    random.Random(C.SEED + EXT_SEED_OFFSET).shuffle(rest)
    n_new_pair = N_PAIR_TARGET - C.N_PAIR_INDICES          # 450
    pair_ext = indices[C.N_PAIR_INDICES:] + rest[:n_new_pair - 50]
    assert len(pair_ext) == n_new_pair
    assert not set(pair_ext) & set(indices[:C.N_PAIR_INDICES])

    needed = sorted(set(rest) | set(pair_ext) | set(indices))
    texts, ext_prep = {}, {"seed": C.SEED, "ext_seed_offset": EXT_SEED_OFFSET,
                           "pair_ext_indices": pair_ext,
                           "abs_ext_indices": rest, "configs": {}}
    for cid, name in sorted(mapping.items()):
        path = os.path.join(C.SAMPLE_LOG_DIR, C.PREFIX + name + ".txt")
        rows, skipped, _ = parse_config(path)
        cleaned, missing = {}, []
        for idx in needed:
            if idx not in rows:
                missing.append(idx)
                continue
            text, truncated, orig_words = clean_110m(rows[idx]["text"])
            cleaned[str(idx)] = {"text": text, "truncated": truncated,
                                 "orig_words": orig_words}
        n_ok = check_overlap_texts(
            "110M", prep["configs"][cid]["rows"], cleaned, cid)
        texts[cid] = cleaned
        ext_prep["configs"][cid] = {
            "n_rows_parsed": len(rows), "n_rows_skipped": skipped,
            "missing_indices": missing,
            "rows": {k: v for k, v in cleaned.items()
                     if int(k) not in set(indices)}}
        print(f"110M {cid}: cleaned={len(cleaned)} missing={len(missing)} "
              f"overlap_verified={n_ok}")
    C.dump_json(ext_prep, os.path.join(C.OUT_DIR, "ext_prep.json"))

    abs_reqs = []
    for cid, name in sorted(mapping.items()):
        if C.BUDGET_OF[name] not in ABS_EXT_BUDGETS_110M:
            continue
        for idx in rest:
            row = texts[cid].get(str(idx))
            if row is None:
                continue
            abs_reqs.append({
                "custom_id": f"abs_{cid}_{idx}",
                "params": C.message_params(
                    C.ABS_SYSTEM,
                    C.ABS_USER_TEMPLATE.format(text=row["text"]),
                    C.ABS_SCHEMA)})
    write_jsonl(abs_reqs, os.path.join(C.OUT_DIR, "ext_abs_requests.jsonl"))

    pair_reqs = build_pairs(
        mapping, lambda cid: C.BUDGET_OF[mapping[cid]], texts, pair_ext,
        C.PAIR_SYSTEM, os.path.join(C.OUT_DIR, "ext_pair_requests.jsonl"))
    return abs_reqs, pair_reqs


def load_8b_config(name):
    """All 1000 rows of a config from the merged _n1000.jsonl (dedup keep
    last, mirroring prep8b)."""
    strategy, k = name.rsplit("_k", 1)
    path = os.path.join(
        C8.DATA_DIR,
        f"{strategy}_bs32_k{k}_seed{C8.FILE_SEED[strategy][int(k)]}"
        "_n1000.jsonl")
    rows = {}
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            assert rec["strategy"].replace("-", "_") == strategy
            assert int(rec["k"]) == int(k)
            assert rec["nucleus_p"] == C8.EXPECT_NUCLEUS_P
            key = rec.get("sample_index", rec["prefix_index"])
            if key < 1000:
                rows[key] = rec
    assert sorted(rows) == list(range(1000)), f"{name}: incomplete n1000"
    return rows


def build_8b():
    prep = C.load_json(os.path.join(C8.OUT_DIR, "prep.json"))
    blinding = C.load_json(os.path.join(C8.OUT_DIR, "blinding.json"))
    mapping = {cid: n for cid, n in blinding["mapping"].items()
               if not n.startswith("ct_")}
    assert len(mapping) == 16
    indices = prep["indices"]
    assert sorted(indices) == list(range(128))

    with open(C8.TRUE_CONT_PATH) as fh:
        true_ntok = {e["prefix_index"]: e["n_tokens"] for e in json.load(fh)}

    rest = list(range(128, 1000))
    random.Random(C8.SEED + EXT_SEED_OFFSET).shuffle(rest)
    n_new_pair = N_PAIR_TARGET - C8.N_PAIR_INDICES         # 450
    pair_ext = indices[C8.N_PAIR_INDICES:] + rest[:n_new_pair - 78]
    assert len(pair_ext) == n_new_pair
    assert not set(pair_ext) & set(indices[:C8.N_PAIR_INDICES])

    needed = sorted(set(rest) | set(pair_ext) | set(indices))
    texts, ext_prep = {}, {"seed": C8.SEED, "ext_seed_offset": EXT_SEED_OFFSET,
                           "pair_ext_indices": pair_ext,
                           "abs_ext_indices": rest, "configs": {}}
    for cid, name in sorted(mapping.items()):
        rows = load_8b_config(name)
        cleaned, empty = {}, []
        for idx in needed:
            text, at_break, word_capped, orig_words = clean_8b(
                rows[idx]["text"])
            if not text:
                empty.append(idx)
                continue
            cleaned[str(idx)] = {
                "text": text, "truncated_at_doc_break": at_break,
                "full_length_continuation": true_ntok[idx] == 256,
                "word_capped": word_capped, "orig_words": orig_words}
        n_ok = check_overlap_texts(
            "8B", prep["configs"][cid]["rows"], cleaned, cid)
        texts[cid] = cleaned
        ext_prep["configs"][cid] = {
            "n_empty_skipped": len(empty), "empty_indices": empty,
            "rows": {k: v for k, v in cleaned.items()
                     if int(k) not in set(indices)}}
        print(f"8B {cid}: cleaned={len(cleaned)} empty={len(empty)} "
              f"overlap_verified={n_ok}")
    C.dump_json(ext_prep, os.path.join(C8.OUT_DIR, "ext_prep.json"))

    abs_reqs = []
    for cid, name in sorted(mapping.items()):
        if int(name.rsplit("_k", 1)[1]) not in ABS_EXT_KS_8B:
            continue
        for idx in rest:
            row = texts[cid].get(str(idx))
            if row is None:
                continue
            abs_reqs.append({
                "custom_id": f"abs_{cid}_{idx}",
                "params": C.message_params(
                    C8.ABS_SYSTEM_8B,
                    C.ABS_USER_TEMPLATE.format(text=row["text"]),
                    C.ABS_SCHEMA)})
    write_jsonl(abs_reqs, os.path.join(C8.OUT_DIR, "ext_abs_requests.jsonl"))

    def budget_of(cid):
        return C8.BUDGET_OF_K[int(mapping[cid].rsplit("_k", 1)[1])]

    pair_reqs = build_pairs(
        mapping, budget_of, texts, pair_ext, C8.PAIR_SYSTEM_8B,
        os.path.join(C8.OUT_DIR, "ext_pair_requests.jsonl"))
    return abs_reqs, pair_reqs


def main():
    all_reqs = []
    for scale, (out_dir, prefixes), build in [
            ("110M", (C.OUT_DIR, ("",)), build_110m),
            ("8B", (C8.OUT_DIR, ("", "ct_")), build_8b)]:
        old = existing_ids(out_dir, prefixes)
        abs_reqs, pair_reqs = build()
        new = {r["custom_id"] for r in abs_reqs + pair_reqs}
        assert len(new) == len(abs_reqs) + len(pair_reqs), "dup ext ids"
        clash = new & old
        assert not clash, f"{scale}: {len(clash)} custom_id collisions"
        all_reqs += abs_reqs + pair_reqs
        print(f"{scale}: {len(abs_reqs)} abs + {len(pair_reqs)} pair, "
              f"no collisions vs {len(old)} existing ids")

    in_tok = sum(len(json.dumps(r["params"])) for r in all_reqs) / 3.6
    cost = (in_tok * IN_PRICE + len(all_reqs) * EST_OUT_TOK * OUT_PRICE) / 1e6
    print(f"\nTOTAL: {len(all_reqs)} requests, est input "
          f"{in_tok / 1e6:.1f} MTok, est cost ${cost:.0f} "
          f"(ceiling ${BUDGET_USD:.0f})")
    assert cost < BUDGET_USD, "cost estimate exceeds budget ceiling"
    print("BUILD_EXT1000_OK")


if __name__ == "__main__":
    main()
