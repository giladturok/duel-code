"""Build the two 8B judge batch request files from out8b/prep.json.

Outputs
-------
out8b/abs_requests.jsonl   1600 = 16 configs x 100 sampled prefix indices.
out8b/pair_requests.jsonl  1440 = 4 budgets x 6 strategy pairs x 30
                           prefix-matched pairs (first 30 sampled indices)
                           x 2 presentation orders. Override the pair count
                           with --pairs N (budget fallback: 20).

custom_ids: abs_{cid}_{idx}, pair_{budget}_{cidA}_{cidB}_{idx}_{AB|BA}
(cidA < cidB lexically; AB = cidA shown as sample A). Requests carry only
opaque config ids b00..b15 — no strategy names anywhere in params.
"""
import argparse
import json
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import common8b as C8


def message_params_8b(system, user_text, schema):
    p = C.message_params(system, user_text, schema)
    return p


def write_jsonl(rows, path):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} requests -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=C8.N_PAIR_INDICES,
                    help="prefix-matched sample pairs per config pair")
    args = ap.parse_args()

    prep = C.load_json(os.path.join(C8.OUT_DIR, "prep.json"))
    blinding = C.load_json(os.path.join(C8.OUT_DIR, "blinding.json"))
    mapping = blinding["mapping"]  # bid -> real config name (grouping only)
    indices = prep["indices"]
    configs = prep["configs"]

    abs_reqs = []
    for cid in sorted(configs):
        for idx in indices:
            row = configs[cid]["rows"].get(str(idx))
            if row is None:
                continue
            abs_reqs.append({
                "custom_id": f"abs_{cid}_{idx}",
                "params": message_params_8b(
                    C8.ABS_SYSTEM_8B,
                    C.ABS_USER_TEMPLATE.format(text=row["text"]),
                    C.ABS_SCHEMA),
            })
    write_jsonl(abs_reqs, os.path.join(C8.OUT_DIR, "abs_requests.jsonl"))

    pair_indices = indices[:args.pairs]
    budget_groups = {}
    for cid, name in mapping.items():
        k = int(name.rsplit("_k", 1)[1])
        budget_groups.setdefault(C8.BUDGET_OF_K[k], []).append(cid)

    pair_reqs = []
    for budget in sorted(budget_groups):
        cids = sorted(budget_groups[budget])
        assert len(cids) == 4, (budget, cids)
        for cid_a, cid_b in combinations(cids, 2):
            for idx in pair_indices:
                row_a = configs[cid_a]["rows"].get(str(idx))
                row_b = configs[cid_b]["rows"].get(str(idx))
                if row_a is None or row_b is None:
                    continue
                for order in ("AB", "BA"):
                    first, second = ((row_a, row_b) if order == "AB"
                                     else (row_b, row_a))
                    pair_reqs.append({
                        "custom_id":
                            f"pair_{budget}_{cid_a}_{cid_b}_{idx}_{order}",
                        "params": message_params_8b(
                            C8.PAIR_SYSTEM_8B,
                            C.PAIR_USER_TEMPLATE.format(
                                a=first["text"], b=second["text"]),
                            C.PAIR_SCHEMA),
                    })
    write_jsonl(pair_reqs, os.path.join(C8.OUT_DIR, "pair_requests.jsonl"))


if __name__ == "__main__":
    main()
