"""Build the two Message Batches request files from out/prep.json.

Outputs
-------
out/abs_requests.jsonl    1600 = 16 configs x 100 rows, absolute ratings.
out/pair_requests.jsonl   2400 = 4 budgets x 6 sampler pairs x 50
                          index-matched sample pairs (first 50 sampled
                          indices) x 2 presentation orders.

custom_ids: abs_{cid}_{idx}, pair_{budget}_{cidA}_{cidB}_{idx}_{AB|BA}
(cidA < cidB lexically; AB = cidA shown as sample A). Requests carry only
opaque config ids — no sampler names anywhere in params.
"""
import json
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def write_jsonl(rows, path):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} requests -> {path}")


def main():
    prep = C.load_json(os.path.join(C.OUT_DIR, "prep.json"))
    blinding = C.load_json(os.path.join(C.OUT_DIR, "blinding.json"))
    mapping = blinding["mapping"]  # cid -> real name (used for grouping only)
    indices = prep["indices"]
    configs = prep["configs"]

    # -------- absolute ratings
    abs_reqs = []
    for cid in sorted(configs):
        for idx in indices:
            row = configs[cid]["rows"].get(str(idx))
            if row is None:
                continue
            abs_reqs.append({
                "custom_id": f"abs_{cid}_{idx}",
                "params": C.message_params(
                    C.ABS_SYSTEM,
                    C.ABS_USER_TEMPLATE.format(text=row["text"]),
                    C.ABS_SCHEMA),
            })
    write_jsonl(abs_reqs, os.path.join(C.OUT_DIR, "abs_requests.jsonl"))

    # -------- pairwise, within-budget
    pair_indices = indices[:C.N_PAIR_INDICES]
    budget_groups = {}
    for cid, name in mapping.items():
        budget_groups.setdefault(C.BUDGET_OF[name], []).append(cid)

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
                        "params": C.message_params(
                            C.PAIR_SYSTEM,
                            C.PAIR_USER_TEMPLATE.format(
                                a=first["text"], b=second["text"]),
                            C.PAIR_SCHEMA),
                    })
    write_jsonl(pair_reqs, os.path.join(C.OUT_DIR, "pair_requests.jsonl"))


if __name__ == "__main__":
    main()
