"""Build the CT add-on batch request files from out8b_nuc/{prep,ct_prep}.json.

Outputs
-------
out8b_nuc/ct_abs_requests.jsonl   512 = 4 CT arms x 128 sampled prefix
                                  indices (same ABS prompt/schema).
out8b_nuc/ct_pair_requests.jsonl  <=960 = 4 budget cells x (CT vs each of
                                  the 4 existing samplers of the cell) x 30
                                  prefix-matched pairs x 2 presentation
                                  orders. Existing-sampler texts come from
                                  out8b_nuc/prep.json (same prep pipeline);
                                  existing configs appear ONLY as
                                  comparison sides. Override the pair
                                  count with --pairs N (cost fallback: 20).

custom_ids: abs_{cid}_{idx}, pair_{budget}_{cidA}_{cidB}_{idx}_{AB|BA}
(cidA < cidB lexically, so cidA is always the existing n00..n15 side and
cidB the CT n16..n19 side; AB = cidA shown as sample A). Requests carry
only opaque ids — no strategy/tau names anywhere in params.

Prints a batch-API cost estimate (Sonnet-5 batched intro $1/$5 per MTok,
intro pricing through 2026-08-31) from character arithmetic.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("JUDGE8B_VARIANT", "nuc")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import common8b_nuc as C8
import common8b_ct as CT

# Batched intro pricing (claude-sonnet-5, 50% batch discount on $2/$10).
PRICE_IN_PER_MTOK = 1.0
PRICE_OUT_PER_MTOK = 5.0
CHARS_PER_TOKEN = 4.0
# Observed on the completed nucleus study (same prompts/model/effort):
# parsed JSON answers average ~55 tokens (abs) / ~45 (pair); pad for
# thinking + wrapper.
EST_OUT_TOKENS = {"abs": 150, "pair": 150}


def write_jsonl(rows, path):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} requests -> {path}")


def estimate_cost(reqs_by_arm):
    total = 0.0
    for arm, reqs in reqs_by_arm.items():
        in_chars = sum(len(r["params"]["system"])
                       + len(r["params"]["messages"][0]["content"])
                       for r in reqs)
        in_tok = in_chars / CHARS_PER_TOKEN + 30 * len(reqs)  # msg overhead
        out_tok = EST_OUT_TOKENS[arm] * len(reqs)
        cost = (in_tok * PRICE_IN_PER_MTOK
                + out_tok * PRICE_OUT_PER_MTOK) / 1e6
        total += cost
        print(f"  {arm}: {len(reqs)} reqs, ~{in_tok / 1e3:.0f}k in + "
              f"~{out_tok / 1e3:.0f}k out tok -> ~${cost:.2f}")
    print(f"  TOTAL ~= ${total:.2f} (batched intro $1/$5 per MTok)")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=CT.N_CT_PAIR_INDICES,
                    help="prefix-matched sample pairs per (CT, sampler) pair")
    args = ap.parse_args()

    prep = C.load_json(os.path.join(CT.OUT_DIR, "prep.json"))
    ct_prep = C.load_json(os.path.join(CT.OUT_DIR, "ct_prep.json"))
    blinding = C.load_json(os.path.join(CT.OUT_DIR, "blinding.json"))
    mapping = blinding["mapping"]
    assert ct_prep["indices"] == prep["indices"]
    indices = prep["indices"]

    rows_of = {}  # cid -> {idx_str: row}, both studies
    for cid, cfg in list(prep["configs"].items()) + \
            list(ct_prep["configs"].items()):
        rows_of[cid] = cfg["rows"]

    ct_cids = sorted(ct_prep["configs"])
    assert ct_cids == [f"n{CT.CT_ID_START + i:02d}" for i in range(4)]

    abs_reqs = []
    for cid in ct_cids:
        for idx in indices:
            row = rows_of[cid].get(str(idx))
            if row is None:
                continue
            abs_reqs.append({
                "custom_id": f"abs_{cid}_{idx}",
                "params": C.message_params(
                    C8.ABS_SYSTEM_8B,
                    C.ABS_USER_TEMPLATE.format(text=row["text"]),
                    C.ABS_SCHEMA),
            })

    pair_indices = indices[:args.pairs]
    ct_cid_of_budget = {CT.CT_BUDGET_OF[mapping[cid]]: cid
                        for cid in ct_cids}
    sampler_cids_of_budget = {}
    for cid, name in mapping.items():
        if name in CT.CT_BUDGET_OF:
            continue
        k = int(name.rsplit("_k", 1)[1])
        sampler_cids_of_budget.setdefault(
            C8.BUDGET_OF_K[k], []).append(cid)

    pair_reqs = []
    for budget in sorted(ct_cid_of_budget, reverse=True):
        ct_cid = ct_cid_of_budget[budget]
        opponents = sorted(sampler_cids_of_budget[budget])
        assert len(opponents) == 4, (budget, opponents)
        for opp in opponents:
            cid_a, cid_b = sorted([opp, ct_cid])
            assert (cid_a, cid_b) == (opp, ct_cid)  # n00..n15 < n16..n19
            for idx in pair_indices:
                row_a = rows_of[cid_a].get(str(idx))
                row_b = rows_of[cid_b].get(str(idx))
                if row_a is None or row_b is None:
                    continue
                for order in ("AB", "BA"):
                    first, second = ((row_a, row_b) if order == "AB"
                                     else (row_b, row_a))
                    pair_reqs.append({
                        "custom_id":
                            f"pair_{budget}_{cid_a}_{cid_b}_{idx}_{order}",
                        "params": C.message_params(
                            C8.PAIR_SYSTEM_8B,
                            C.PAIR_USER_TEMPLATE.format(
                                a=first["text"], b=second["text"]),
                            C.PAIR_SCHEMA),
                    })

    print("cost estimate:")
    estimate_cost({"abs": abs_reqs, "pair": pair_reqs})
    write_jsonl(abs_reqs, os.path.join(CT.OUT_DIR, "ct_abs_requests.jsonl"))
    write_jsonl(pair_reqs, os.path.join(CT.OUT_DIR, "ct_pair_requests.jsonl"))


if __name__ == "__main__":
    main()
