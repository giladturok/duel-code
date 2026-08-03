"""Unblind and analyze the 8B prefix-grid judge results; emit out8b/summary.md.

Inputs:  out8b/blinding.json, out8b/prep.json, out8b/abs_results.jsonl,
         out8b/pair_results.jsonl
Outputs: out8b/summary.md (+ printed to stdout)

Correlation rows are emitted only for configs with a non-None value in
common8b.DUEL_PPL_8B / common8b_ct.DUEL_PPL_CT — fill those dicts (keyed
by config name) when the scoring jobs land and rerun.

CT add-on (nuc variant): if out8b_nuc/ct_prep.json and ct_*_results.jsonl
exist, the 4 confidence-threshold arms (n16..n19) are folded into the
absolute table, doc-break slices, length-bias check, and each budget
cell's pairwise matrix (CT rows only against the 4 samplers of the cell).
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import os as _os
if _os.environ.get("JUDGE8B_VARIANT") == "nuc":
    import common8b_nuc as C8
else:
    import common8b as C8
import common8b_ct as CT
import analyze as A  # spearman, perm_p, boot_ci, wilson helpers


def short_name(name):
    if name.startswith("ct_tau"):
        return f"CT tau{name[len('ct_tau'):]}"
    strategy, k = name.rsplit("_k", 1)
    return f"{ {'left_to_right': 'l2r'}.get(strategy, strategy) } k{k}".strip()


def budget_of(name):
    if name.startswith("ct_tau"):
        return CT.CT_BUDGET_OF[name]
    return C8.BUDGET_OF_K[int(name.rsplit("_k", 1)[1])]


def main():
    blinding = C.load_json(os.path.join(C8.OUT_DIR, "blinding.json"))
    mapping = blinding["mapping"]                # bid -> real name
    inv = {v: k for k, v in mapping.items()}
    prep = C.load_json(os.path.join(C8.OUT_DIR, "prep.json"))
    abs_results = C.load_jsonl(os.path.join(C8.OUT_DIR, "abs_results.jsonl"))
    pair_results = C.load_jsonl(os.path.join(C8.OUT_DIR, "pair_results.jsonl"))

    # CT add-on: fold the 4 confidence-threshold arms in when present.
    ct_paths = [os.path.join(C8.OUT_DIR, p) for p in
                ("ct_prep.json", "ct_abs_results.jsonl",
                 "ct_pair_results.jsonl")]
    have_ct = all(os.path.exists(p) for p in ct_paths)
    all_configs = list(C8.CONFIGS)
    duel_ppl = dict(C8.DUEL_PPL_8B)
    if have_ct:
        ct_prep = C.load_json(ct_paths[0])
        assert ct_prep["indices"] == prep["indices"]
        prep["configs"].update(ct_prep["configs"])
        abs_results += C.load_jsonl(ct_paths[1])
        pair_results += C.load_jsonl(ct_paths[2])
        all_configs += CT.CT_CONFIGS
        duel_ppl.update(CT.DUEL_PPL_CT)

    lines = []
    emit = lines.append

    emit(f"# LLM-judge study: {C8.STUDY_LABEL}")
    emit("")
    emit(f"Judge: `{C.MODEL}` (effort={C.EFFORT}, structured outputs), "
         f"blinded configs, seed {C8.SEED}. Budgets: k=1/2/4/8 -> NFE "
         "256/128/64/32.")
    emit("")
    emit(C8.STUDY_NOTE)
    emit("")
    if have_ct:
        emit(CT.CT_NOTE)
        emit("")

    # ---------------- absolute ratings
    abs_by_cfg = defaultdict(lambda: defaultdict(dict))  # name->dim->{idx: s}
    n_missing_abs = 0
    for r in abs_results:
        if r["status"] != "ok":
            n_missing_abs += 1
            continue
        _, bid, idx = r["custom_id"].split("_")
        name = mapping[bid]
        for dim in C.ABS_DIMS:
            abs_by_cfg[name][dim][int(idx)] = float(r["parsed"][dim])

    order = sorted(all_configs, key=lambda nm: (-budget_of(nm), nm))

    emit("## Absolute ratings (mean [95% bootstrap CI], n<=100 per config)")
    emit("")
    emit("| config | NFE | " + " | ".join(C.ABS_DIMS) + " | n |")
    emit("|" + "---|" * (len(C.ABS_DIMS) + 3))
    stats = {}
    for name in order:
        dims = abs_by_cfg[name]
        cells = []
        for dim in C.ABS_DIMS:
            m, lo, hi = A.boot_ci(list(dims[dim].values()))
            stats.setdefault(name, {})[dim] = m
            cells.append(f"{m:.2f} [{lo:.2f},{hi:.2f}]")
        n = len(dims["overall"])
        nfe = budget_of(name)
        nfe_cell = f"~{nfe}" if name.startswith("ct_tau") else str(nfe)
        emit(f"| {short_name(name)} | {nfe_cell} | " + " | ".join(cells) +
             f" | {n} |")
    emit("")
    if n_missing_abs:
        emit(f"Missing/failed absolute ratings: {n_missing_abs}.")
        emit("")

    # ---------------- slices by prefix metadata
    for flag, label in [
            ("full_length_continuation",
             "true continuation is full-length (256 tok)"),
            ("truncated_at_doc_break",
             "sample was truncated at a <|endoftext|> break")]:
        emit(f"## Overall score sliced by {label}")
        emit("")
        emit("| config | mean (flag=True) | n | mean (flag=False) | n |")
        emit("|---|---|---|---|---|")
        for name in order:
            cfg = prep["configs"][inv[name]]
            groups = {True: [], False: []}
            for idx, sc in abs_by_cfg[name]["overall"].items():
                row = cfg["rows"].get(str(idx))
                if row is not None:
                    groups[bool(row[flag])].append(sc)
            t, f = groups[True], groups[False]
            mt = f"{np.mean(t):.2f}" if t else "n/a"
            mf = f"{np.mean(f):.2f}" if f else "n/a"
            emit(f"| {short_name(name)} | {mt} | {len(t)} | {mf} | "
                 f"{len(f)} |")
        emit("")

    # ---------------- correlations vs DUEL ppl (hook: None entries skipped)
    emit("## Correlation vs DUEL ppl")
    emit("")
    have = [nm for nm in order if duel_ppl.get(nm) is not None]
    pending = [nm for nm in order if duel_ppl.get(nm) is None]
    if len(have) < 3:
        emit("TODO: skipped — `DUEL_PPL_8B` has no values yet. Fill it "
             "and rerun.")
    else:
        emit("| config | DUEL ppl | judge overall |")
        emit("|---|---|---|")
        for nm in have:
            emit(f"| {short_name(nm)} | {duel_ppl[nm]:.2f} | "
                 f"{stats[nm]['overall']:.2f} |")
        emit("")
        duel = [duel_ppl[nm] for nm in have]
        overall = [stats[nm]["overall"] for nm in have]
        rho, p = A.perm_p(overall, duel)
        emit(f"Spearman(judge overall, DUEL ppl) over {len(have)} configs: "
             f"**rho={rho:+.3f}**, permutation p={p:.4f} (10k).")
    if pending:
        emit("")
        emit("Pending DUEL-ppl values (separate scoring job; fill "
             "`DUEL_PPL_CT` in common8b_ct.py and rerun): "
             + ", ".join(short_name(nm) for nm in pending) + ".")
    emit("")

    # ---------------- pairwise, within budget
    units = defaultdict(dict)
    n_missing_pair = 0
    for r in pair_results:
        if r["status"] != "ok":
            n_missing_pair += 1
            continue
        _, budget, cid_a, cid_b, idx, present = r["custom_id"].split("_")
        w = r["parsed"]["winner"]
        if w == "tie":
            winner = "tie"
        elif present == "AB":
            winner = cid_a if w == "A" else cid_b
        else:
            winner = cid_b if w == "A" else cid_a
        units[(int(budget), cid_a, cid_b, int(idx))][present] = winner

    emit("## Within-budget pairwise win rates "
         "(row beats column; order-averaged, tie=0.5)")
    emit("")
    for budget in sorted({b for b, *_ in units}, reverse=True):
        cfgs = [nm for nm in order if budget_of(nm) == budget]
        bids = [inv[nm] for nm in cfgs]
        score, nunits = {}, {}
        flips_agree, flips_total = 0, 0
        for (b, ca, cb, idx), verdicts in units.items():
            if b != budget:
                continue
            vals = [1.0 if w == ca else 0.5 if w == "tie" else 0.0
                    for w in verdicts.values()]
            if len(verdicts) == 2:
                flips_total += 1
                v = list(verdicts.values())
                if v[0] == v[1]:
                    flips_agree += 1
            u = float(np.mean(vals))
            score[(ca, cb)] = score.get((ca, cb), 0.0) + u
            nunits[(ca, cb)] = nunits.get((ca, cb), 0) + 1
        emit(f"### Budget ~{budget} NFE")
        emit("")
        emit("| | " + " | ".join(short_name(nm) for nm in cfgs) + " |")
        emit("|" + "---|" * (len(cfgs) + 1))
        for i, (ri, rnm) in enumerate(zip(bids, cfgs)):
            row = [f"**{short_name(rnm)}**"]
            for j, cj in enumerate(bids):
                if i == j:
                    row.append("—")
                    continue
                key = (ri, cj) if (ri, cj) in score else (cj, ri)
                if key not in score:
                    row.append("n/a")
                    continue
                n = nunits[key]
                s = score[key] if key == (ri, cj) else n - score[key]
                lo, hi = A.wilson(s, n)
                row.append(f"{s / n:.2f} [{lo:.2f},{hi:.2f}]")
            emit("| " + " | ".join(row) + " |")
        emit("")
        if flips_total:
            emit(f"Flip-consistency (AB/BA verdicts agree): "
                 f"{flips_agree}/{flips_total} = "
                 f"{flips_agree / flips_total:.2f}")
            emit("")
    if n_missing_pair:
        emit(f"Missing/failed pairwise judgments: {n_missing_pair}.")
        emit("")

    # ---------------- length-bias check
    emit("## Length-bias check: Spearman(orig_words, overall) within config")
    emit("")
    emit("| config | rho | n |")
    emit("|---|---|---|")
    for name in order:
        cfg = prep["configs"][inv[name]]
        xs, ys = [], []
        for idx, sc in abs_by_cfg[name]["overall"].items():
            row = cfg["rows"].get(str(idx))
            if row is not None:
                xs.append(row["orig_words"])
                ys.append(sc)
        rho = A.spearman(xs, ys) if len(xs) > 2 else float("nan")
        emit(f"| {short_name(name)} | {rho:+.3f} | {len(xs)} |")
    emit("")

    text = "\n".join(lines) + "\n"
    out_path = os.path.join(C8.OUT_DIR, "summary.md")
    with open(out_path, "w") as fh:
        fh.write(text)
    print(text)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
