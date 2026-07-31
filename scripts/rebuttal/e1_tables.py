#!/usr/bin/env python3
"""Render the four E1 joint cost tables (R2 W2/Q1) from e1_parse.py output.

One table per model (MDLM, BD3-LM L'=4/8/16). DUEL rows and ELBO rows share a
frame, separated by a rule, because the xELBO column only reads if both are
present.

Forward counts are ANALYTIC, and deliberately so -- val/num_decoding_steps is
the quantity under dispute (it omits the block-commit forward,
exact_likelihood.py:180). The logged value is carried in its own column so the
discrepancy is visible rather than smoothed over.

Usage: python scripts/rebuttal/e1_tables.py [E1_ROOT]
"""
import json
import math
import os
import re
import sys

L = 1024
N_VAL_SEQS = 110480          # OWT validation split, L=1024, GPT-2 tokenizer
                             # (6905 batches x 16 in logs/mdlm_owt.log)


def fwd_per_seq(kind, algo, lp, k):
    """True forward passes per sequence."""
    nb = L // lp
    if kind == "elbo":
        return 1                       # one forward per pass, any L'
    if algo == "mdlm":
        # Uncached path: no commit forward, but every forward is full-length.
        return nb * math.ceil(lp / k)
    # Cached path: +1 commit forward per block, never counted by `steps`.
    return nb * (math.ceil(lp / k) + 1)


def fwd_reported(kind, algo, lp, k):
    """What val/num_decoding_steps reports (the paper's NFE x-axis)."""
    if kind == "elbo":
        return 1
    return (L // lp) * math.ceil(lp / k)


def fwd_len(kind, algo, lp):
    if kind == "elbo":
        # BD3-LM concatenates (xt, x0) -> 2L (diffusion.py:1495-1496,
        # configs/algo/bd3lm.yaml:9). MDLM has cross_attn: False.
        return 2 * L if algo == "bd3lm" else L
    return L if algo == "mdlm" else lp


def blocks_per_fwd(kind, algo, lp):
    """Blocks touched per forward (= t-draws per sequence, for the ELBO)."""
    nb = L // lp
    if kind == "elbo":
        # MDLM's ELBO runs at block_size = L, so num_blocks = 1: exactly ONE
        # t-draw per sequence per forward. BD3-LM gets L/L' stratified draws
        # inside the same single forward, for free. NOTE: this asymmetry does
        # NOT show up as extra ELBO variance for MDLM -- measured rel sd over 4
        # MC repeats is 2.26% (MDLM) vs 2.22-2.78% (BD3-LM). See
        # rebuttal/e1_110m_results.md section 1.
        return 1 if algo == "mdlm" else nb
    return nb if algo == "mdlm" else 1  # BD3-LM: 1 live block + cached prefix


ARM = re.compile(
    r"^(duel|elbo)__(mdlm|bd3lm)(\d+)__(?:(\w+?)__k(\d+)|rep(\d+))__bs(\d+)__(\w+?)(_fullsplit)?$")


def load(root):
    rows = json.load(open(os.path.join(root, "e1_results.json")))
    out = []
    for r in rows:
        m = ARM.match(r["arm"])
        if not m:
            continue                    # calibration / sweep arms
        kind, algo, blk, sampler, k, rep, bs, backend, full = m.groups()
        lp = 4 if algo == "mdlm" and kind == "duel" else int(blk)
        if algo == "mdlm" and kind == "elbo":
            lp = 4                      # table grouping only; block_size was L
        r.update(kind=kind, algo=algo, lp=lp, blk=int(blk),
                 sampler=sampler, k=int(k) if k else None,
                 rep=int(rep) if rep else None, bs=int(bs),
                 backend=backend, fullsplit=bool(full))
        out.append(r)
    return out


def fmt(x, n=2):
    return "--" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def table(rows, algo, lp, title):
    sel = [r for r in rows if r["algo"] == algo and r["lp"] == lp
           and not r["fullsplit"] and r.get("exit_code") == "0"
           and r.get("s_per_batch") is not None]
    duel = sorted([r for r in sel if r["kind"] == "duel"],
                  key=lambda r: (r["sampler"], r["k"]))
    elbo = sorted([r for r in sel if r["kind"] == "elbo"],
                  key=lambda r: (r["backend"] != "sdpa", r["rep"]))

    # xELBO baseline = the K=1 sdpa ELBO on the same node, same sequences.
    base = next((r["s_per_batch"] for r in elbo
                 if r["backend"] == "sdpa" and r["rep"] == 1), None)

    lines = [f"### {title}", "",
             "| arm | real fwd/seq | reported fwd/seq | fwd length | blocks/fwd |"
             " s/batch (B=32) | s/seq | full split (GPU-h) | peak GPU mem (MiB) | ppl | xELBO |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(r, label, k, kind):
        spb, sig = r.get("s_per_batch"), r.get("s_per_batch_sigma")
        sseq = spb / r["bs"] if spb else None
        xe = spb / base if (spb and base) else None
        return (f"| {label} | {fwd_per_seq(kind, algo, lp, k or 1)} "
                f"| {fwd_reported(kind, algo, lp, k or 1)} "
                f"| {fwd_len(kind, algo, lp)} | {blocks_per_fwd(kind, algo, lp)} "
                f"| {fmt(spb)} ± {fmt(sig)} | {fmt(sseq, 4)} "
                f"| {fmt(sseq * N_VAL_SEQS / 3600, 2) if sseq else '--'} "
                f"| {r.get('peak_mem_mib', '--')} | {fmt(r.get('ppl'))} "
                f"| {fmt(xe, 1)}x |")

    for r in duel:
        mark = ""
        if r["k"] == 1:
            mark = " **(paper protocol)**"
        elif r["k"] == lp:
            mark = " **(k=L', native parallel)**"
        lines.append(row(r, f"DUEL {r['sampler'].replace('block_', '')} k={r['k']}{mark}",
                         r["k"], "duel"))

    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    # MC=K is K repeated passes: cost multiplies exactly, ppl is the mean in
    # NLL space over the first K reps (same sequences, independent t).
    sdpa = [r for r in elbo if r["backend"] == "sdpa"]
    for K in (1, 2, 4):
        rs = [r for r in sdpa if r["rep"] and r["rep"] <= K]
        if len(rs) < K:
            continue
        spb = sum(r["s_per_batch"] for r in rs[:1]) * K
        nlls = [math.log(r["ppl"]) for r in rs if r.get("ppl")]
        ppl = math.exp(sum(nlls) / len(nlls)) if nlls else None
        spread = (max(r["ppl"] for r in rs) - min(r["ppl"] for r in rs)) if K > 1 else 0.0
        lines.append(
            f"| ELBO K={K}{' **(paper protocol)**' if K == 1 else ''} "
            f"| {K} | {K} | {fwd_len('elbo', algo, lp)} "
            f"| {blocks_per_fwd('elbo', algo, lp)} | {fmt(spb)} "
            f"| {fmt(spb / 32, 4)} "
            f"| {fmt(spb / 32 * N_VAL_SEQS / 3600, 2)} "
            f"| {rs[0].get('peak_mem_mib', '--')} "
            f"| {fmt(ppl)}{f' (spread {spread:.2f})' if K > 1 else ''} "
            f"| {fmt(spb / base, 1) if base else '--'}x |")

    for r in elbo:
        if r["backend"] != "sdpa":
            lines.append(row(r, f"ELBO K=1 [{r['backend']}]", 1, "elbo"))

    lines.append("")
    if algo == "mdlm":
        lines.append(
            "MDLM has **no native block size**: L'=4 here is an *evaluation* "
            "choice (the decoding order is blocked), not a property of the "
            "checkpoint. Its ELBO row uses `block_size = L = 1024`, i.e. one "
            "t-draw per sequence. MDLM also cannot use the KV cache "
            "(`diffusion.py:583-590`), so every DUEL forward is full-length.")
    lines.append("")
    return "\n".join(lines)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "logs/e1_cost"
    rows = load(root)
    print(table(rows, "mdlm", 4, "MDLM (L'=4, evaluation blocking)"))
    for lp in (4, 8, 16):
        print(table(rows, "bd3lm", lp, f"BD3-LM L'={lp}"))
    print(f"\nFull-split extrapolation basis: {N_VAL_SEQS} sequences "
          f"= {-(-N_VAL_SEQS // 32)} batches at B=32.")


if __name__ == "__main__":
    main()
