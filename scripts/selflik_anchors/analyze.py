"""E-B analysis on the four CoLA anchor cells.

Three statistics, per passage, reported in this order:

  1. duel_ppl        -- DUEL exact perplexity under block_greedy k=1. REFERENCE-FREE.
                        Lower = the model assigns the passage higher likelihood.
  2. gen_ppl         -- GPT-2 Large perplexity of the same passage.
  3. D_hat           -- log(gen_ppl) - log(duel_ppl), per-token nats.
                        = (1/T) * log[ p_duel(x) / p_gpt2(x) ]. Lower = the DUEL
                        model likes the passage less, relative to GPT-2 Large.

INTERPRETATION WARNING: the anchors are NOT samples from the model, so D_hat here
is NOT a reverse-KL estimate; it is only the scoring function applied to known-good
and known-degenerate text. Likewise duel_ppl here is the likelihood of *given* text,
not of the model's own samples.

Bootstrap CIs are over passages. The four cells share document indices (the repeat
anchor at index i is built from the real anchor at index i), so the real-vs-repeat
comparison is also reported paired.
"""
import argparse
import json

import numpy as np

CELLS = ['anchor-owt-real', 'anchor-owt-repeat',
         'anchor-owt-word-shuffled', 'anchor-owt-token-shuffled']


def boot_mean(x, n, rng, alpha=0.05):
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def table(rec, key, transform, label, rng, nboot, summary, fmt='{:12.4f}'):
    print(f'\n=== {label} ===')
    print(f'{"cell":28s} {"n":>5s} {"mean":>12s} {"95% CI":>28s} {"median":>12s}')
    out = {}
    for c in CELLS:
        if c not in rec:
            continue
        v = np.array([transform(r) for r in rec[c]])
        m, lo, hi = boot_mean(v, nboot, rng)
        out[c] = dict(n=len(v), mean=m, lo=lo, hi=hi, median=float(np.median(v)))
        print(f'{c:28s} {len(v):5d} ' + fmt.format(m) +
              f'  [{fmt.format(lo)}, {fmt.format(hi)}]  ' + fmt.format(np.median(v)))
    summary[key] = out
    return out


def paired(rec, transform, label, rng, nboot, summary, key):
    """Paired difference other - real, on document indices present in both."""
    base = {r['index']: r for r in rec['anchor-owt-real']}
    print(f'\n--- paired difference vs anchor-owt-real: {label} ---')
    out = {}
    for c in CELLS[1:]:
        if c not in rec:
            continue
        pairs = [(base[r['index']], r) for r in rec[c] if r['index'] in base]
        if not pairs:
            continue
        d = np.array([transform(o) - transform(b) for b, o in pairs])
        m, lo, hi = boot_mean(d, nboot, rng)
        frac = float(np.mean(d > 0))
        out[c] = dict(n=len(pairs), mean=m, lo=lo, hi=hi, frac_positive=frac)
        print(f'  {c:28s} n={len(pairs):5d}  delta={m:11.4f}  '
              f'[{lo:11.4f}, {hi:11.4f}]  frac(other>real)={frac:.3f}')
    summary[key] = out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duel', required=True, nargs='+',
                    help='one or more per-passage DUEL npz files (concatenated)')
    ap.add_argument('--genppl', required=True)
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--boot', type=int, default=20000)
    args = ap.parse_args()

    parts = [np.load(p, allow_pickle=False) for p in args.duel]
    d = {k: np.concatenate([p[k] for p in parts])
         for k in ('cell', 'index', 'n_content', 'nll_per_token', 'answer_lens', 'duel_ppl')}
    g = np.load(args.genppl, allow_pickle=False)

    dkey = {(str(c), int(i)): j for j, (c, i) in enumerate(zip(d['cell'], d['index']))}
    gkey = {(str(c), int(i)): j for j, (c, i) in enumerate(zip(g['cell'], g['index']))}
    keys = sorted(set(dkey) & set(gkey))
    print(f'{len(keys)} passages joined (duel {len(dkey)}, genppl {len(gkey)})')

    rec = {}
    for c, i in keys:
        jd, jg = dkey[(c, i)], gkey[(c, i)]
        rec.setdefault(c, []).append(dict(
            index=i,
            duel_ppl=float(d['duel_ppl'][jd]),
            duel_nll=float(d['nll_per_token'][jd]),
            duel_tokens=int(d['answer_lens'][jd]),
            gen_ppl=float(g['gen_ppl'][jg]),
            gen_tokens=float(g['gen_lengths'][jg]),
        ))
    for c in rec:
        rec[c].sort(key=lambda r: r['index'])

    rng = np.random.default_rng(0)
    summary = {'n_joined': len(keys)}

    print('\nmean DUEL-scored token count per cell:')
    for c in CELLS:
        if c in rec:
            print(f'  {c:28s} {np.mean([r["duel_tokens"] for r in rec[c]]):8.1f}')

    # ---- 1. DUEL exact perplexity ALONE (reference-free) --------------------
    table(rec, 'duel_ppl', lambda r: r['duel_ppl'],
          '1. DUEL exact perplexity (reference-free; lower = higher model likelihood)',
          rng, args.boot, summary)
    table(rec, 'duel_nll', lambda r: r['duel_nll'],
          '1b. DUEL exact NLL per token (nats)', rng, args.boot, summary)
    for c in CELLS:
        if c in rec:
            nll = np.array([r['duel_nll'] for r in rec[c]])
            w = np.array([r['duel_tokens'] for r in rec[c]])
            summary.setdefault('duel_micro_ppl', {})[c] = float(np.exp((nll * w).sum() / w.sum()))
    print('\ntoken-weighted micro-average duel_ppl (the val/exact_ppl convention):')
    for c, v in summary.get('duel_micro_ppl', {}).items():
        print(f'  {c:28s} {v:12.4f}')
    paired(rec, lambda r: r['duel_nll'], 'DUEL NLL/token (positive = real is better)',
           rng, args.boot, summary, 'paired_duel_nll')

    # ---- 2. GPT-2 Large generative perplexity -------------------------------
    table(rec, 'gen_ppl', lambda r: r['gen_ppl'],
          '2. GPT-2 Large generative perplexity', rng, args.boot, summary)

    # ---- 3. D_hat -----------------------------------------------------------
    table(rec, 'd_hat', lambda r: np.log(r['gen_ppl']) - np.log(r['duel_ppl']),
          '3. D_hat = log(gen_ppl) - log(duel_ppl), per-token nats (lower = better)',
          rng, args.boot, summary)
    paired(rec, lambda r: np.log(r['gen_ppl']) - np.log(r['duel_ppl']),
           'D_hat (positive = worse than real)', rng, args.boot, summary, 'paired_d_hat')

    # ---- length-stratified control ------------------------------------------
    # The repeat anchor is systematically longer than the real anchor and per-token
    # perplexity falls with length, so check that the signs survive within bands.
    if 'anchor-owt-real' in rec and 'anchor-owt-repeat' in rec:
        print('\n--- length-stratified control (bands on DUEL-scored token count) ---')
        print(f'{"band":>12s}  {"n_real":>6s} {"duel_real":>10s} {"Dhat_real":>10s}   '
              f'{"n_rep":>6s} {"duel_rep":>10s} {"Dhat_rep":>10s}')
        summary['length_bands'] = {}
        for lo_b, hi_b in [(0, 256), (256, 512), (512, 768), (768, 1024)]:
            row = {}
            for c in ('anchor-owt-real', 'anchor-owt-repeat'):
                sel = [r for r in rec[c] if lo_b <= r['duel_tokens'] < hi_b]
                if sel:
                    row[c] = dict(
                        n=len(sel),
                        duel_ppl=float(np.mean([r['duel_ppl'] for r in sel])),
                        d_hat=float(np.mean([np.log(r['gen_ppl']) - np.log(r['duel_ppl'])
                                             for r in sel])))
            if len(row) == 2:
                a, b = row['anchor-owt-real'], row['anchor-owt-repeat']
                print(f'[{lo_b:4d},{hi_b:4d})  {a["n"]:6d} {a["duel_ppl"]:10.3f} '
                      f'{a["d_hat"]:10.4f}   {b["n"]:6d} {b["duel_ppl"]:10.3f} '
                      f'{b["d_hat"]:10.4f}')
                summary['length_bands'][f'{lo_b}-{hi_b}'] = row

    # ---- verdicts -----------------------------------------------------------
    print('\n=== VERDICTS (real vs repeat) ===')
    for name, key, better_is in [('DUEL exact ppl alone', 'duel_ppl', 'lower'),
                                 ('D_hat', 'd_hat', 'lower')]:
        s = summary[key]
        if 'anchor-owt-real' not in s or 'anchor-owt-repeat' not in s:
            continue
        r, p = s['anchor-owt-real'], s['anchor-owt-repeat']
        real_better = r['mean'] < p['mean']
        disjoint = (r['hi'] < p['lo']) or (p['hi'] < r['lo'])
        verdict = 'PASS' if (real_better and disjoint) else 'FAIL'
        print(f'{name:24s}: real={r["mean"]:.4f} [{r["lo"]:.4f},{r["hi"]:.4f}]  '
              f'repeat={p["mean"]:.4f} [{p["lo"]:.4f},{p["hi"]:.4f}]  '
              f'CIs disjoint={disjoint}  real strictly better={real_better}  -> {verdict}')
        summary.setdefault('verdicts', {})[key] = verdict

    if args.out_json:
        with open(args.out_json, 'w') as fh:
            json.dump(summary, fh, indent=2)
        print(f'\nwrote {args.out_json}')


if __name__ == '__main__':
    main()
