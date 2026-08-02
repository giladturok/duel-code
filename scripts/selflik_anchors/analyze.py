"""E-B analysis: D_hat = log(gen_ppl) - log(duel_ppl), per passage, per-token nats.

Joins the two per-passage npz files on (cell, index), then reports per-cell means
with bootstrap 95% CIs over passages, plus the paired real-vs-repeat difference
(the cells share document indices, so the difference can be bootstrapped paired).

NOTE ON INTERPRETATION: the anchors are not samples from the model, so D_hat here
is NOT a reverse-KL estimate. It is only the scoring function evaluated on
known-good and known-degenerate text.
"""
import argparse
import json
import os

import numpy as np

CELLS = ['anchor-owt-real', 'anchor-owt-repeat',
         'anchor-owt-word-shuffled', 'anchor-owt-token-shuffled']


def boot_mean(x, n=20000, rng=None, alpha=0.05):
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duel', required=True)
    ap.add_argument('--genppl', required=True)
    ap.add_argument('--out-json', default=None)
    ap.add_argument('--boot', type=int, default=20000)
    args = ap.parse_args()

    d = np.load(args.duel, allow_pickle=False)
    g = np.load(args.genppl, allow_pickle=False)

    dkey = {(str(c), int(i)): j for j, (c, i) in enumerate(zip(d['cell'], d['index']))}
    gkey = {(str(c), int(i)): j for j, (c, i) in enumerate(zip(g['cell'], g['index']))}
    keys = sorted(set(dkey) & set(gkey))
    print(f'{len(keys)} passages joined '
          f'(duel {len(dkey)}, genppl {len(gkey)})')

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
            n_content=int(d['n_content'][jd]),
        ))

    rng = np.random.default_rng(0)
    summary = {}
    print(f'\n{"cell":28s} {"n":>4s} {"tok":>7s} {"duel_ppl":>10s} {"gen_ppl":>10s} '
          f'{"D_hat":>8s}  {"D_hat 95% CI":>20s}')
    for c in CELLS:
        if c not in rec:
            continue
        rows = sorted(rec[c], key=lambda r: r['index'])
        dp = np.array([r['duel_ppl'] for r in rows])
        gp = np.array([r['gen_ppl'] for r in rows])
        dh = np.log(gp) - np.log(dp)
        m_dp, lo_dp, hi_dp = boot_mean(dp, args.boot, rng)
        m_gp, lo_gp, hi_gp = boot_mean(gp, args.boot, rng)
        m_dh, lo_dh, hi_dh = boot_mean(dh, args.boot, rng)
        tok = np.mean([r['duel_tokens'] for r in rows])
        summary[c] = dict(n=len(rows), mean_tokens=float(tok),
                          duel_ppl=[m_dp, lo_dp, hi_dp],
                          gen_ppl=[m_gp, lo_gp, hi_gp],
                          d_hat=[m_dh, lo_dh, hi_dh],
                          median_d_hat=float(np.median(dh)))
        print(f'{c:28s} {len(rows):4d} {tok:7.1f} {m_dp:10.3f} {m_gp:10.3f} '
              f'{m_dh:8.4f}  [{lo_dh:7.4f}, {hi_dh:7.4f}]')

    # paired differences against anchor-owt-real
    if 'anchor-owt-real' in rec:
        base = {r['index']: r for r in rec['anchor-owt-real']}
        print(f'\npaired D_hat difference vs anchor-owt-real (positive = worse than real)')
        summary['paired_vs_real'] = {}
        for c in CELLS[1:]:
            if c not in rec:
                continue
            pairs = [(base[r['index']], r) for r in rec[c] if r['index'] in base]
            diff = np.array([
                (np.log(b['gen_ppl']) - np.log(b['duel_ppl']))
                for b, _ in pairs])
            diff = np.array([
                (np.log(o['gen_ppl']) - np.log(o['duel_ppl']))
                - (np.log(b['gen_ppl']) - np.log(b['duel_ppl']))
                for b, o in pairs])
            m, lo, hi = boot_mean(diff, args.boot, rng)
            frac = float(np.mean(diff > 0))
            summary['paired_vs_real'][c] = dict(n=len(pairs), mean=m, lo=lo, hi=hi,
                                                frac_worse=frac)
            print(f'  {c:28s} n={len(pairs):4d} delta={m:8.4f}  '
                  f'[{lo:7.4f}, {hi:7.4f}]  frac worse={frac:.3f}')

    # length-stratified control: the repeat anchor is systematically longer than the
    # real anchor, and per-token ppl falls with length, so check the sign survives
    # within narrow length bands.
    if 'anchor-owt-real' in rec and 'anchor-owt-repeat' in rec:
        print('\nlength-stratified check (bands on duel-scored token count):')
        allrows = [(c, r) for c in ('anchor-owt-real', 'anchor-owt-repeat') for r in rec[c]]
        bands = [(0, 256), (256, 512), (512, 768), (768, 1024)]
        summary['length_bands'] = {}
        for lo_b, hi_b in bands:
            out = {}
            for c in ('anchor-owt-real', 'anchor-owt-repeat'):
                sel = [r for r in rec[c] if lo_b <= r['duel_tokens'] < hi_b]
                if not sel:
                    continue
                dh = np.array([np.log(r['gen_ppl']) - np.log(r['duel_ppl']) for r in sel])
                out[c] = (len(sel), float(dh.mean()))
            if len(out) == 2:
                print(f'  [{lo_b:4d},{hi_b:4d})  real n={out["anchor-owt-real"][0]:4d} '
                      f'D_hat={out["anchor-owt-real"][1]:7.4f}   '
                      f'repeat n={out["anchor-owt-repeat"][0]:4d} '
                      f'D_hat={out["anchor-owt-repeat"][1]:7.4f}')
                summary['length_bands'][f'{lo_b}-{hi_b}'] = out

    if args.out_json:
        with open(args.out_json, 'w') as fh:
            json.dump(summary, fh, indent=2)
        print(f'\nwrote {args.out_json}')


if __name__ == '__main__':
    main()
