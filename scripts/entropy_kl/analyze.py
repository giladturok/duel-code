"""Tables for the entropy / KL experiment.

  python analyze.py entropy  out/gen_*.npz
  python analyze.py kl       out/kl_*.npz
  python analyze.py steps    out/gen_*.npz
"""
import os
import sys

import numpy as np

RULES = ['block_greedy', 'block_left_to_right', 'block_probability_margin']
SHORT = {'block_greedy': 'greedy', 'block_left_to_right': 'l2r',
         'block_probability_margin': 'margin'}
SEQ_LEN = 1024
NTOK = SEQ_LEN - 1


def boot_ci(x, reps=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    means = x[idx].mean(axis=1)
    return np.quantile(means, [alpha / 2, 1 - alpha / 2])


def entropy_table(paths):
    rows = []
    for p in sorted(paths):
        d = np.load(p, allow_pickle=True)
        rb, pa = d['H_rb_per_token'], d['H_path_per_token']
        lo_r, hi_r = boot_ci(rb)
        lo_p, hi_p = boot_ci(pa)
        rows.append(dict(
            rule=SHORT.get(str(d['strategy']), str(d['strategy'])),
            k=int(d['k']), nfe=int(d['nfe']), n=len(rb),
            H_rb=rb.mean(), rb_lo=lo_r, rb_hi=hi_r,
            se_rb=rb.std(ddof=1) / np.sqrt(len(rb)),
            H_path=pa.mean(), path_lo=lo_p, path_hi=hi_p,
            se_path=pa.std(ddof=1) / np.sqrt(len(pa)),
            var_ratio=pa.var(ddof=1) / max(rb.var(ddof=1), 1e-30),
            agree=pa.mean() - rb.mean(),
            gen_ppl=float(d['gen_ppl']) if d['gen_ppl'].size else np.nan))
    rows.sort(key=lambda r: (r['nfe'], r['rule']))
    hdr = (f"{'rule':<8}{'k':>3}{'NFE':>6}{'n':>5}  {'H_rb (95% CI)':>26}"
           f"  {'H_path (95% CI)':>26}  {'path-rb':>9}{'var ratio':>10}{'genPPL':>9}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['rule']:<8}{r['k']:>3}{r['nfe']:>6}{r['n']:>5}  "
              f"{r['H_rb']:>8.4f} [{r['rb_lo']:.4f},{r['rb_hi']:.4f}]  "
              f"{r['H_path']:>8.4f} [{r['path_lo']:.4f},{r['path_hi']:.4f}]  "
              f"{r['agree']:>+9.4f}{r['var_ratio']:>10.1f}{r['gen_ppl']:>9.2f}")
    print('\nnats/token; CIs are 10k-resample bootstrap over sequences.')
    print('"path-rb" is the agreement of the two unbiased estimators of the same H(P_F).')
    print('"var ratio" = Var(path)/Var(RB) per sequence: the Rao-Blackwell gain.')
    return rows


def steps_table(paths):
    """Per-step entropy H_t, averaged over blocks and sequences."""
    for p in sorted(paths):
        d = np.load(p, allow_pickle=True)
        ent = d['ent_at_pos'][:, 1:]
        sib = d['step_in_block'][:, 1:]
        k, rule = int(d['k']), SHORT.get(str(d['strategy']), str(d['strategy']))
        n_steps = sib.max() + 1
        # H_t = total entropy revealed at within-block step t, summed over the
        # 64 blocks, averaged over sequences. sum_t H_t == H(P_F) exactly.
        tot = np.array([np.nansum(np.where(sib == t, ent, 0.0), axis=1).mean()
                        for t in range(n_steps)])
        print(f'\n{rule} k={k} NFE={int(d["nfe"])}  '
              f'sum_t H_t = {tot.sum() / NTOK:.4f} nats/token '
              f'(== H_rb {d["H_rb_per_token"].mean():.4f})')
        print('  H_t/token by within-block reveal step: ' +
              '  '.join(f'{t}:{v / NTOK:.4f}' for t, v in enumerate(tot)))


def kl_table(paths):
    cells = {}
    for p in sorted(paths):
        d = np.load(p, allow_pickle=True)
        a = SHORT.get(str(d['gen_strategy']), str(d['gen_strategy']))
        b = SHORT.get(str(d['score_strategy']), str(d['score_strategy']))
        k = int(d['gen_k'])
        cells[(k, a, b)] = d['kl_per_token']
    ks = sorted({c[0] for c in cells})
    order = ['greedy', 'l2r', 'margin']
    for k in ks:
        nfe = 1024 // k
        print(f'\nKL(P_A || P_B) per token, nats -- NFE {nfe} (k={k})')
        print(f"{'A \\ B':<10}" + ''.join(f'{b:>22}' for b in order))
        for a in order:
            line = f'{a:<10}'
            for b in order:
                v = cells.get((k, a, b))
                if v is None:
                    line += f'{"--":>22}'
                else:
                    lo, hi = boot_ci(v)
                    line += f'{v.mean():>10.4f} [{lo:.4f},{hi:.4f}]'.rjust(22)
            print(line)
        diag = [cells[(k, a, a)] for a in order if (k, a, a) in cells]
        if diag:
            m = max(np.abs(v).max() for v in diag)
            print(f'  sanity: max |per-sequence KL(P_A||P_A)| = {m:.3e} nats/token '
                  f'(hard gate, must be ~0)')
        off = [(a, b, cells[(k, a, b)]) for a in order for b in order
               if a != b and (k, a, b) in cells]
        bad = [(a, b, v.mean()) for a, b, v in off if v.mean() < 0]
        print(f'  sanity: {len(off) - len(bad)}/{len(off)} off-diagonal KLs >= 0'
              + (f'  NEGATIVE: {bad}' if bad else ''))


if __name__ == '__main__':
    what, files = sys.argv[1], sys.argv[2:]
    files = [f for f in files if os.path.exists(f)]
    {'entropy': entropy_table, 'kl': kl_table, 'steps': steps_table}[what](files)
