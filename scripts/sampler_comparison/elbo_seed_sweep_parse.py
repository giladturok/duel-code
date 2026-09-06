"""Parse the ELBO seed sweep and compare its spread to the enumerated bound.

See scripts/sampler_comparison/elbo_seed_sweep.sh for why.
"""
import glob, math, re, statistics, sys

# Reference values, all AG News / bd3lm-owt-block_size4 / 384 seqs x 1023 tokens.
EXACT_UNIFORM_NLL = 4.106184482574463   # logs/..._decomp_oracle_4gpufix.log
PRIOR_ELBO = {                          # two draws of the SAME elbo config
    '2026-03-24 (paper Tab.7)': 4.121850490570068,
    '2026-07-28 (decomposition)': 4.130551815032959,
}
N_BLOCKS = 384 * (1024 // 4)
VALID_VAR = 34.242889404296875          # per-block variance, decomp_elbo run

pat_nll = re.compile(r'val/nll\s*│\s*([0-9.]+)')
rows = []
for path in sorted(glob.glob('logs/elbo_seed/*.log')):
    seed = int(re.search(r'seed(\d+)', path).group(1))
    m = pat_nll.findall(open(path, errors='ignore').read())
    if m:
        rows.append((seed, float(m[-1])))

if not rows:
    sys.exit('no finished runs yet')

nlls = [n for _, n in rows]
mean = statistics.mean(nlls)
sd = statistics.stdev(nlls) if len(nlls) > 1 else float('nan')
se = sd / math.sqrt(len(nlls))
pred_se = math.sqrt(VALID_VAR / N_BLOCKS)

print(f'{len(rows)} seeds')
for s, n in rows:
    print(f'  seed {s:>3}  nll {n:.6f}  ppl {math.exp(n):.3f}')
print()
print(f'mean nll      {mean:.6f}   ppl {math.exp(mean):.3f}')
print(f'seed-to-seed  sd {sd:.6f} nats  ({math.exp(mean)*sd:.3f} ppl)')
print(f'  predicted   sd {pred_se:.6f} nats from valid_var/N_blocks')
print(f'se of mean    {se:.6f} nats  ({math.exp(mean)*se:.3f} ppl)')
print()
print(f'exact uniform-order   nll {EXACT_UNIFORM_NLL:.6f}  ppl {math.exp(EXACT_UNIFORM_NLL):.3f}')
print(f'  mean ELBO - exact = {mean - EXACT_UNIFORM_NLL:+.6f} nats'
      f'  = {(mean - EXACT_UNIFORM_NLL)/se:+.2f} se of the mean')
for name, v in PRIOR_ELBO.items():
    print(f'  single draw {name:<28} {v:.6f}  '
          f'{(v - EXACT_UNIFORM_NLL)/sd:+.2f} sd above exact')
