"""How MAUVE's cost scales with sample count.

MAUVE's own guidance is ~5000 samples; the DUEL paper's Table 5 protocol uses 1000.
This script measures the two stages separately so the 5000-sample cost can be stated
rather than guessed:

  1. FEATURIZATION - GPT-2 Large forward per text. Expected linear in n.
     Timed at several n by tiling the real 1000-sample set (cost depends only on the
     number of texts and their token lengths, and tiling preserves the length
     distribution exactly). Tiled inputs are SYNTHETIC and are used for cost only -
     no MAUVE *score* from a tiled set is meaningful.

  2. CLUSTERING + DIVERGENCE - PCA, k-means, histogram, divergence curve. NOT linear:
     mauve/compute_mauve.py sets
         num_buckets = max(2, round(min(n_p, n_q) / 10))
     so the number of clusters grows WITH n. k-means is O(n_total * k * d * iters),
     and with k = n/10 that is O(n^2). Timed from precomputed features (no GPU work)
     by tiling a real feature matrix, which isolates this stage exactly.

Usage:
  python scripts/mauve_scaling.py --sample-file <csv> --out results/mauve_scaling.json
"""

import argparse
import ast
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def read_samples(path, limit=None):
    """sample_logs/*.txt are CSVs; sample text lives in column 5 as a repr'd list."""
    csv.field_size_limit(10 ** 9)
    out = []
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if len(row) >= 6:
                out.append(ast.literal_eval(row[5])[0])
    return out[:limit] if limit else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-file', required=True)
    ap.add_argument('--cache-dir', default='/share/kuleshov/ssahoo/textdiffusion/data')
    ap.add_argument('--featurize-ns', default='250,500,1000,2000')
    ap.add_argument('--cluster-ns', default='1000,2000,5000')
    ap.add_argument('--max-text-length', type=int, default=1024)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import datasets
    import mauve
    from mauve.utils import featurize_tokens_from_model, get_tokenizer, get_model

    res = {'meta': {
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'node': os.environ.get('SLURMD_NODENAME'),
        'job_id': os.environ.get('SLURM_JOB_ID'),
        'max_text_length': args.max_text_length,
        'note': 'tiled inputs are synthetic, used for COST only; scores meaningless',
    }}

    samples = read_samples(args.sample_file)
    print(f'[info] {len(samples)} real samples', flush=True)

    # ---- real reference text, same construction as diffusion.py:1336-1361 ----
    ds = datasets.load_dataset('openwebtext', split='train[-100000:]',
                               cache_dir=args.cache_dir, streaming=False,
                               trust_remote_code=True)
    idx = torch.randperm(len(ds))[:len(samples)].tolist()
    reference = [ex['text'] for ex in ds.select(idx)]

    tok = get_tokenizer('gpt2-large')
    model = get_model('gpt2-large', tok, 0)

    def tokenize(texts):
        return [torch.LongTensor(tok.encode(t, truncation=True,
                                            max_length=args.max_text_length)).unsqueeze(0)
                for t in texts]

    # ================= 1. FEATURIZATION SCALING =================
    feat_rows = []
    base_tokens = tokenize(samples)
    for n in [int(x) for x in args.featurize_ns.split(',')]:
        toks = [base_tokens[i % len(base_tokens)] for i in range(n)]
        _sync()
        t0 = time.perf_counter()
        feats = featurize_tokens_from_model(model, toks, 1, name=f'feat{n}')
        _sync()
        dt = time.perf_counter() - t0
        feat_rows.append({'n': n, 'seconds': dt, 'sec_per_text': dt / n,
                          'synthetic': n > len(base_tokens)})
        print(f'[featurize] n={n:5d}  {dt:8.2f} s  ({dt/n*1000:.1f} ms/text)', flush=True)
    res['featurization'] = feat_rows

    # real features for both sides, used as the tiling seed for stage 2
    p_feats = featurize_tokens_from_model(model, tokenize(reference), 1, name='ref').cpu().numpy()
    q_feats = featurize_tokens_from_model(model, base_tokens, 1, name='gen').cpu().numpy()
    del model
    torch.cuda.empty_cache()

    # ================= 2. CLUSTERING + DIVERGENCE SCALING =================
    clus_rows = []
    for n in [int(x) for x in args.cluster_ns.split(',')]:
        pf = np.concatenate([p_feats] * (n // len(p_feats) + 1))[:n]
        qf = np.concatenate([q_feats] * (n // len(q_feats) + 1))[:n]
        t0 = time.perf_counter()
        out = mauve.compute_mauve(p_features=pf, q_features=qf, verbose=False)
        dt = time.perf_counter() - t0
        clus_rows.append({'n': n, 'seconds': dt,
                          'num_buckets': int(out.num_buckets),
                          'synthetic': n > len(q_feats)})
        print(f'[cluster+divergence] n={n:5d}  {dt:8.2f} s  '
              f'(num_buckets={out.num_buckets})', flush=True)
    res['clustering'] = clus_rows

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'[info] wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
