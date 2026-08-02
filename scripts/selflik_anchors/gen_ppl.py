"""Per-passage GPT-2 Large generative perplexity of the anchor passages.

Uses the repo's own `metrics.Metrics.record_generative_perplexity`
(`metrics.py:186-255`) verbatim with `batch_size=1`, so each appended entry of
`metrics.gen_ppls` is one passage. That keeps these numbers on exactly the same
protocol as every other gen-ppl number in the repo: retokenize with the eval
model's tokenizer, truncate/pad to `max_length=1024`, sliding window with
`eval_context_size=1024` and `stride=512` (a single window at this length),
EOS-valued positions excluded from the token count.

Eval model is `config.eval.gen_ppl_eval_model_name_or_path` = gpt2-large
(configs/config.yaml).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from duel_ll import build_config, load_anchor_rows  # noqa: E402
import metrics as metrics_mod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchors', default=os.path.join(
        ROOT, 'scripts/cola_quality/out/anchors.jsonl'))
    ap.add_argument('--cells', default=('anchor-owt-real,anchor-owt-repeat,'
                                        'anchor-owt-word-shuffled,anchor-owt-token-shuffled'))
    ap.add_argument('--n-per-cell', type=int, default=256)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = build_config(eval_batch_size=1)
    rows = load_anchor_rows(args.anchors, args.cells.split(','), args.n_per_cell)
    texts = [r['text'] for r in rows]
    print(f'{len(texts)} passages', flush=True)

    m = metrics_mod.Metrics(cfg)
    m.to('cuda')
    t0 = time.time()
    m.record_generative_perplexity(
        texts, max_length=cfg.model.length, batch_size=1, device='cuda')
    print(f'done in {time.time() - t0:.0f}s', flush=True)

    gen_ppls = np.array(m.gen_ppls, dtype=np.float64)
    gen_lengths = np.array(m.gen_lengths, dtype=np.float64)
    assert len(gen_ppls) == len(rows), (len(gen_ppls), len(rows))

    corpus = float(m.gen_ppl.compute().item())
    print(f'corpus (token-weighted) gen_ppl: {corpus:.4f}', flush=True)
    for c in args.cells.split(','):
        sel = np.array([r['cell'] == c for r in rows])
        print(f'  {c:28s} mean gen_ppl {gen_ppls[sel].mean():9.3f}  '
              f'mean scored tokens {gen_lengths[sel].mean():7.1f}', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out,
             cell=np.array([r['cell'] for r in rows]),
             index=np.array([r['index'] for r in rows]),
             gen_ppl=gen_ppls,
             gen_lengths=gen_lengths,
             gen_entropy=np.array(m.gen_entropies, dtype=np.float64),
             corpus_gen_ppl=np.array(corpus))
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
