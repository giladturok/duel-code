"""Per-passage DUEL exact log-likelihood of arbitrary text.

The repo only ever computes DUEL likelihood over a prepared HF dataset, aggregated
into a single `val/exact_ppl` (`scripts/sampler_comparison/duel_ppl.sh` ->
`main.py:_ppl_eval` -> `diffusion.Diffusion._validation_step_exact_likelihood`).
E-B needs (a) arbitrary text as input and (b) a *per-passage* number, so this script
drives `diffusion.Diffusion._compute_exact_ll` directly with the identical config.

Protocol (matches `scripts/sampler_comparison/duel_ppl.sh` exactly):
  BD3-LM L'=16 OWT (`kuleshov-group/bd3lm-owt-block_size16`), model=small,
  algo=bd3lm, backbone=hf_dit, attn_backend=sdpa, model.length=1024, block_size=16,
  strategy=block_greedy, k=1, KV cache on. Log-probs are fp32 (`F.log_softmax`
  is in torch's fp32 autocast policy), model runs under bf16 autocast to match
  Lightning's `trainer.precision='bf16'`.

Sequence layout mirrors the OWT valid pipeline (`dataloader._group_texts` with
`insert_valid_eos=False`, `insert_valid_special=True`): [BOS] + <=1022 content
tokens + [EOS], right-padded to 1024. `attention_mask` is 1 over [BOS..EOS].
`_compute_exact_ll` then zeroes position 0 (algo.ignore_bos=True), so
`answer_lens = 1 + n_content` and the *last* attended position (the EOS) is the one
dropped, exactly as in the OWT run.

Modes:
  --source anchors : scores passages from scripts/cola_quality/out/anchors.jsonl
  --source owt     : re-runs the prepared openwebtext-valid-1k split, as a gate
                     against the published val/exact_ppl = 22.0434
                     (logs/bd3lm_openwebtext-split_block_size16_exact_ll_block_greedy_1.log)
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('TRITON_NUM_STAGES', '1')

import numpy as np
import torch
import hydra
import omegaconf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import dataloader  # noqa: E402
import diffusion  # noqa: E402

SEQ_LEN = 1024
BLOCK_SIZE = 16
CKPT = f'kuleshov-group/bd3lm-owt-block_size{BLOCK_SIZE}'

BASE_OVERRIDES = [
    'model=small',
    'algo=bd3lm',
    'algo.backbone=hf_dit',
    'data=openwebtext-split',
    'data.valid=openwebtext-valid-1k',
    'data.insert_valid_eos=False',
    f'model.length={SEQ_LEN}',
    f'block_size={BLOCK_SIZE}',
    f'eval.checkpoint_path={CKPT}',
    'eval.exact_ll_strategy=block_greedy',
    'eval.exact_ll_k=1',
    '+eval.exact_ll_use_kv_cache=true',
    'sampling.kv_cache=true',
    'mode=duel_ppl',
    'model.attn_backend=sdpa',
]


def build_config(eval_batch_size):
    for name, fn in [('cwd', os.getcwd),
                     ('device_count', torch.cuda.device_count),
                     ('eval', eval),
                     ('div_up', lambda x, y: (x + y - 1) // y)]:
        omegaconf.OmegaConf.register_new_resolver(name, fn, replace=True)
    with hydra.initialize_config_dir(
            config_dir=os.path.join(ROOT, 'configs'), version_base=None):
        cfg = hydra.compose(
            config_name='config',
            overrides=BASE_OVERRIDES + [f'loader.eval_batch_size={eval_batch_size}'])
    return cfg


# --------------------------------------------------------------------------- data

def load_anchor_rows(path, cells, n_per_cell):
    """Return rows for the first `n_per_cell` document indices of each cell.

    The four anchor kinds share a document index (`anchor-owt-repeat` at index i is
    built from `anchor-owt-real` at index i), so taking the same indices keeps the
    cells paired.
    """
    by_cell = {c: {} for c in cells}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r['cell'] in by_cell:
                by_cell[r['cell']][r['index']] = r['text']
    rows = []
    for c in cells:
        idxs = sorted(by_cell[c])[:n_per_cell]
        for i in idxs:
            rows.append({'cell': c, 'index': i, 'text': by_cell[c][i]})
    return rows


def encode(rows, tokenizer):
    """[BOS] + <=1022 content tokens + [EOS], right-padded with EOS to 1024."""
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    ids = np.full((len(rows), SEQ_LEN), eos, dtype=np.int64)
    attn = np.zeros((len(rows), SEQ_LEN), dtype=np.int64)
    for r, row in enumerate(rows):
        t = tokenizer(row['text'], add_special_tokens=False)['input_ids'][:SEQ_LEN - 2]
        seq = [bos] + t + [eos]
        ids[r, :len(seq)] = seq
        attn[r, :len(seq)] = 1
        row['n_content'] = len(t)
    return ids, attn


def load_owt_rows(cfg, tokenizer, limit):
    _, valid_loader = dataloader.get_dataloaders(
        cfg, tokenizer, skip_train=True, valid_seed=None)
    ds = valid_loader.dataset
    n = len(ds) if limit is None else min(limit, len(ds))
    ids = np.stack([np.asarray(ds[i]['input_ids']) for i in range(n)])
    attn = np.stack([np.asarray(ds[i]['attention_mask']) for i in range(n)]).astype(np.int64)
    rows = [{'cell': 'owt-valid-1k', 'index': i, 'n_content': int(attn[i].sum()) - 2}
            for i in range(n)]
    return rows, ids, attn


# --------------------------------------------------------------------------- run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['anchors', 'owt'], default='anchors')
    ap.add_argument('--anchors', default=os.path.join(
        ROOT, 'scripts/cola_quality/out/anchors.jsonl'))
    ap.add_argument('--cells', default=('anchor-owt-real,anchor-owt-repeat,'
                                        'anchor-owt-word-shuffled,anchor-owt-token-shuffled'))
    ap.add_argument('--n-per-cell', type=int, default=256)
    ap.add_argument('--limit', type=int, default=None, help='owt mode: #sequences')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16',
                    help="bf16 mirrors Lightning trainer.precision='bf16'")
    ap.add_argument('--no-sort', action='store_true',
                    help='do not length-sort batches (for batching-invariance checks)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = build_config(args.batch_size)
    tokenizer = dataloader.get_tokenizer(cfg)

    if args.source == 'anchors':
        cells = args.cells.split(',')
        rows = load_anchor_rows(args.anchors, cells, args.n_per_cell)
        ids, attn = encode(rows, tokenizer)
    else:
        rows, ids, attn = load_owt_rows(cfg, tokenizer, args.limit)

    print(f'{len(rows)} sequences; mean content tokens '
          f'{np.mean([r["n_content"] for r in rows]):.1f}', flush=True)

    model = diffusion.Diffusion(cfg, tokenizer=tokenizer)
    model = model.to('cuda')
    model.eval()
    # Mirror on_validation_epoch_start: swap in EMA weights. For an HF checkpoint the
    # EMA shadow is initialised from the loaded weights, so this is a no-op, but the
    # reference run does it and it costs nothing.
    if model.ema is not None:
        import itertools
        model.ema.store(itertools.chain(model.backbone.parameters(),
                                        model.noise.parameters()))
        model.ema.copy_to(itertools.chain(model.backbone.parameters(),
                                          model.noise.parameters()))
    model.backbone.eval()
    model.noise.eval()

    order = np.arange(len(rows))
    if not args.no_sort:
        # cost per batch ~ max content length in the batch (blocks with no valid
        # token anywhere in the batch are skipped), so length-sort the batches.
        order = np.argsort([-r['n_content'] for r in rows], kind='stable')

    nll_pt = np.full(len(rows), np.nan)
    ans_len = np.zeros(len(rows), dtype=np.int64)
    steps_per_batch = []
    t0 = time.time()
    for b0 in range(0, len(order), args.batch_size):
        sel = order[b0:b0 + args.batch_size]
        x0 = torch.from_numpy(ids[sel]).cuda()
        am = torch.from_numpy(attn[sel]).cuda().float()
        ctx = (torch.autocast('cuda', dtype=torch.bfloat16)
               if args.precision == 'bf16' else torch.autocast('cuda', enabled=False))
        with torch.no_grad(), ctx:
            res = model._compute_exact_ll(x0=x0, attention_mask=am, use_kv_cache=True)
        nll_pt[sel] = res['nll_per_token'].float().cpu().numpy()
        ans_len[sel] = res['answer_lens'].long().cpu().numpy()
        steps_per_batch.append(int(res['steps']) if res['steps'] is not None else -1)
        done = b0 + len(sel)
        el = time.time() - t0
        print(f'{done}/{len(order)}  {el:.0f}s  {el / done:.2f} s/seq  '
              f'running_ppl={np.exp(np.nansum(nll_pt[order[:done]] * ans_len[order[:done]]) / ans_len[order[:done]].sum()):.4f}',
              flush=True)

    micro = float(np.exp((nll_pt * ans_len).sum() / ans_len.sum()))
    print(f'\nmicro-averaged exact_ppl over all {len(rows)} sequences: {micro:.4f}',
          flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out,
             cell=np.array([r['cell'] for r in rows]),
             index=np.array([r['index'] for r in rows]),
             n_content=np.array([r['n_content'] for r in rows]),
             nll_per_token=nll_pt,
             answer_lens=ans_len,
             duel_ppl=np.exp(nll_pt),
             steps_per_batch=np.array(steps_per_batch),
             micro_exact_ppl=np.array(micro),
             precision=np.array(args.precision))
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
