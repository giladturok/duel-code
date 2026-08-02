"""Stage 2: score Stage-1 samples from rule A under rule B, exactly.

KL(P_A || P_B) = E_{x ~ P_A}[log P_A(x) - log P_B(x)]. Both terms come from the DUEL
exact-likelihood scorer (`diffusion.Diffusion._compute_exact_ll` ->
`exact_likelihood.compute_exact_loglikelihood_cached`), the same code
`scripts/sampler_comparison/duel_ppl.sh` runs on the OWT validation split, just pointed
at generated token ids. No reference model and no real data are involved.

Scoring convention. A generated sequence is [BOS] + 1023 sampled tokens. The event whose
likelihood we want is exactly those 1023 tokens *given* BOS. The stock scorer masks the
first `attention_mask.sum()` positions, which would score BOS and drop the last token, so
`valid_mask` / `lengths_override` (added to `_compute_exact_ll` and
`compute_exact_loglikelihood_cached`) are used to mask positions 1..1023 and keep
position 0 as context. `lengths` stays 1024 so that the strategy's block pointer
(`num_unmasked // block_size`) tracks the generation-time sampler step for step.

Correctness gate: with rule B == rule A this must reproduce Stage 1's accumulated path
log-prob for the same sequence (that is the "KL(P_A||P_A) ~ 0" check, in its strongest
form -- an independent re-derivation of the order and the log-probs).
"""
import argparse
import itertools
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


def build_config(strategy, k, batch_size):
    for name, fn in [('cwd', os.getcwd),
                     ('device_count', torch.cuda.device_count),
                     ('eval', eval),
                     ('div_up', lambda x, y: (x + y - 1) // y)]:
        omegaconf.OmegaConf.register_new_resolver(name, fn, replace=True)
    overrides = [
        'model=small',
        'algo=bd3lm',
        'algo.backbone=hf_dit',
        'data=openwebtext-split',
        'data.valid=openwebtext-valid-1k',
        'data.insert_valid_eos=False',
        f'model.length={SEQ_LEN}',
        f'block_size={BLOCK_SIZE}',
        f'eval.checkpoint_path={CKPT}',
        f'eval.exact_ll_strategy={strategy}',
        f'eval.exact_ll_k={k}',
        '+eval.exact_ll_use_kv_cache=true',
        'sampling.kv_cache=true',
        'mode=duel_ppl',
        'model.attn_backend=sdpa',
        f'loader.eval_batch_size={batch_size}',
        '~wandb',
    ]
    with hydra.initialize_config_dir(
            config_dir=os.path.join(ROOT, 'configs'), version_base=None):
        return hydra.compose(config_name='config', overrides=overrides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True, help='Stage-1 npz produced by rule A')
    ap.add_argument('--score-strategy', required=True, help='rule B')
    ap.add_argument('--score-k', type=int, required=True)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('-n', '--n-samples', type=int, default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    src = np.load(args.samples, allow_pickle=True)
    ids = src['input_ids'].astype(np.int64)
    if args.n_samples:
        ids = ids[:args.n_samples]
    assert ids.shape[1] == SEQ_LEN, ids.shape
    # Stage-1 path log-likelihood of the same sequences (the log P_A(x) term)
    ll_gen = -src['H_path_per_token'][:len(ids)] * (SEQ_LEN - 1)

    cfg = build_config(args.score_strategy, args.score_k, args.batch_size)
    tokenizer = dataloader.get_tokenizer(cfg)
    model = diffusion.Diffusion(cfg, tokenizer=tokenizer).to('cuda')
    model.eval()
    if model.ema is not None:
        model.ema.store(itertools.chain(model.backbone.parameters(),
                                        model.noise.parameters()))
        model.ema.copy_to(itertools.chain(model.backbone.parameters(),
                                          model.noise.parameters()))
    model.backbone.eval()
    model.noise.eval()

    assert (ids[:, 0] == tokenizer.bos_token_id).all(), 'position 0 is not BOS'

    n = len(ids)
    nll = np.full(n, np.nan)
    t0 = time.time()
    for b0 in range(0, n, args.batch_size):
        sel = slice(b0, min(b0 + args.batch_size, n))
        x0 = torch.from_numpy(ids[sel]).cuda()
        bs = x0.shape[0]
        attn = torch.ones(bs, SEQ_LEN, device='cuda')
        # mask + score positions 1..1023; position 0 (BOS) is given context
        valid = torch.ones(bs, SEQ_LEN, dtype=torch.bool, device='cuda')
        valid[:, 0] = False
        lengths = torch.full((bs,), SEQ_LEN, dtype=torch.long, device='cuda')
        with torch.no_grad():
            res = model._compute_exact_ll(x0=x0, attention_mask=attn,
                                          use_kv_cache=True, valid_mask=valid,
                                          lengths_override=lengths)
        assert (res['answer_lens'] == SEQ_LEN - 1).all()
        nll[sel] = res['nll'].float().cpu().numpy()
        el = time.time() - t0
        done = min(b0 + args.batch_size, n)
        print(f'{done}/{n}  {el:.0f}s  {el / done:.2f} s/seq  '
              f'nll/tok={np.nanmean(nll[:done]) / (SEQ_LEN - 1):.4f}', flush=True)

    ll_score = -nll
    d = (ll_gen - ll_score) / (SEQ_LEN - 1)   # per-token log-ratio
    print(f'\nsamples from {os.path.basename(args.samples)} scored under '
          f'{args.score_strategy} k={args.score_k}')
    print(f'  -log P_B / token = {(nll / (SEQ_LEN - 1)).mean():.5f}')
    print(f'  -log P_A / token = {(-ll_gen / (SEQ_LEN - 1)).mean():.5f}   (Stage 1 path)')
    print(f'  KL_hat/token     = {d.mean():.6f} +/- '
          f'{d.std(ddof=1) / np.sqrt(len(d)):.6f}')
    print(f'  max |per-seq diff| = {np.abs(d).max():.3e}  '
          '(should be ~0 only when rule B == rule A)')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out,
             samples_file=np.array(args.samples),
             gen_strategy=src['strategy'], gen_k=src['k'],
             score_strategy=np.array(args.score_strategy),
             score_k=np.array(args.score_k),
             ll_gen=ll_gen, ll_score=ll_score,
             kl_per_token=d)
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
