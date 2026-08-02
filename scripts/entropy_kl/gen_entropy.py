"""Stage 1: exact per-token entropy H(P_F) of a deterministic unmasking rule F.

Generates n samples from BD3-LM L'=16 (OWT) under rule F at parallelism k, with
`sampling.nucleus_p=1.0` so that the accumulated path log-prob is exactly log P_F(x).
Records, per revealed position, both the Rao-Blackwellised entropy and the sampled
token's log-prob (see tracer.DuelTrace), plus the raw token ids so Stage 2 can score
the same samples under a *different* rule.

Protocol matches scripts/sampler_comparison/gen_ppl.sh EXCEPT nucleus_p, which is 1.0
here and 0.9 there. With nucleus_p=0.9 the samples come from a truncated q and the
estimator is the cross-entropy H(q_0.9, P_F), not H(P_F); these samples are therefore
NOT comparable with the existing CoLA / gen-ppl / MAUVE numbers.

The `uniform` (stochastic) sampler is deliberately unsupported: for a stochastic policy
the accumulated path log-prob is a lower bound on log p(x), not the exact likelihood.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataloader  # noqa: E402
import diffusion  # noqa: E402
from tracer import DuelTrace  # noqa: E402

SEQ_LEN = 1024
BLOCK_SIZE = 16
CKPT = f'kuleshov-group/bd3lm-owt-block_size{BLOCK_SIZE}'
DETERMINISTIC_RULES = ('block_greedy', 'block_left_to_right',
                       'block_probability_margin')


def build_config(strategy, k, batch_size, seed):
    for name, fn in [('cwd', os.getcwd),
                     ('device_count', torch.cuda.device_count),
                     ('eval', eval),
                     ('div_up', lambda x, y: (x + y - 1) // y)]:
        omegaconf.OmegaConf.register_new_resolver(name, fn, replace=True)
    overrides = [
        'model=small',
        'algo=bd3lm',
        'algo.T=5000',
        'algo.backbone=hf_dit',
        'data=openwebtext-split',
        f'model.length={SEQ_LEN}',
        f'block_size={BLOCK_SIZE}',
        'mode=sample_eval',
        f'eval.checkpoint_path={CKPT}',
        'model.attn_backend=sdpa',
        f'seed={seed}',
        'sampling.nucleus_p=1.0',
        'sampling.kv_cache=true',
        # `first_hitting` is read only by `_ddpm_caching_update` / `_semi_ar_sampler`
        # (diffusion.py:992,1699,1704); `_strategy_based_sampler` never touches it.
        # It is off here purely to lift the `eval_batch_size == 1` assertion at
        # diffusion.py:197-200 so generation can be batched. No effect on this path.
        'sampling.first_hitting=false',
        f'+sampling.strategy={strategy}',
        f'+sampling.strategy_k={k}',
        f'loader.eval_batch_size={batch_size}',
        '~wandb',
    ]
    with hydra.initialize_config_dir(
            config_dir=os.path.join(ROOT, 'configs'), version_base=None):
        return hydra.compose(config_name='config', overrides=overrides)


def assert_protocol(cfg, model):
    """Hard gates: the estimator is only H(P_F) if the sampler is untruncated."""
    assert cfg.sampling.nucleus_p == 1.0, cfg.sampling.nucleus_p
    # no temperature / top-k knobs exist in this config; fail loudly if one appears
    unexpected = [key for key in ('temperature', 'top_k', 'topk', 'top_p')
                  if key in cfg.sampling]
    assert not unexpected, f'unexpected truncation knobs in sampling cfg: {unexpected}'
    # _nucleus_sample early-returns the identical object at p == 1.0
    probe = torch.rand(2, SEQ_LEN, 8)
    assert model._nucleus_sample(probe) is probe, '_nucleus_sample is not identity'
    assert cfg.sampling.var_length is False, 'var_length would truncate sequences'
    print('[protocol] nucleus_p=1.0, no temperature/top-k, _nucleus_sample is identity,'
          ' var_length=False', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', required=True, choices=DETERMINISTIC_RULES)
    ap.add_argument('--k', type=int, required=True, choices=[1, 2, 4, 8])
    ap.add_argument('-n', '--n-samples', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=2)
    ap.add_argument('--gen-ppl', action='store_true',
                    help='also run the GPT-2-large generative perplexity metric')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = build_config(args.strategy, args.k, args.batch_size, args.seed)
    tokenizer = dataloader.get_tokenizer(cfg)
    model = diffusion.Diffusion(cfg, tokenizer=tokenizer).to('cuda')
    model.eval()
    if model.ema is not None:  # mirrors restore_model_and_sample
        model.ema.store(itertools.chain(model.backbone.parameters(),
                                        model.noise.parameters()))
        model.ema.copy_to(itertools.chain(model.backbone.parameters(),
                                          model.noise.parameters()))
    model.backbone.eval()
    model.noise.eval()
    assert model.generation_strategy is not None, 'strategy sampler not active'
    assert_protocol(cfg, model)

    torch.manual_seed(args.seed)
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    ids, ent, lp, step_in_block, block_of = [], [], [], [], []
    checks, nfes = None, None
    t0 = time.time()
    for b in range(n_batches):
        bs = min(args.batch_size, args.n_samples - b * args.batch_size)
        trace = DuelTrace(bs, SEQ_LEN, 'cuda')
        model._duel_trace = trace
        try:
            x, steps = model._strategy_based_sampler(
                n_samples=bs, num_strides=SEQ_LEN // BLOCK_SIZE, seqlen=SEQ_LEN)
        finally:
            model._duel_trace = None
        assert x is not None, 'sampler returned None (early stop)'
        assert x.shape == (bs, SEQ_LEN), x.shape
        nfes = steps
        checks = checks or trace.checks
        s = trace.summary()
        assert (s['n_revealed'] == SEQ_LEN - 1).all(), s['n_revealed']
        ids.append(x.cpu().numpy())
        ent.append(trace.ent.cpu().numpy())
        lp.append(trace.lp.cpu().numpy())
        step_in_block.append(trace.step_in_block.cpu().numpy())
        block_of.append(trace.block_of.cpu().numpy())
        el = time.time() - t0
        done = sum(a.shape[0] for a in ids)
        print(f'{done}/{args.n_samples}  {el:.0f}s  {el / done:.2f} s/seq  '
              f'nfe={steps}  H_rb/tok='
              f'{(s["H_rb_total"] / (SEQ_LEN - 1)).mean().item():.4f}  '
              f'H_path/tok='
              f'{(s["H_path_total"] / (SEQ_LEN - 1)).mean().item():.4f}', flush=True)

    ids = np.concatenate(ids)
    ent = np.concatenate(ent)
    lp = np.concatenate(lp)
    n_tok = SEQ_LEN - 1
    H_rb = np.nansum(ent[:, 1:], axis=1) / n_tok
    H_path = -np.nansum(lp[:, 1:], axis=1) / n_tok
    print(f'\ncell {args.strategy} k={args.k} n={len(ids)} NFE={nfes}')
    print(f'  H_rb   = {H_rb.mean():.5f} +/- {H_rb.std(ddof=1) / np.sqrt(len(H_rb)):.5f}'
          f'  nats/token')
    print(f'  H_path = {H_path.mean():.5f} +/- '
          f'{H_path.std(ddof=1) / np.sqrt(len(H_path)):.5f} nats/token')
    print(f'  variance ratio path/rb = '
          f'{H_path.var(ddof=1) / max(H_rb.var(ddof=1), 1e-30):.2f}x')
    print(f'  one-off checks: {checks}')

    text = tokenizer.batch_decode(torch.from_numpy(ids))
    gen_ppl = np.array(np.nan)
    if args.gen_ppl:
        model.metrics.record_generative_perplexity(
            text, SEQ_LEN, batch_size=min(8, len(text)), device='cuda')
        gen_ppl = np.array(model.metrics.gen_ppl.compute().item())
        print(f'  gen_ppl (GPT-2 large, retokenised) = {gen_ppl}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        strategy=np.array(args.strategy), k=np.array(args.k),
        seed=np.array(args.seed), nfe=np.array(nfes),
        input_ids=ids.astype(np.int32),
        ent_at_pos=ent.astype(np.float64),
        lp_at_pos=lp.astype(np.float64),
        step_in_block=np.concatenate(step_in_block).astype(np.int16),
        block_of=np.concatenate(block_of).astype(np.int16),
        H_rb_per_token=H_rb, H_path_per_token=H_path,
        gen_ppl=gen_ppl,
        checks=np.array(str(checks)),
        samples=np.array(text, dtype=object))
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
