"""Read-only cost decomposition for the sampler-comparison evaluation metrics.

Measures, as separate line items, what it costs to EVALUATE a fixed set of
already-generated samples with:

  1. generative perplexity  (metrics.record_generative_perplexity, metrics.py:185)
       a. GPT-2 Large checkpoint load          (metrics.py:195-197)
       b. re-tokenization                      (metrics.py:135 _eval_retokenize)
       c. the sliding-window scoring loop      (metrics.py:222-247)
  2. the OpenWebText reference-corpus build    (diffusion.py:1336-1361)
  3. MAUVE                                     (metrics.record_mauve_score, metrics.py:270)

No samples are generated. Nothing in the repo is modified: components (a) and
(b) are timed by temporarily wrapping `transformers.AutoModelForCausalLM.
from_pretrained` and `Metrics._eval_retokenize` with timing shims, so the real
`record_generative_perplexity` body runs unaltered and the scoring loop is
recovered as (total - load - retokenize).

Every timed region is bracketed by torch.cuda.synchronize() and preceded by a
warmup where a warmup is meaningful. Peak GPU memory per component comes from
torch.cuda.reset_peak_memory_stats()/max_memory_allocated(); a separate
nvidia-smi poll (see the companion .sh) gives the whole-process peak.

Usage:
  python scripts/cost_decomposition.py \
      --sample-file sample_logs/samples_bd3lm_len1024_blocksize16_block-greedy_k1.txt \
      --label bd3lm_greedy_k1 \
      --out results/cost_bd3lm_greedy_k1.json
"""

import argparse
import ast
import contextlib
import csv
import json
import os
import sys
import time

# This file lives in <repo>/scripts/; make the repo root importable so that
# `import metrics` picks up <repo>/metrics.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import transformers


# --------------------------------------------------------------------------
# timing helpers
# --------------------------------------------------------------------------

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timings:
    """Collects {name: {seconds, peak_mem_bytes}}."""

    def __init__(self):
        self.data = {}

    @contextlib.contextmanager
    def region(self, name):
        _sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        yield
        _sync()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        self.data[name] = {'seconds': dt, 'peak_mem_bytes': peak}
        print(f'[timing] {name:38s} {dt:10.3f} s   peak {peak / 2**30:7.3f} GiB',
              flush=True)

    def record(self, name, seconds, peak_mem_bytes=0):
        self.data[name] = {'seconds': seconds, 'peak_mem_bytes': peak_mem_bytes}
        print(f'[timing] {name:38s} {seconds:10.3f} s   peak '
              f'{peak_mem_bytes / 2**30:7.3f} GiB', flush=True)


# --------------------------------------------------------------------------
# config: rebuild the production config for metrics.Metrics without loading
# any diffusion backbone. Mirrors scripts/sampler_comparison/gen_ppl.sh.
# --------------------------------------------------------------------------

def build_config(perplexity_batch_size):
    import hydra
    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'configs')
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name='config',
            overrides=[
                'model=small',
                'algo=bd3lm',
                'data=openwebtext-split',
                'model.length=1024',
                'block_size=16',
                'mode=sample_eval',
                'loader.eval_batch_size=1',
                'sampling.nucleus_p=0.9',
                f'eval.perplexity_batch_size={perplexity_batch_size}',
            ],
        )
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-file', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--n-samples', type=int, default=1000)
    ap.add_argument('--max-length', type=int, default=1024)
    ap.add_argument('--perplexity-batch-size', type=int, default=1,
                    help='Production protocol is 1 (loader.eval_batch_size=1 '
                         'in scripts/sampler_comparison/gen_ppl.sh).')
    ap.add_argument('--skip-mauve', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = 'cuda'
    T = Timings()
    meta = {
        'label': args.label,
        'sample_file': args.sample_file,
        'perplexity_batch_size': args.perplexity_batch_size,
        'max_length': args.max_length,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'pytorch_cuda_alloc_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        'torch': torch.__version__,
    }

    # ---- load samples ------------------------------------------------------
    # sample_logs/*.txt are CSVs written by utils.update_and_save_csv via
    # main.py:104-113, NOT plain text. Columns:
    #   0 gen_ppl | 1 gen_nfes | 2 gen_entropy | 3 gen_lengths
    #   4 mauve_score | 5 samples (repr of a 1-element list) | 6 seed
    # The sample text must be pulled out of column 5 and literal_eval'd;
    # reading the file line-by-line feeds the leading floats and the
    # `["<|endoftext|>...` wrapper to the scorer and corrupts the metric values.
    with T.region('00_read_sample_file'):
        csv.field_size_limit(10 ** 9)
        text_samples, nfes = [], []
        with open(args.sample_file, newline='') as f:
            for row in csv.reader(f):
                if len(row) < 6:
                    continue
                text_samples.append(ast.literal_eval(row[5])[0])
                nfes.append(float(row[1]))
    text_samples = text_samples[:args.n_samples]
    meta['nfe_per_sample'] = (sum(nfes) / len(nfes)) if nfes else None
    meta['n_samples'] = len(text_samples)
    print(f'[info] loaded {len(text_samples)} samples from {args.sample_file}',
          flush=True)

    import metrics as metrics_mod

    # ---- Metrics construction (loads the GPT-2 *tokenizer*, metrics.py:86) --
    cfg = build_config(args.perplexity_batch_size)
    with T.region('01_metrics_init_tokenizer_load'):
        M = metrics_mod.Metrics(cfg)
    M.to(device)

    # ======================================================================
    # 1. GENERATIVE PERPLEXITY
    # ======================================================================
    # Warm the HF cache + CUDA context so the *timed* load is a warm-cache
    # load, not a first-touch page-in of the 3.1 GB checkpoint.
    print('[info] warmup: GPT-2 Large load + one forward', flush=True)
    _warm = transformers.AutoModelForCausalLM.from_pretrained(
        M.gen_ppl_eval_model_name_or_path).eval().to(device)
    with torch.no_grad():
        _warm(torch.ones((1, 128), dtype=torch.long, device=device))
    _sync()
    del _warm
    torch.cuda.empty_cache()

    # Shim from_pretrained and _eval_retokenize so the *real*
    # record_generative_perplexity body (metrics.py:185) runs unmodified while
    # its two sub-costs are attributed separately.
    load_stats = {}
    retok_stats = {}

    orig_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
    orig_retokenize = metrics_mod.Metrics._eval_retokenize

    def timed_from_pretrained(*a, **kw):
        _sync()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = orig_from_pretrained(*a, **kw)
        _sync()
        load_stats['seconds'] = time.perf_counter() - t0
        load_stats['peak_mem_bytes'] = torch.cuda.max_memory_allocated()
        return out

    def timed_retokenize(self, *a, **kw):
        _sync()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = orig_retokenize(self, *a, **kw)
        _sync()
        retok_stats['seconds'] = time.perf_counter() - t0
        retok_stats['peak_mem_bytes'] = torch.cuda.max_memory_allocated()
        return out

    transformers.AutoModelForCausalLM.from_pretrained = timed_from_pretrained
    metrics_mod.Metrics._eval_retokenize = timed_retokenize
    try:
        with T.region('10_genppl_TOTAL'):
            M.record_generative_perplexity(
                text_samples,
                args.max_length,
                args.perplexity_batch_size,
                device=device,
            )
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = orig_from_pretrained
        metrics_mod.Metrics._eval_retokenize = orig_retokenize

    # NOTE: the .to(device) of the checkpoint happens at metrics.py:198,
    # *outside* from_pretrained, so it lands in the residual below rather than
    # in 11_. The residual is therefore (H2D copy + scoring loop).
    T.record('11_genppl_model_load_from_pretrained',
             load_stats.get('seconds', float('nan')),
             load_stats.get('peak_mem_bytes', 0))
    T.record('12_genppl_retokenize',
             retok_stats.get('seconds', float('nan')),
             retok_stats.get('peak_mem_bytes', 0))
    scoring = (T.data['10_genppl_TOTAL']['seconds']
               - load_stats.get('seconds', 0.0)
               - retok_stats.get('seconds', 0.0))
    T.record('13_genppl_scoring_loop_plus_h2d', scoring,
             T.data['10_genppl_TOTAL']['peak_mem_bytes'])

    meta['gen_ppl'] = float(M.gen_ppl.compute().item())
    print(f"[result] gen_ppl = {meta['gen_ppl']:.4f}", flush=True)

    # free the eval model before MAUVE so peak memory is attributable
    torch.cuda.empty_cache()

    if not args.skip_mauve:
        # ==================================================================
        # 2. OPENWEBTEXT REFERENCE-CORPUS BUILD  (diffusion.py:1336-1361)
        #    Reproduced verbatim. In production this sits INSIDE the timed
        #    region of restore_model_and_sample.
        # ==================================================================
        import datasets

        n_samples = len(text_samples)
        with T.region('20_owt_load_dataset'):
            reference_dataset = datasets.load_dataset(
                'openwebtext',
                split='train[-100000:]',
                cache_dir=cfg.data.cache_dir,
                streaming=False,
                trust_remote_code=True,
            )

        with T.region('21_owt_select_subset'):
            if len(reference_dataset) > n_samples:
                indices = torch.randperm(len(reference_dataset))[:n_samples].tolist()
                reference_subset = reference_dataset.select(indices)
            else:
                reference_subset = reference_dataset
            reference_text = [ex['text'] for ex in reference_subset]

        def truncate_to_tokens(text, max_tokens, tokenizer):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
            return tokenizer.decode(tokens)

        max_text_length = 1024
        with T.region('22_owt_truncate_encode_decode_loop'):
            reference_text = [
                truncate_to_tokens(t, max_text_length, M.tokenizer)
                for t in reference_text
            ]

        owt_total = sum(T.data[k]['seconds'] for k in
                        ('20_owt_load_dataset', '21_owt_select_subset',
                         '22_owt_truncate_encode_decode_loop'))
        T.record('23_owt_build_TOTAL', owt_total, 0)

        # ==================================================================
        # 3. MAUVE  (metrics.py:270)
        # ==================================================================
        with T.region('30_mauve_TOTAL'):
            score = M.record_mauve_score(
                generated_text=text_samples,
                reference_text=reference_text,
                max_text_length=max_text_length,
                device_id=0,
            )
        meta['mauve'] = float(score)
        print(f'[result] mauve = {score:.4f}', flush=True)

    out = {'meta': meta, 'timings': T.data}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[info] wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
