# [DUEL: Exact Likelihood for Masked Diffusion via Deterministic Unmasking](https://arxiv.org/abs/2603.01367)

By [Gilad Turok](https://giladturok.github.io), [Chris De Sa](https://www.cs.cornell.edu/~cdesa/), [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/)

[![Paper](https://img.shields.io/badge/Paper-PDF-green)](https://arxiv.org/pdf/2603.01367)
[![arXiv](https://img.shields.io/badge/arXiv-2603.01367-b31b1b)](https://arxiv.org/abs/2603.01367)
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://giladturok.github.io/duel/)

<p align="center">
  <img src="mdm_generation_step.png" width="700">
</p>

We introduce **DUEL** ⚔️ (**D**eterministic **U**nmasking **E**xact **L**ikelihood), a framework for computing exact log-likelihoods under the test-time distribution of masked diffusion models (MDMs). By pairing a pretrained denoiser with a deterministic unmasking policy, DUEL collapses the super-exponentially large sum over unmasking orders to a single term, enabling exact (not bound) likelihood evaluation.

Key findings:
- DUEL narrows the MDM-to-autoregressive perplexity gap by up to 32% on in-domain data and 82% on zero-shot benchmarks
- Probability margin is the best-performing unmasking strategy under fixed compute budgets
- MDMs can even surpass autoregressive models with oracle unmasking orderings

This codebase builds on the [BD3-LM](https://github.com/kuleshov-group/bd3lms) framework.

## Code Organization

| File | Description |
|------|-------------|
| `main.py` | Entry point for all experiments |
| `diffusion.py` | Forward/reverse diffusion, training, sampling |
| `exact_likelihood.py` | DUEL exact log-likelihood computation |
| `selection_strategies.py` | Deterministic unmasking policies (greedy, L2R, prob. margin, conf. threshold) |
| `dataloader.py` | Data loading utilities |
| `metrics.py` | Evaluation metrics (PPL, BPD, gen. PPL, MAUVE) |
| `noise_schedule.py` | Noise schedules |
| `models/` | Network architectures (DiT, AR transformer) |
| `configs/` | Hydra configuration files |
| `scripts/` | Shell scripts for all experiments |

### Scripts Directory

| Directory | Experiment |
|-----------|------------|
| `scripts/duel_ppl/` | In-domain exact likelihood (DUEL) |
| `scripts/elbo_ppl/` | In-domain ELBO likelihood |
| `scripts/duel_zs_ppl/` | Zero-shot exact likelihood (DUEL) |
| `scripts/elbo_zs_ppl/` | Zero-shot ELBO likelihood |
| `scripts/sampler_duel_ppl/` | Sampler comparison via exact PPL |
| `scripts/sampler_gen_ppl/` | Sampler comparison via generative PPL |
| `scripts/gen_ppl/` | Generative perplexity evaluation |

## Getting Started

### Installation

```bash
conda env create -f environment.yml
conda activate duel
pip install -r requirements.txt
```

BD3-LMs and AR models don't require FlashAttention, but MDLM and SEDD baselines do. To install:
```bash
pip install flash-attn==2.5.6 --no-build-isolation
```
This requires CUDA toolkit and a compatible GPU (Ampere or newer). If installation fails, see the [flash-attn repo](https://github.com/Dao-AILab/flash-attention) for troubleshooting.

Create output directories:

```bash
mkdir -p outputs watch_folder logs sample_logs
```
- `outputs/` — Hydra run directories (configs, checkpoints)
- `watch_folder/` — SLURM stdout/stderr logs
- `logs/` — script output logs
- `sample_logs/` — generated text samples

### Configuration

**Data paths:** Dataset cache directories are configured in `configs/data/*.yaml` via the `cache_dir` field. Update these to point to your local data directory.

**Checkpoint paths:** Some scripts reference local checkpoint paths (e.g., `/share/kuleshov/...`). For OWT models, HuggingFace model IDs are used and checkpoints download automatically. For LM1B models, update the `eval.checkpoint_path` in the relevant scripts to point to your local checkpoint files.

## Checkpoints

### OpenWebText (auto-downloaded from HuggingFace)

| Model | HuggingFace ID |
|-------|---------------|
| BD3-LM (L'=4) | `kuleshov-group/bd3lm-owt-block_size4` |
| BD3-LM (L'=8) | `kuleshov-group/bd3lm-owt-block_size8` |
| BD3-LM (L'=16) | `kuleshov-group/bd3lm-owt-block_size16` |
| MDLM | `kuleshov-group/mdlm-owt` |
| AR | [Google Drive](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing) |
| SEDD | [Google Drive](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing) |

### LM1B (download manually)

LM1B checkpoints for BD3-LM, MDLM, SEDD, and AR are available via Google Drive. Download and update the checkpoint paths in the relevant scripts under `scripts/duel_ppl/` and `scripts/elbo_ppl/`.

## Reproducing Experiments

The main entry point is `main.py` with three evaluation modes:
- `mode=duel_ppl` — DUEL exact log-likelihood
- `mode=elbo_ppl` — ELBO-based perplexity
- `mode=sample_eval` — Generative sampling + metrics

To run **all experiments** across all tables (including all block sizes and sampler configurations):
```bash
bash scripts/run_all.sh
```

Individual scripts are designed for SLURM but can be run directly with `python`. For example, to run a script without SLURM:

```bash
# Instead of: sbatch scripts/elbo_ppl/ppl_owt_bd3lm.sh
# Run directly:
BLOCK_SIZE=4 python -u main.py \
    loader.eval_batch_size=16 \
    model=small \
    algo=bd3lm \
    algo.backbone=hf_dit \
    data=openwebtext-split \
    data.insert_valid_special=False \
    model.length=1024 \
    model.attn_backend=flex \
    block_size=${BLOCK_SIZE} \
    eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
    wandb.project=duel \
    +wandb.name=elbo-owt-bd3lm \
    mode=elbo_ppl
```

**Adjusting batch size:** Change `loader.eval_batch_size` to fit your GPU memory (e.g., `loader.eval_batch_size=8` for smaller GPUs).

**Weights & Biases logging:** All scripts log to W&B project `duel` by default. To disable W&B, replace the `wandb.project=...` and `wandb.name=...` lines with `wandb=null`.

### Table 1: In-Domain Perplexity (OWT)

**ELBO perplexity:**
```bash
# BD3-LM (set BLOCK_SIZE=4, 8, or 16 inside the script)
bash scripts/elbo_ppl/ppl_owt_bd3lm.sh

# MDLM, SEDD, AR
bash scripts/elbo_ppl/ppl_owt_mdlm.sh
bash scripts/elbo_ppl/ppl_owt_sedd.sh
bash scripts/elbo_ppl/ppl_owt_ar.sh
```

**DUEL exact perplexity:**
```bash
# BD3-LM (set BLOCK_SIZE=4, 8, or 16 inside the script)
bash scripts/duel_ppl/bd3lm_owt.sh

# MDLM, SEDD
bash scripts/duel_ppl/mdlm_owt.sh
bash scripts/duel_ppl/sedd_owt.sh
```

### Table 2: In-Domain Perplexity (LM1B)

Same structure as OWT. Scripts are in `scripts/elbo_ppl/ppl_lm1b_*.sh` and `scripts/duel_ppl/*_lm1b.sh`. **Note:** LM1B scripts require local checkpoint paths — update `eval.checkpoint_path` before running.

### Table 3: Zero-Shot Perplexity

**ELBO:**
```bash
bash scripts/elbo_zs_ppl/ppl_zs_owt_bd3lm.sh  # BD3-LM
bash scripts/elbo_zs_ppl/ppl_zs_owt_mdlm.sh   # MDLM
bash scripts/elbo_zs_ppl/ppl_zs_owt_sedd.sh   # SEDD
bash scripts/elbo_zs_ppl/ppl_zs_owt_ar.sh     # AR
```

**DUEL exact:**
```bash
bash scripts/duel_zs_ppl/owt_bd3lm_block_greedy.sh  # BD3-LM
bash scripts/duel_zs_ppl/owt_mdlm_block_greedy.sh   # MDLM
bash scripts/duel_zs_ppl/owt_sedd_block_greedy.sh   # SEDD
```

These evaluate on: AG News, LAMBADA, PTB, WikiText-2, WikiText-103, PubMed, ArXiv, LM1B.

### Table 4: Sampler Comparison (BD3-LM L'=16, OWT)

**Exact perplexity under different unmasking strategies:**
```bash
bash scripts/sampler_duel_ppl/block_greedy.sh              # Greedy confidence
bash scripts/sampler_duel_ppl/block_left_to_right.sh       # Left-to-right
bash scripts/sampler_duel_ppl/block_probability_margin.sh  # Probability margin
bash scripts/sampler_duel_ppl/block_conf_thresh.sh         # Confidence threshold
bash scripts/sampler_duel_ppl/elbo.sh                      # ELBO baseline
```

**Generative perplexity under different unmasking strategies:**
```bash
bash scripts/sampler_gen_ppl/block_greedy.sh              # Greedy confidence
bash scripts/sampler_gen_ppl/block_left_to_right.sh       # Left-to-right
bash scripts/sampler_gen_ppl/block_probability_margin.sh  # Probability margin
bash scripts/sampler_gen_ppl/block_confidence_threshold.sh # Confidence threshold
bash scripts/sampler_gen_ppl/uniform.sh                   # Uniform baseline
```

### Acknowledgements

This repository was built off of [BD3-LMs](https://github.com/kuleshov-group/bd3lms), [MDLM](https://github.com/kuleshov-group/mdlm), and [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).

## Citation

```bibtex
@article{turok2026duel,
  title={DUEL: Exact Likelihood for Masked Diffusion via Deterministic Unmasking},
  author={Turok, Gilad and De Sa, Chris and Kuleshov, Volodymyr},
  journal={arXiv preprint arXiv:2603.01367},
  year={2026}
}
```
