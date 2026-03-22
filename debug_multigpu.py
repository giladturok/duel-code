"""Debug script to find where multi-GPU evaluation hangs."""
import os
import sys
import torch
import lightning as L
import hydra
import omegaconf

# Print env info immediately
rank = os.environ.get('SLURM_PROCID', '?')
local_rank = os.environ.get('SLURM_LOCALID', '?')
cvd = os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')
n_gpus = torch.cuda.device_count()
print(f'[RANK {rank}] SLURM_LOCALID={local_rank}, CUDA_VISIBLE_DEVICES={cvd}, device_count={n_gpus}', flush=True)

# Import project modules
print(f'[RANK {rank}] Importing project modules...', flush=True)
import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

# Use lm1b SEDD config (small/fast) for testing
@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    rank = os.environ.get('SLURM_PROCID', '?')

    print(f'[RANK {rank}] Config loaded. devices={config.trainer.devices}, mode={config.mode}', flush=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    # Step 1: Load model
    print(f'[RANK {rank}] Step 1: Loading model...', flush=True)
    if 'hf' in config.algo.backbone:
        model = diffusion.Diffusion(config, tokenizer=tokenizer).to('cuda')
    else:
        model = diffusion.Diffusion.load_from_checkpoint(
            config.eval.checkpoint_path,
            tokenizer=tokenizer,
            config=config,
            strict=False,
            weights_only=False).to('cuda')
    print(f'[RANK {rank}] Step 1 DONE: Model loaded on {next(model.parameters()).device}', flush=True)

    # Step 2: Create wandb logger
    print(f'[RANK {rank}] Step 2: Creating wandb logger...', flush=True)
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            **config.wandb)
    print(f'[RANK {rank}] Step 2 DONE: wandb_logger={type(wandb_logger).__name__}', flush=True)

    # Step 3: Create trainer
    print(f'[RANK {rank}] Step 3: Creating trainer...', flush=True)
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    print(f'[RANK {rank}] Step 3 DONE: Trainer created', flush=True)

    # Step 4: Create dataloaders
    print(f'[RANK {rank}] Step 4: Creating dataloaders...', flush=True)
    seed = config.seed
    L.seed_everything(seed)
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=seed)
    print(f'[RANK {rank}] Step 4 DONE: DataLoader created with {len(valid_ds.dataset)} samples', flush=True)

    # Step 5: Run validation
    print(f'[RANK {rank}] Step 5: Calling trainer.validate()...', flush=True)
    trainer.validate(model, valid_ds)
    print(f'[RANK {rank}] Step 5 DONE: Validation complete!', flush=True)


if __name__ == '__main__':
    main()
