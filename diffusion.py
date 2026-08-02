import itertools
import logging
from collections import OrderedDict
from dataclasses import dataclass

import datasets
import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

import dataloader
import metrics
import models
import noise_schedule
import utils
from exact_likelihood import (
  compute_exact_loglikelihood,
  compute_exact_loglikelihood_cached,
  compute_exact_loglikelihood_cached_permutations,
  compute_exact_loglikelihood_cached_subset_dp,
)
from selection_strategies import (
    BlockLeftToRightStrategy,
    BlockProbabilityMarginStrategy,
    BlockGreedyConfidenceStrategy,
    BlockConfidenceThresholdStrategy,
    BlockPermutationStrategy,
    BlockSubsetDPStrategy,
    ConfidenceThresholdStrategy,
    GreedyConfidenceStrategy,
    LeftToRightStrategy,
    ProbabilityMarginStrategy,
)

def _sample_categorical(categorical_probs):
  gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
  samples = (categorical_probs / gumbel_norm).argmax(dim=-1)
  return samples

def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.algo.sampler
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.cross_attn = self.config.algo.cross_attn
    self.ignore_bos = self.config.algo.ignore_bos
    self.mdlm_loss_scale = self.config.algo.mdlm_loss_scale
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    if hasattr(self.config, 'algo'):
      self.parameterization = self.config.algo.parameterization
    else:
      self.parameterization = self.config.parameterization
    if hasattr(self.config, 'block_size'):
      self.block_size = self.config.block_size
    else:
      self.block_size = self.config.model.length
    if self.parameterization == 'ar':
      self.block_size = 1
    if self.config.algo.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.algo.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
      #  egenerate mask if pretrained model uses flex attention mask
      # and current model uses sdpa mask
      if getattr(self.backbone.config, 'attn_backend', None) == 'flex' and \
        self.config.model.attn_backend == 'sdpa':
        self.backbone.config.attn_backend = 'sdpa'
        for i in self.backbone.backbone.blocks:
          i.attn_backend = 'sdpa'
        self.backbone.backbone.gen_mask(self.config.model.length, self.block_size, attn_backend='sdpa')
    else:
      raise ValueError(f'Unknown backbone: {self.config.algo.backbone}')

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length

    self.noise = noise_schedule.get_noise(self.config)
    self.metrics = metrics.Metrics(config)

    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.var_min = self.config.algo.var_min
    if self.var_min:
      self.register_buffer('sampling_eps_min', torch.tensor(
        self.config.training.sampling_eps_min))
      self.register_buffer('sampling_eps_max', torch.tensor(
        self.config.training.sampling_eps_max))
      
    self.exact_eval_enabled = (self.config.mode == 'duel_ppl')
    self.exact_ll_use_kv_cache = False
    if self.exact_eval_enabled:
      self.exact_ll_strategy_name = getattr(
        self.config.eval, 'exact_ll_strategy', 'block_greedy'
      )
      self.exact_ll_k = getattr(self.config.eval, 'exact_ll_k', 1)
      self._init_exact_ll_strategy()
      self.exact_ll_use_kv_cache = getattr(
        self.config.eval, 'exact_ll_use_kv_cache', False)
      
    self.generation_strategy = None
    if hasattr(self.config, 'sampling') and hasattr(self.config.sampling, 'strategy'):
        self._init_generation_strategy()
    
    self.time_conditioning = self.config.algo.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()
    # In Diffusion.__init__, after exact_eval initialization:


  def _init_generation_strategy(self):
    """Initialize generation strategy from config."""
    strategy_name = getattr(self.config.sampling, 'strategy', 'uniform')
    k = getattr(self.config.sampling, 'strategy_k', 1)
    
    strategy_map = {
        'greedy': GreedyConfidenceStrategy,
        'block_greedy': BlockGreedyConfidenceStrategy,
        'left_to_right': LeftToRightStrategy,
        'block_left_to_right': BlockLeftToRightStrategy,
        'probability_margin': ProbabilityMarginStrategy,
        'block_probability_margin': BlockProbabilityMarginStrategy,
        'confidence_threshold': ConfidenceThresholdStrategy,
        'block_confidence_threshold': BlockConfidenceThresholdStrategy,
    }
    
    if strategy_name == 'uniform':
        self.generation_strategy = None  # Use original _semi_ar_sampler
        return
        
    strategy_cls = strategy_map.get(strategy_name)
    if strategy_cls is None:
        raise ValueError(f"Unknown generation strategy: {strategy_name}")
    
    # Block strategies need block_size
    if 'block' in strategy_name:
        self.generation_strategy = strategy_cls(block_size=self.block_size, k=k)
    else:
        self.generation_strategy = strategy_cls(k=k)
    
  def _get_parameters(self):
    parameters = [self.backbone.parameters(),
                  self.noise.parameters()]
    return itertools.chain(* parameters)

  def on_validation_model_zero_grad(self) -> None:
    '''
    Small hack to avoid first validation on resume. 
    This will NOT work if the gradient accumulation step should be performed at this point.
    '''
    super().on_validation_model_zero_grad()
    if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
        self.trainer.sanity_checking = True
        self._restarting_skip_val_flag = False

  def _validate_configuration(self):
    if self.config.mode == 'sample_eval' and \
        self.config.sampling.first_hitting:
      assert self.config.loader.eval_batch_size == 1
    assert self.config.algo.backbone in {
      'dit', 'ar', 'hf_dit'}
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
    if self.config.sampling.kv_cache:
      assert self.config.algo.name in {'ar', 'bd3lm'}
      
    if self.parameterization in {'sedd'}:
      assert self.time_conditioning
    
    if self.config.mode == 'sample_eval':
      assert self.config.model.attn_backend != 'flex', 'FlexAttention mask not supported at inference.'
    if self.config.model.attn_backend == 'flex':
      assert self.config.algo.name == 'bd3lm', 'Custom FlexAttention mask only supported for BD3LM.'
      
  def _init_exact_ll_strategy(self):
    """Initialize exact LL decoding strategy."""
    if self.exact_ll_strategy_name == 'greedy':
        self.exact_ll_strategy = GreedyConfidenceStrategy(k=self.exact_ll_k)
    elif self.exact_ll_strategy_name == 'block_greedy':
        self.exact_ll_strategy = BlockGreedyConfidenceStrategy(
            block_size=self.block_size,
            k=self.exact_ll_k
        )
    elif self.exact_ll_strategy_name == 'left_to_right':
        self.exact_ll_strategy = LeftToRightStrategy(k=self.exact_ll_k)
    elif self.exact_ll_strategy_name == 'block_left_to_right':
        self.exact_ll_strategy = BlockLeftToRightStrategy(
            block_size=self.block_size,
            k=self.exact_ll_k
        )
    elif self.exact_ll_strategy_name == 'probability_margin':
        self.exact_ll_strategy = ProbabilityMarginStrategy(k=self.exact_ll_k)
    elif self.exact_ll_strategy_name == 'block_probability_margin':
        self.exact_ll_strategy = BlockProbabilityMarginStrategy(
            block_size=self.block_size,
            k=self.exact_ll_k
        )
    elif self.exact_ll_strategy_name == 'block_confidence_threshold':
        self.exact_ll_strategy = BlockConfidenceThresholdStrategy(
            block_size=self.block_size,
            k=self.exact_ll_k,
        )
    elif self.exact_ll_strategy_name == 'block_permutation':
        self.exact_ll_strategy = BlockPermutationStrategy(
            block_size=self.block_size,
            k=self.exact_ll_k,
        )
    elif self.exact_ll_strategy_name == 'block_subset_dp':
        self.exact_ll_strategy = BlockSubsetDPStrategy(
            block_size=self.block_size,
            k=self.exact_ll_k
        )
    elif self.exact_ll_strategy_name == 'confidence_threshold':
        self.exact_ll_strategy = ConfidenceThresholdStrategy(
            k=self.exact_ll_k,
        )
    else:
        raise ValueError(f"Unknown strategy: {self.exact_ll_strategy_name}")
      
  def _compute_sigma_for_exact_ll(self, x_t: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
    num_masked = (x_t == self.mask_index).sum(dim=-1, keepdim=True).float()
    
    if attention_mask is not None:
        seq_len = attention_mask.sum(dim=-1, keepdim=True).float()
    else:
        seq_len = torch.full((x_t.shape[0], 1), x_t.shape[-1], 
                             device=x_t.device, dtype=torch.float)
    
    mask_frac = num_masked / seq_len
    
    # Clamp to avoid numerical issues
    eps = 1e-1
    mask_frac = mask_frac.clamp(min=eps, max=1.0 - eps)
    
    # CORRECT: σ = -log(1 - mask_frac), NOT -log(1 - (1-ε)*mask_frac)
    sigma = -torch.log1p(-mask_frac)
    
    sigma_min = self.noise.sigma_min.to(sigma.device)
    sigma_max = self.noise.sigma_max.to(sigma.device)
    
    sigma = torch.clamp(sigma, min=sigma_min, max=sigma_max)
    return sigma

  def _compute_exact_ll(self, x0, attention_mask, use_kv_cache=None,
                        valid_mask=None, lengths_override=None):
    """
    Compute exact log-likelihood using deterministic decoding.
    
    Handles both SUBS (MDLM/BD3-LM) and SEDD parameterizations correctly:
    - SUBS: sigma=0 is fine (not used in output parameterization)
    - SEDD: sigma must reflect current mask fraction (RADD Theorem 1)
    """
    
    if self.ignore_bos:
        attention_mask[:, 0] = 0

    # `valid_mask` / `lengths_override` let a caller score an explicit set of
    # positions instead of the "first `attention_mask.sum()` positions" prefix
    # convention. Needed to score *generated* sequences, where position 0 is a
    # given BOS context token and positions 1..L-1 are the scored event; the
    # prefix convention would instead score BOS and drop the last token.
    # Both default to None => byte-for-byte the previous behaviour.
    if (valid_mask is not None or lengths_override is not None) and not use_kv_cache:
        raise ValueError('valid_mask/lengths_override require use_kv_cache=True')

    def model_forward_fn(inputs, commit=False):
        """
        Forward pass that computes correct sigma for SEDD.
        
        Args:
            inputs: Current sequence with masks [B, L]
            commit: If True, advance KV cache after forward
        Returns:
            logits: [B, L, V] log-probabilities (SUBS) or log-scores (SEDD)
        """
        # Compute sigma based on parameterization
        if self.parameterization == 'sedd':
            # SEDD requires sigma reflecting current noise level
            sigma = self._compute_sigma_for_exact_ll(inputs, attention_mask)
        else:
            # SUBS/AR: backbone may use sigma for time conditioning,
            # but output parameterization doesn't depend on it
            sigma = torch.zeros(inputs.shape[0], 1, device=inputs.device)
        
        with torch.no_grad():
            logits = self.forward(
                inputs,
                sigma,
                sample_mode=True,
                store_kv=commit
            )
        
        # At masked positions, exclude mask token from normalization
        # SEDD sets mask token log-score to 0, but we want -inf for exact LL        
        if self.parameterization == "sedd":
            is_masked = (inputs == self.mask_index)
            # Get explicit indices where mask is True
            batch_idx, seq_idx = is_masked.nonzero(as_tuple=True)
            # Set mask token logit to -inf at those positions
            logits[batch_idx, seq_idx, self.mask_index] = float('-inf')
            
        return logits
    
    # Determine caching strategy
    if use_kv_cache is None:
        use_kv_cache = getattr(self, 'exact_ll_use_kv_cache', False)

    extras = None
    if isinstance(self.exact_ll_strategy, BlockPermutationStrategy):
        if not hasattr(self.backbone, 'reset_kv_cache'):
            raise RuntimeError("Backbone does not support KV caching.")
        self.backbone.reset_kv_cache(eval_batch_size=x0.size(0))

        ll_total, steps, extras = compute_exact_loglikelihood_cached_permutations(
            x0=x0,
            model_forward_fn=model_forward_fn,
            mask_token_id=self.mask_index,
            strategy=self.exact_ll_strategy,
            attention_mask=attention_mask,
            block_size=self.block_size,
            return_per_order=True,
        )
    elif isinstance(self.exact_ll_strategy, BlockSubsetDPStrategy):
        if not hasattr(self.backbone, 'reset_kv_cache'):
            raise RuntimeError("Backbone does not support KV caching.")
        self.backbone.reset_kv_cache(eval_batch_size=x0.size(0))

        ll_total, steps, extras = compute_exact_loglikelihood_cached_subset_dp(
            x0=x0,
            model_forward_fn=model_forward_fn,
            mask_token_id=self.mask_index,
            strategy=self.exact_ll_strategy,
            attention_mask=attention_mask,
            block_size=self.block_size,
            return_per_order=True,
        )
    elif use_kv_cache:
        if not hasattr(self.backbone, 'reset_kv_cache'):
            raise RuntimeError("Backbone does not support KV caching.")
        self.backbone.reset_kv_cache(eval_batch_size=x0.size(0))

        ll_total, steps = compute_exact_loglikelihood_cached(
            x0=x0,
            model_forward_fn=model_forward_fn,
            mask_token_id=self.mask_index,
            strategy=self.exact_ll_strategy,
            attention_mask=attention_mask,
            block_size=self.block_size,
            valid_mask=valid_mask,
            lengths_override=lengths_override,
        )
    else:
        ll_total, steps = compute_exact_loglikelihood(
            x0=x0,
            model_forward_fn=model_forward_fn,
            mask_token_id=self.mask_index,
            strategy=self.exact_ll_strategy,
            attention_mask=attention_mask,
        )
    
    # Convert to NLL and perplexity
    answer_lens = (valid_mask.sum(dim=1) if valid_mask is not None
                   else attention_mask.sum(dim=1))
    nll = -ll_total
    nll_per_token = nll / answer_lens
    ppl = torch.exp(nll_per_token)
    
    results = {
        'nll': nll,
        'nll_per_token': nll_per_token,
        'ppl': ppl,
        'answer_lens': answer_lens,
        'steps': steps if steps is not None else None,
    }

    # Extra reductions of the same per-permutation table (block_permutation
    # only). Identical tokens, identical normalization -> directly comparable.
    if extras is not None:
        for key, ll in (('uniform_order', extras['ll_uniform_order']),
                        ('mixture', extras['ll_mixture'])):
            results[f'nll_per_token_{key}'] = (-ll) / answer_lens

    return results

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    if hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'sdpa':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(*args, **kwargs)
    elif hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'flex':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(self.device)
    if hasattr(self, 'sampling_eps_min') and torch.is_tensor(self.sampling_eps_min):
      self.sampling_eps_min = self.sampling_eps_min.to(*args, **kwargs)
      self.sampling_eps_max = self.sampling_eps_max.to(*args, **kwargs)
    return self

  def _replace_ckpt_keys(self, checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
      new_state_dict[k.replace('_orig_mod.', '')] = v
    checkpoint['state_dict'] = new_state_dict
    return checkpoint

  def on_load_checkpoint(self, checkpoint):
    logging.getLogger(__name__).info(f'Loading checkpoint at step {checkpoint["global_step"]}')
    self._restarting_skip_val_flag = True

    # for models compiled with `torch.compile`
    if '_orig_mod.' in list(checkpoint['state_dict'].keys())[0]:
      checkpoint = self._replace_ckpt_keys(checkpoint)

    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    if 'sampling_eps_min' in checkpoint.keys():
      self.sampling_eps_min = checkpoint['sampling_eps_min']
      self.sampling_eps_max = checkpoint['sampling_eps_max']
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    if hasattr(self, 'sampling_eps_min'):
      checkpoint['sampling_eps_min'] = self.sampling_eps_min
      checkpoint['sampling_eps_max'] = self.sampling_eps_max
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    # cause of overfitting for block size 1?
    if self.parameterization == 'ar':
      return None
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma, sample_mode=False, store_kv=False):
    """Returns log score.
    
    Nit: It is not clearly documented when we return logits vs log probabilities.
    """
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      if self.config.algo.name == 'bd3lm':
        logits = self.backbone(x, sigma,
                              store_kv=store_kv,
                              sample_mode=sample_mode)
      elif self.config.algo.name == 'ar':
        if self.config.algo.backbone == 'hf_dit':
          logits = self.backbone(x, None)     
        else:
          logits = self.backbone(x, sigma, sample_mode=sample_mode, store_kv=store_kv)
        logits[:, :, self.mask_index] = self.neg_infinity
        logits = logits.log_softmax(-1)
      else:
        logits = self.backbone(x, sigma)

    if self.cross_attn and not sample_mode:
      x = x[:, :self.config.model.length]
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                      xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                        xt=x,
                                        sigma=sigma)
    return logits
    
  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    del batch_idx
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'])
    self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return losses.loss

  def on_validation_epoch_start(self):
    self.metrics.reset()
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.eval()
    self.backbone.eval()
    self.noise.eval()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0
    self.sampling_eps = self.config.training.sampling_eps

  def on_validation_epoch_end(self):
    if not self.exact_eval_enabled:
      for k, v in self.metrics.valid_nlls.items():
        self.log(name=k,  value=v.compute(), on_step=False,
                on_epoch=True, sync_dist=True)
    if self.exact_eval_enabled:
      exact_nll = self.metrics.exact_valid_nlls.compute()
      exact_ppl = torch.exp(exact_nll)
      
      self.log('val/exact_nll', exact_nll,
                on_epoch=True, sync_dist=True)
      self.log('val/exact_ppl', exact_ppl,
                on_epoch=True, sync_dist=True)
      # Extra reductions of the same per-permutation table (oracle path only).
      for key in self.metrics.EXACT_EXTRA_KEYS:
        metric = self.metrics.exact_extra_metric(key)
        if float(metric.weight) == 0.0:
          continue  # strategy did not produce per-order values
        nll = metric.compute()
        self.log(f'val/exact_nll_{key}', nll,
                 on_epoch=True, sync_dist=True)
        self.log(f'val/exact_ppl_{key}', torch.exp(nll),
                 on_epoch=True, sync_dist=True)
    if self.ema:
      self.ema.restore(self._get_parameters())
    if self.var_min and not self.trainer.sanity_checking and not self.exact_eval_enabled:
      self._clipped_schedule_search()
      self.log('sampling_eps_min',
               self.sampling_eps_min,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
      self.log('sampling_eps_max',
               self.sampling_eps_max,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
  
  def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
    """Checks if the current sampling interval is valid for reporting likelihood."""
    if (sampling_eps_min == 1e-3 \
        and sampling_eps_max == 1 \
        and not (self.block_size == 1 and self.config.training.eval_nll)):
      return True # elbo
    elif (self.block_size == 1 and sampling_eps_min >= 1):
      return True # nll (block size 1)
    return False # not a valid elbo (biased estimate)
      
  def validation_step(self, batch, batch_idx):
    """
    Validation step supporting multiple evaluation modes:
    1. Exact log-likelihood (if exact_eval_enabled)
    2. Variance minimization (if var_min)
    3. Standard ELBO/NLL evaluation
    """
    # ========================================
    # Check exact evaluation first
    # ========================================
    if self.exact_eval_enabled:
        return self._validation_step_exact_likelihood(batch)
    
    # ========================================
    # Original validation logic (from GitHub)
    # ========================================
    if self.var_min:
        for noise_clip_start in self.metrics.valid_vars.keys():
            sampling_eps_min, sampling_eps_max = noise_clip_start
            if self._check_val_sampling_intvl(sampling_eps_min, sampling_eps_max) == True:
                # compute and record nelbo
                losses_clip = self._loss(batch['input_ids'],
                                  batch['attention_mask'],
                                  sampling_eps_min=sampling_eps_min,
                                  sampling_eps_max=sampling_eps_max)
                losses = Loss(
                    nlls=losses_clip.nlls.clone(),
                    token_mask=losses_clip.token_mask,
                    loss=losses_clip.loss.clone())
            elif len(self.metrics.valid_vars[noise_clip_start]) < 100:
                # elbo from clipped schedule (biased estimate)
                losses_clip = self._loss(batch['input_ids'],
                                  batch['attention_mask'],
                                  sampling_eps_min=sampling_eps_min,
                                  sampling_eps_max=sampling_eps_max)
            if len(self.metrics.valid_vars[noise_clip_start]) < 100:
                # only report variance over 100 batches
                nlls = losses_clip.nlls
                self.metrics.valid_vars[noise_clip_start].append(
                    nlls.reshape(
                        nlls.shape[0], -1, self.block_size).mean(-1))
    elif self.block_size == 1:
        # nll
        losses = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            sampling_eps_min=1,
                            sampling_eps_max=1)
    else:
        # nelbo
        losses = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            sampling_eps_min=1e-3,
                            sampling_eps_max=1)
    
    self.metrics.valid_nlls.update(losses.nlls, losses.token_mask)

    # Per-step PPL weighted by actual tokens (excludes padding)
    batch_nll = losses.nlls.sum() / losses.token_mask.sum()
    batch_ppl = torch.exp(batch_nll)
    self.log(
        'val/ppl_step',
        batch_ppl,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
        sync_dist=True
    )

    # Running average PPL (averaged in NLL space, then exponentiated)
    running_ppl = self.metrics.valid_nlls['ppl'].compute()
    self.log(
        'val/ppl_running',
        running_ppl,
        on_step=True,
        on_epoch=False,
        prog_bar=False,
        sync_dist=True
    )

    return losses.loss


  def _validation_step_exact_likelihood(self, batch):
    """
    Compute exact log-likelihood via deterministic iterative unmasking.
    
    This method works for both BD3-LM and MDLM because:
    - sample_mode=True forces both models to use non-causal attention
    - BD3-LM: Uses block-causal mask or no mask (with KV cache)
    - MDLM: Uses non-causal attention (always)
    
    Args:
        batch: Dictionary with 'input_ids' and 'attention_mask'
    
    Returns:
        torch.Tensor: Zero loss (exact evaluation doesn't use loss for backprop)
    """
    # Compute exact log-likelihood
    exact_results = self._compute_exact_ll(
        x0=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        use_kv_cache=self.exact_ll_use_kv_cache,
    )
    
    # Update cumulative metrics across batches
    self.metrics.exact_valid_nlls.update(
        exact_results['nll_per_token'],
        exact_results['answer_lens']
    )
    # Same weighting as above so all columns share one micro-average.
    for key in self.metrics.EXACT_EXTRA_KEYS:
        val = exact_results.get(f'nll_per_token_{key}')
        if val is not None:
            self.metrics.exact_extra_metric(key).update(
                val, exact_results['answer_lens'])

    # Log current batch perplexity (weighted by sequence length)
    answer_lens = exact_results['answer_lens']
    batch_nll = (exact_results['nll_per_token'] * answer_lens).sum() / answer_lens.sum()
    batch_ppl = torch.exp(batch_nll)
    self.log(
        'val/exact_ppl_step',
        batch_ppl,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
        sync_dist=True
    )

    # Log running average perplexity
    running_nll = self.metrics.exact_valid_nlls.compute()
    running_ppl = torch.exp(running_nll)
    self.log(
        'val/exact_ppl_running',
        running_ppl,
        on_step=True,
        on_epoch=False,
        prog_bar=False,
        sync_dist=True
    )

    # Log number of decoding steps (if available)
    if exact_results['steps'] is not None:
        self.log(
            'val/num_decoding_steps',
            float(exact_results['steps']),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True
        )
    
    # Return zero loss (exact evaluation doesn't use loss for optimization)
    device = batch['input_ids'].device
    return torch.tensor(0.0, device=device)

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      self._get_parameters(),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]
  
  def _resample_q_xt(
      self, x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max):
    """Resamples x_t if the percentage of masked tokens is outside the bounds
    defined by sampling_eps_min and sampling_eps_max."""
    perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    while (perc_masked < sampling_eps_min).any() or \
      (perc_masked > sampling_eps_max).any():
      # if a bound is epsilon, don't resample
      if sampling_eps_min == 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked > sampling_eps_max)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max == 1:
        regen_idx = (perc_masked < sampling_eps_min)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked < sampling_eps_min) | (perc_masked > sampling_eps_max)
      regen_idx = regen_idx.repeat_interleave(block_size,dim=-1)
      move_indices[regen_idx] = (torch.rand(
        * x.shape, device=x.device) < p)[regen_idx]
      xt = torch.where(move_indices, self.mask_index, x)
      xt = xt.reshape(xt.shape[0], -1, block_size)
      perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    return xt
  
  def q_xt(
      self, x, p, block_size=None, sampling_eps_min=None, sampling_eps_max=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      p: float torch.Tensor with shape (batch_size, 1).
      block_size: int, block size.
      sampling_eps_min: float, minimum percentage of masked tokens.
      sampling_eps_max: float, maximum percentage of masked tokens.
    """
    if block_size is None:
      block_size = self.block_size
  
    move_indices = torch.rand(
      * x.shape, device=x.device) <= p
    xt = torch.where(move_indices, self.mask_index, x)

    if block_size == 1 and sampling_eps_min == 1.0:
      return torch.full_like(x, self.mask_index)

    # no need to resample for bounds 1e-3, 1
    if self.config.training.resample and \
      not (sampling_eps_min == 1e-3 and sampling_eps_max == 1.0):
      xt = xt.reshape(xt.shape[0], -1, block_size)
      xt = self._resample_q_xt(x,
                               xt,
                               move_indices,
                               p,
                               block_size,
                               sampling_eps_min,
                               sampling_eps_max)
      xt = xt.reshape(xt.shape[0], -1)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64, device=self.device)

  @torch.no_grad()
  def _nucleus_sample(self, p_x0):
    p = self.config.sampling.nucleus_p
    if p == 1.0:
      return p_x0
    p_x0_ = p_x0[:, -self.block_size:].clone()
    sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cum_probs <= p
    nucleus_mask[..., 0] = 1
    sorted_probs = sorted_probs * nucleus_mask
    p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
    p_x0_ /= p_x0_.sum(-1, keepdim=True)
    p_x0[:, -self.block_size:] = p_x0_
    return p_x0

  @torch.no_grad()
  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    _, move_chance_t = self.noise(t)
    _, move_chance_s = self.noise(t - dt)
    sigma_t = self._sigma_from_p(move_chance_t)
    move_chance_t = move_chance_t[:, None]
    move_chance_s = move_chance_s[:, None]
    mask_prob = move_chance_s / move_chance_t

    if p_x0 is None:
      if self.config.sampling.kv_cache:
        p_x0 = self.forward(x[:, -self.block_size:],
                        sigma_t,
                        sample_mode=True).to(torch.float64)
      else:   
        p_x0 = self.forward(x,
                          sigma_t,
                          sample_mode=True).to(torch.float64)
        p_x0 = p_x0[:, -self.block_size:]
      p_x0 = p_x0.exp()
      p_x0 = self._nucleus_sample(p_x0)

    if self.config.sampling.first_hitting:
      x_block = _sample_categorical(p_x0)
      # randomly and uniformly select an index in the block (among masked tokens)
      num_masked = (x[:, -self.block_size:] == self.mask_index).sum(-1)
      ind = torch.randint(0, num_masked, (x_block.shape[0],))
      ind = (x[:, -self.block_size:] == self.mask_index).nonzero()[ind, 1]
      mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
      x_block = x_block * mask + x[:, -self.block_size:] * (1 - mask)
    else:
      q_xs = p_x0 * (1 - mask_prob)
      q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
      x_block = _sample_categorical(q_xs)
    copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
    x_block =  copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
    x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

    # compute kv cache if all tokens in a block are sampled
    if self.config.sampling.kv_cache and self.mask_index not in x_block:
      _ = self.forward(x_block, sigma_t, sample_mode=True, store_kv=True)

    if not torch.allclose(x_new, x):
      return None, x_new
    else:
      return p_x0, x_new
    
  @torch.no_grad()
  def _strategy_based_sampler(
      self, 
      n_samples: int, 
      num_strides: int, 
      seqlen: int,
  ) -> tuple[torch.Tensor, int]:
      """
      Generate samples using a selection strategy (greedy, L2R, margin, etc.).
      
      Key difference from _semi_ar_sampler:
      - Computes logits FIRST, then selects positions based on strategy
      - No noise schedule; purely iterative unmasking
      
      Args:
          n_samples: Batch size
          num_strides: Number of blocks to generate
          seqlen: Total sequence length
          
      Returns:
          x_accum: Generated sequences [B, L]
          sampling_steps: Number of forward passes
      """
      device = self.device
      sampling_steps = 0
      strategy = self.generation_strategy
      
      # Reset KV cache
      if self.config.sampling.kv_cache:
          self.backbone.reset_kv_cache(eval_batch_size=n_samples)
      
      # Initialize with BOS + masks
      x_accum = self._sample_prior(n_samples, self.block_size).to(device)
      x_accum[:, 0] = self.tokenizer.bos_token_id
      
      for stride_num in tqdm(range(num_strides), desc="Blocks"):
          # Extend sequence for new block (except first)
          if stride_num > 0:
              new_block = self._sample_prior(n_samples, self.block_size).to(device)
              x_accum = torch.cat([x_accum, new_block], dim=1)
          
          # Define current block bounds
          block_start = stride_num * self.block_size
          block_end = min(block_start + self.block_size, x_accum.shape[1])
          block_slice = slice(block_start, block_end)
          
          # Iteratively unmask within current block
          x_accum, steps = self._unmask_block_with_strategy(
              x_accum, 
              block_slice, 
              strategy,
          )
          sampling_steps += steps
          
          # Commit block to KV cache after fully unmasked
          if self.config.sampling.kv_cache:
              sigma = torch.zeros(n_samples, 1, device=device)
              _ = self.forward(
                  x_accum[:, block_slice], 
                  sigma, 
                  sample_mode=True, 
                  store_kv=True
              )
          
          # Check stopping conditions
          if x_accum.shape[1] > 256:
              stop, x_accum = self._check_stop_conds(x_accum)
              if stop and not self.config.sampling.var_length:
                  return None, None
              elif stop:
                  break
      
      return x_accum, sampling_steps


  @torch.no_grad()
  def _unmask_block_with_strategy(
      self,
      x: torch.Tensor,           # [B, L] current sequence
      block_slice: slice,        # Current block to unmask
      strategy,                  # Selection strategy
  ) -> tuple[torch.Tensor, int]:
      """
      Iteratively unmask a single block using the given strategy.
      
      Algorithm:
  ```
      while block has masked tokens:
          logits = model(x)           # Forward pass
          p_x0 = nucleus(softmax(logits))
          positions = strategy.select(log(p_x0), is_masked)
          x[positions] ~ Categorical(p_x0[positions])
  ```
      
      Returns:
          x: Sequence with block fully unmasked
          steps: Number of forward passes
      """
      batch_size, seq_len = x.shape
      device = x.device
      steps = 0
      
      # Track unmasking progress for strategy
      lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
      
      while True:
          # Check if block is fully unmasked
          is_masked_in_block = (x[:, block_slice] == self.mask_index)
          if not is_masked_in_block.any():
              break
          
          steps += 1
          
          # 1. Forward pass (only current block for BD3-LM with cache)
          sigma = torch.zeros(batch_size, 1, device=device)
          if self.config.sampling.kv_cache:
              logits = self.forward(
                  x[:, block_slice], sigma, sample_mode=True, store_kv=False
              )  # [B, block_size, V]
              # Pad to full sequence for strategy interface
              full_logits = torch.full(
                  (batch_size, seq_len, logits.shape[-1]), 
                  float('-inf'), 
                  device=device
              )
              full_logits[:, block_slice] = logits
          else:
              full_logits = self.forward(x, sigma, sample_mode=True)  # [B, L, V]
          
          # 2. Apply nucleus sampling
          p_x0 = full_logits.softmax(dim=-1)
          p_x0 = self._nucleus_sample(p_x0)
          log_probs = (p_x0 + 1e-10).log()  # Numerical stability
          
          # 3. Select positions via strategy
          is_maskable = (x == self.mask_index)
          # Restrict to current block
          block_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
          block_mask[:, block_slice] = True
          is_maskable = is_maskable & block_mask
          
          num_unmasked = (x != self.mask_index).sum(dim=1)
          
          positions = strategy.select_positions(
              log_probs=log_probs,
              is_maskable=is_maskable,
              num_unmasked=num_unmasked,
              lengths=lengths,
          )  # [B, k]
          
          # 4. Sample tokens for selected positions
          x = self._sample_at_positions(x, p_x0, positions)

          # 5. Optional instrumentation (entropy / path log-prob tracing).
          # No-op unless `self._duel_trace` has been set; see
          # scripts/entropy_kl/tracer.py.
          trace = getattr(self, '_duel_trace', None)
          if trace is not None:
              trace.record(full_logits=full_logits,
                           p_x0=p_x0,
                           positions=positions,
                           x_after=x,
                           block_slice=block_slice,
                           step_in_block=steps - 1)

      return x, steps


  def _sample_at_positions(
      self,
      x: torch.Tensor,           # [B, L]
      p_x0: torch.Tensor,        # [B, L, V] 
      positions: torch.Tensor,   # [B, k]
  ) -> torch.Tensor:
      """
      Sample tokens from p_x0 at specified positions.
      
      Uses Gumbel-max trick for sampling:
      $$
      \hat{x}_i = \arg\max_v \left[ \log p(v) - \log(-\log u_v) \right], \quad u_v \sim \text{Uniform}(0,1)
      $$
      """
      batch_size = x.shape[0]
      device = x.device
      
      valid = positions >= 0  # [B, k]
      if not valid.any():
          return x
      
      x = x.clone()
      safe_positions = positions.clamp(min=0)  # [B, k]
      
      # Gather p_x0 at selected positions: [B, k, V]
      pos_expanded = safe_positions.unsqueeze(-1).expand(-1, -1, p_x0.shape[-1])
      p_at_pos = p_x0.gather(1, pos_expanded)  # [B, k, V]
      
      # Gumbel-max sampling
      gumbel_noise = -torch.log(-torch.log(torch.rand_like(p_at_pos) + 1e-10) + 1e-10)
      sampled_tokens = (p_at_pos.log() + gumbel_noise).argmax(dim=-1)  # [B, k]
      
      # Scatter sampled tokens back, only at valid positions
      for b in range(batch_size):
          for i, pos in enumerate(positions[b]):
              if pos >= 0:
                  x[b, pos] = sampled_tokens[b, i]
      
      return x

  @torch.no_grad()
  def _ar_sampler(self, bsz, context_len=1024):
    # reset kvs
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache()

    with torch.amp.autocast('cuda', dtype=torch.float32):
      # precompute token buffer
      num_pred_tokens = self.num_tokens - 1
      x = torch.zeros(
        (bsz, num_pred_tokens + 1),
        dtype=torch.long,
        device=self.device)
      x[:, 0] = self.tokenizer.bos_token_id
      stop = False
      for i in tqdm(range(num_pred_tokens)):
        # need to sample a gumbel for each token
        # to save memory in variable-length sampling
        noise = (torch.distributions.Gumbel(0, 1)
                .sample((bsz, self.vocab_size))
                .to(self.device))
        next_logits = self.forward(
          x[:, :i + 1][:, -context_len:],
          None,
          store_kv=self.config.sampling.kv_cache)[:, -1:].to(torch.float64)
    
        next_logits = next_logits.exp()
        next_logits = self._nucleus_sample(next_logits).log()
        y = (next_logits[:, -1] + noise).argmax(-1)
        # check if we need to resample (or stop sampling for variable-length sampling)
        if (i+1) > 256:
          stop, x_out = self._check_stop_conds(x[:, :i+1])
          if stop:
            x = x_out
        if (stop and not self.config.sampling.var_length) \
          or (stop and x.shape[-1] == 1):
          return None
        elif stop:
          break
        x[:, i + 1] = y
      return x
  
  @torch.no_grad()
  def _sample(
    self, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None):
    """Generate samples from the model."""
    if seqlen is None:
      seqlen = self.config.model.length
    if batch_size_per_gpu is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    samples = []
    if self.parameterization == 'ar':
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          num_tries += 1
          sample_i = self._ar_sampler(batch_size_per_gpu)
          if num_tries > 10:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.gen_nfes.append(self.config.model.length)
      samples = torch.cat(samples, dim=0) 
      return self.tokenizer.batch_decode(samples)
    # Strategy-based sampling (new path)
    if self.generation_strategy is not None:
        num_strides = seqlen // self.block_size
        for _ in range(self.config.sampling.num_sample_batches):
            sample_i, num_tries = None, 0
            while sample_i is None and num_tries < 10:
                num_tries += 1
                sample_i, nfes = self._strategy_based_sampler(
                    n_samples=batch_size_per_gpu,
                    num_strides=num_strides,
                    seqlen=seqlen,
                )
            if sample_i is None:
                raise ValueError('Strategy-based sampling failed.')
            samples.append(sample_i)
            self.metrics.nfes.update(nfes)
            self.metrics.gen_nfes.append(nfes)
    elif self.sampler == 'semi_ar':
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          num_tries += 1
          sample_i, nfes = self._semi_ar_sampler(
            n_samples=batch_size_per_gpu,
            num_strides=max(1, seqlen // self.block_size), 
            num_steps=num_steps,
            seqlen=seqlen)
          if num_tries > 10:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.nfes.update(nfes)
        self.metrics.gen_nfes.append(nfes)
    else:
      nfes = num_steps
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          sample_i = self._analytic_sampler(
            n_samples=batch_size_per_gpu,
            num_steps=num_steps,
            seqlen=seqlen,
            eps=eps)
          num_tries += 1
          if num_tries > 10 and sample_i is None:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.nfes.update(nfes)
        self.metrics.gen_nfes.append(nfes)
    samples = torch.cat(samples, dim=0) 
    return self.tokenizer.batch_decode(samples)

  def _sigma_from_p(self, p):
    return torch.min(- torch.log(1 - p), self.noise.sigma_max)

  def restore_model_and_sample(self, num_steps, eps=1e-5, seqlen=None):
    """Generate samples from the model."""
    if self.ema:  
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(
      seqlen=seqlen,
      batch_size_per_gpu=self.config.loader.eval_batch_size,
      num_steps=num_steps,
      eps=eps)
    self.metrics.record_generative_perplexity(
      samples,
      self.config.model.length,
      self.config.loader.eval_batch_size,
      self.device)
    
    n_samples = len(samples)
    reference_dataset = datasets.load_dataset(
        "openwebtext",
        split="train[-100000:]",
        cache_dir=self.config.data.cache_dir,
        streaming=False,
        trust_remote_code=True,
    )
    if len(reference_dataset) > n_samples:
        indices = torch.randperm(len(reference_dataset))[:n_samples].tolist()
        reference_subset = reference_dataset.select(indices)
    else:
        reference_subset = reference_dataset
        
    reference_text = [ex['text'] for ex in reference_subset]
    
    def truncate_to_tokens(text, max_tokens, tokenizer):
        """Truncate text to approximately max_tokens."""
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return tokenizer.decode(tokens)
    
    max_text_length = 1024 # Adjust as needed
    reference_text = [
        truncate_to_tokens(t, max_text_length, self.tokenizer) 
        for t in reference_text
    ]
    
    mauve_score = self.metrics.record_mauve_score(
        generated_text=samples,
        reference_text=reference_text,
        max_text_length=max_text_length,
        device_id=0,
    )
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma).to(torch.float64)
    if self.config.sampling.nucleus_p == 1.0:
      return model_output.exp()
    model_output = model_output - model_output.logsumexp(-1, keepdim=True)
    model_output = self._nucleus_sample(model_output.exp())
    return model_output

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, dt):
    sigma_t = self._sigma_from_p(self.noise(t)[1])
    sigma_s = self._sigma_from_p(self.noise(t - dt)[1])
    dsigma = sigma_t - sigma_s
    score = self.get_score(x, sigma_t)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)


  def _denoiser_update(self, x, t):
    sigma = self._sigma_from_p(self.noise(t)[1])
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples


  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(
      self, batch_dims, device, sampling_eps_min, sampling_eps_max, block_size=None):
    if block_size is None:
      block_size = self.block_size
    n = batch_dims[-1]
    num_blocks = n // block_size
    _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)

    # antithetic sampling along blocks & batches (for uniform sampling)
    if self.antithetic_sampling:
      offset_b = torch.arange(batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
      offset_b = offset_b.view(batch_dims[0], num_blocks)
      _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1
    t = _eps_b
    if block_size != self.config.model.length:
      t = t.repeat_interleave(block_size, dim=-1)

    # nll
    if sampling_eps_max >= 1 and sampling_eps_min >= 1:
      return torch.ones_like(t)
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.num_tokens:
      assert seqlen == 2 * self.num_tokens
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.num_tokens)
      end = start + self.num_tokens
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation ppl, since the val
      # examples will all start and end with BOS/EOS
      if self.config.data.insert_train_special == True:
        input_tokens[:, 0] = self.tokenizer.bos_token_id
        output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    
    return input_tokens, output_tokens, new_attention_mask

  def _forward_pass_diffusion(self, x0, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if t is None:
      t = self._sample_t(x0.shape,
                         x0.device,
                         sampling_eps_min,
                         sampling_eps_max)

    loss_scale, p = self.noise(t)
    sigma = self._sigma_from_p(p[:,0].unsqueeze(-1))
    dsigma = - loss_scale * torch.expm1(sigma) # used for sedd

    # below is needed to reproduce mdlm/sedd numbers with models from sahoo et al
    # (numerical imprecision computing probs under loglinear schedule)
    if self.mdlm_loss_scale:
      sigma, dsigma = self.noise.total_noise(t), self.noise.rate_noise(t)
      p = 1 - torch.exp(-sigma)
      loss_scale = - (dsigma / torch.expm1(sigma))

    xt = self.q_xt(x0,
                   p,
                   sampling_eps_min=sampling_eps_min,
                   sampling_eps_max=sampling_eps_max)
    if sampling_eps_min is not None and sampling_eps_min > 0.5:
      loss_scale = - torch.ones_like(loss_scale)
    if self.ignore_bos:
      xt[:, 0] = x0[:, 0]
    
    x_input = xt
    if self.cross_attn:
      x_input = torch.cat((xt, x0), dim=-1)

    model_output = self.forward(x_input, sigma=sigma)

    if self.parameterization == 'sedd':
      return dsigma * self._score_entropy(
        model_output, sigma, xt, x0)

    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    loss = loss_scale * log_p_theta
    return loss

  def _loss(self, x0, attention_mask, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if sampling_eps_min is None and hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = self.sampling_eps_min
      sampling_eps_max = self.sampling_eps_max
    elif not hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)
    if self.parameterization == 'ar':
      output = self.forward(input_tokens, None)
      loss = - output.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(
        input_tokens,
        sampling_eps_min=sampling_eps_min,
        sampling_eps_max=sampling_eps_max,)
    
    if self.ignore_bos and not self.training:
      attention_mask[:, 0] = 0
      
    nlls = (loss * attention_mask)
    token_nll = nlls.sum() / attention_mask.sum()
    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _clipped_schedule_search(self):
    # collect losses per batch across devices and sum them per interval
    best_var = float('inf')
    for (eps_min, eps_max), var in self.metrics.valid_vars.items():
      all_vars = torch.tensor(0., device=self.device)
      for i in range(len(var)):
        agg_var = var[i].to(self.device)
        agg_var = self.all_gather(agg_var)
        all_vars += agg_var.var()
      if all_vars < best_var:
        best_var = all_vars
        sampling_eps_min_best = eps_min
        sampling_eps_max_best = eps_max
      self.log(f'valid_var_{round(eps_min, 2)} - {round(eps_max, 2)}',
                all_vars / len(var),
                on_epoch=True,
                on_step=False,
                sync_dist=True)
    if self.config.algo.fix_clipping == False:
      self.sampling_eps_min.fill_(sampling_eps_min_best)
      self.sampling_eps_max.fill_(sampling_eps_max_best)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def _analytic_sampler(
    self, n_samples, num_steps, seqlen, eps=1e-5): 
    x = self._sample_prior(
      n_samples,
      seqlen).to(self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    for i in tqdm(range(num_steps), desc='step'):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      x = self._analytic_update(x=x, t=t, dt=dt)
    # denoising step 
    t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                  device=self.device)
    x = self._denoiser_update(x=x, t=t)
    
    stop, x = self._check_stop_conds(x)
    if stop:
      return None
    return x

  @torch.no_grad
  def _semi_ar_sampler(
    self, n_samples, num_steps, num_strides, seqlen, context_size=1024):
    if seqlen is None:
      seqlen = self.config.model.length
    sampling_steps = 0
          
    mdlm_semi_ar = self.config.algo.name == 'mdlm' and self.config.model.length > self.block_size
    if mdlm_semi_ar:
      # sliding window of length 512 for mdlm semi-ar decoding
      num_strides = self.config.model.length // 512
      num_strides -= 1

    ones = torch.ones((n_samples,1), dtype=self.dtype,
                      device=self.device)
    
    # reset kvs
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache(eval_batch_size=self.config.loader.eval_batch_size)

    for stride_num in tqdm(range(num_strides)):
      # sample next block
      if stride_num == 0:
        x_accum = self._sample_prior(n_samples, self.block_size).to(self.device)
        x_accum[:, 0] = self.tokenizer.bos_token_id
      else:
        if mdlm_semi_ar:
          x = self._sample_prior(n_samples, 512).to(self.device)
        else:
          x = self._sample_prior(n_samples, self.block_size).to(self.device)
        x_accum = torch.cat((x_accum, x), dim=1)

      # compute logits in a sliding window (context passed to model can't exceed context_size)
      end_idx = (stride_num + 1) * self.block_size
      start_idx = max(end_idx - context_size, 0)
      fwd_idx = torch.arange(start_idx, end_idx)
      if mdlm_semi_ar and stride_num > 0: # MDLM
        fwd_idx = torch.arange(512*(stride_num), (512*(stride_num))+self.block_size)

      dt = 1 / num_steps
      p_x0_cache = None
      timesteps = torch.linspace(1, 0, num_steps, device=self.device)
      t = 1
      for i in range(num_steps):
        if self.mask_index not in x_accum:
          break

        # faster (equivalent) sampler from zheng et al (2025)
        if self.config.sampling.first_hitting:
          u = np.random.rand()
          num_masked = (x_accum[:, fwd_idx] == self.mask_index).sum(-1).item()
          t *= u**(1 / num_masked)
              
        elif not self.config.sampling.first_hitting:
          t = timesteps[i]

        p_x0_cache, x_next = self._ddpm_caching_update(
            x=x_accum[:, fwd_idx],
            t=t * ones,
            dt=dt,
            p_x0=p_x0_cache,)
        if p_x0_cache is None:
          sampling_steps += 1
       
        x_accum[:, fwd_idx] = x_next

      # check if we need to resample (or stop sampling for variable-length sampling)
      if x_accum.shape[1] > 256:
        stop, x_accum = self._check_stop_conds(x_accum)
        if (stop and not self.config.sampling.var_length) \
          or (stop and x.shape[-1] == 1):
          return None, None
        elif stop:
          break
    return x_accum, sampling_steps
  
  def _compute_entropy(self, x):
    _, counts = torch.unique(x, return_counts=True, sorted=False)
    entropy = torch.special.entr(counts.float() / counts.sum()).sum()
    return entropy
  
  def _check_stop_conds(self, x):
    """Check if sampling should stop based on 1) eos, 2) entropy, or 3) likelihood.
    Entropy/likelihood evaluated on last 256 token-block.
    
    Args:
      x: torch.Tensor, current sample.
    Returns:
      stop: bool, whether to stop sampling.
      x: torch.Tensor, sample (potentially truncated for variable-length sampling).
    """
    stop = False # stop sampling?
    truncate_idx = None # truncate sample? (variable-length sampling only)

    # CRITERION 2: always stop sampling if entropy is low
    entropy = self._compute_entropy(x[:, -256:])
    # if entropy < 4:
    #   stop = True

    # for variable length sampling, check if we should stop
    # sampling, and where to truncate the sample
    if self.config.sampling.var_length:
      # CRITERION 1: stop at sampled EOS token
      if len(torch.where(x == self.tokenizer.eos_token_id)[0]) > 1:
        stop = True
        eos_idx = torch.where(x == self.tokenizer.eos_token_id)
        if len(eos_idx[0]) > 1:
          truncate_idx = min(eos_idx[1][1]+1, x.shape[1])

      # CRITERION 2: stop if entropy/likelihood is low
      # if entropy < 4:
        # stop = True
        # truncate_idx = x.shape[1] - 256

    # truncate sample (variable-length sampling only)
    if truncate_idx is not None:
      x = x[:, :truncate_idx]
      if x.ndim == 1:
        x = x.unsqueeze(0)

    return stop, x
