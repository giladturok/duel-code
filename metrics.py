import torch
from torch import Tensor
import torchmetrics
import typing
from typing import Union
import transformers
import os
import torch.nn.functional as F
from tqdm import tqdm
import math
import mauve

LOG2 = torch.log(torch.tensor(2.0))

class NLL(torchmetrics.aggregation.MeanMetric):
  pass

class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2

class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
      Perplexity
    """
    return torch.exp(self.mean_value / self.weight)

class NFEs(torchmetrics.aggregation.MeanMetric):
  pass

class Metrics:
  # Extra exact-NLL reductions emitted only by the `block_permutation` (oracle)
  # path: the uniform-order expectation and the normalized order-mixture, both
  # reduced from the same per-permutation table as the oracle max.
  EXACT_EXTRA_KEYS = ('uniform_order', 'mixture')

  def exact_extra_metric(self, key):
    return getattr(self, f'exact_valid_nlls_{key}')

  def __init__(self, config=None) -> None:
    self.config=config
    metrics = torchmetrics.MetricCollection({
        'nll': NLL(), 'bpd': BPD(), 'ppl': Perplexity()})
    if hasattr(config, 'block_size'):
      self.block_size = config.block_size
    else:
      self.block_size = config.model.length
    self.exact_eval_enabled = (config.mode == 'duel_ppl')
    if self.exact_eval_enabled:
      self.exact_valid_nlls = torchmetrics.aggregation.MeanMetric()
      # Extra reductions over the same per-permutation table, populated only by
      # the `block_permutation` (oracle) path. Same weighting (`answer_lens`)
      # as `exact_valid_nlls` so the token-weighted micro-average matches.
      self.exact_valid_nlls_uniform_order = \
        torchmetrics.aggregation.MeanMetric()
      self.exact_valid_nlls_mixture = \
        torchmetrics.aggregation.MeanMetric()

    self.nfes = NFEs()
    self.train_nlls = metrics.clone(prefix='train/')
    self.valid_nlls = metrics.clone(prefix='val/')
    self.gen_ppl = Perplexity()
    self.gen_entropy = NLL()
    self.gen_ppls, self.gen_nfes, self.gen_entropies, self.gen_lengths \
      = [], [], [], []
    self.mauve_scores = []
    self.mauve_score_mean = torchmetrics.aggregation.MeanMetric()

    self.sampling_eps = config.training.sampling_eps
    if getattr(config.algo, 'clip_search_delta', None):
      self.clip_search_delta = config.algo.clip_search_delta
    self.valid_vars = {self.sampling_eps: []}
    if getattr(config.algo, 'var_min', None):
      self.valid_vars = self.init_valid_vars()
    self.eval_ppl_batch_size = \
     self.config.eval.perplexity_batch_size
    self.gen_ppl_eval_model_name_or_path = \
      config.eval.gen_ppl_eval_model_name_or_path
    self.tokenizer = transformers.AutoTokenizer.\
      from_pretrained(self.gen_ppl_eval_model_name_or_path)
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

  def init_valid_vars(self):
    eps = self.sampling_eps
    if self.block_size > 1:
      eps = self.sampling_eps
      self.valid_vars = {(eps, 1): []}
      for width in self.config.algo.clip_search_widths:
        for i in torch.arange(0, 1 - width + self.clip_search_delta, self.clip_search_delta):
          min = torch.clamp(i, min=self.sampling_eps).item()
          max = torch.clamp(i + width, min=self.sampling_eps).item()
          self.valid_vars[(min, max)] = []
    else:
      eps = self.sampling_eps
      self.valid_vars = {
        (eps, 1): [],
        (1, 1): []}


  def to(self, *args, **kwargs):
    self.train_nlls = self.train_nlls.to(*args, **kwargs)
    self.valid_nlls = self.valid_nlls.to(*args, **kwargs)
    self.gen_ppl = self.gen_ppl.to(*args, **kwargs)
    self.nfes = self.nfes.to(*args, **kwargs)
    self.gen_entropy = self.gen_entropy.to(*args, **kwargs)
    self.mauve_score_mean = self.mauve_score_mean.to(*args, **kwargs)
    
    # Add exact_valid_nlls if it exists
    if self.exact_eval_enabled:
        self.exact_valid_nlls = self.exact_valid_nlls.to(*args, **kwargs)
        for k in self.EXACT_EXTRA_KEYS:
          setattr(self, f'exact_valid_nlls_{k}',
                  self.exact_extra_metric(k).to(*args, **kwargs))

  def reset(self):
    self.gen_ppls, self.gen_nfes, self.gen_entropies, self.gen_lengths \
      = [], [], [], []
    self.train_nlls.reset()
    self.valid_nlls.reset()
    self.gen_ppl.reset()
    self.gen_entropy.reset()
    self.nfes.reset()
    if getattr(self.config.algo, 'var_min', None):
      self.init_valid_vars()
    if self.exact_eval_enabled:
      self.exact_valid_nlls.reset()
      for k in self.EXACT_EXTRA_KEYS:
        self.exact_extra_metric(k).reset()
    self.mauve_scores = []
    self.mauve_score_mean.reset()

  @torch.no_grad()
  def _eval_retokenize(self, text_samples, max_length,
                       device):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.tokenizer(text_samples,
                             **tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(device)
      samples = samples.to(device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def record_generative_perplexity(
    self,
    text_samples: typing.List[str],
    max_length: int,
    batch_size: Union[int, None] = None,
    retokenize: bool = True,
    stride=512,
    device='cuda') -> None:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path).eval()
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self._eval_retokenize(
         text_samples, max_length=max_length, device=device)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(device)
      eval_context_size = samples.shape[-1]
    if batch_size is None:
      batch_size = min(self.eval_ppl_batch_size,
                      samples.shape[0])

    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      samples_batch = samples[i * batch_size: (i + 1) * batch_size]
      attn_mask_batch = attn_mask[i * batch_size: (i + 1) * batch_size]

      nlls_accum = torch.zeros_like(samples_batch, dtype=torch.float32)
      valid_tokens_accum = torch.zeros_like(samples_batch, dtype=torch.float32)

      num_strides = math.ceil((samples_batch.shape[-1] - eval_context_size + stride) / stride)
      num_strides = max(num_strides, 1)

      # Computes Gen. PPL in a sliding window for sequences longer than eval_context_size
      for i in tqdm(range(num_strides), desc='Sliding Window Gen PPL'):
        if i == 0:
          # for the first stride, use the entire eval_context_size
          start = 0
          end = min(eval_context_size, samples_batch.shape[-1])
        else:
          # then, move the window by stride
          start = i * stride
          end = min(start + eval_context_size, samples_batch.shape[-1])
        sample_chunk = samples_batch[..., start:end]
        attn_mask_chunk = attn_mask_batch[..., start:end]

        logits = eval_model(sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1], sample_chunk[..., 1:], reduction='none')
        valid_tokens = (sample_chunk[..., 1:] != self.tokenizer.eos_token_id).to(torch.float)
        
        if i == 0:
          # for the first stride, update the nlls of the entire eval_context_size
          nlls_accum[..., start + 1:end] += nlls
          valid_tokens_accum[..., start + 1:end] += valid_tokens
        else:
          # only update the nlls of the last stride
          update_start = (start+eval_context_size-stride)
          update_window = end - update_start
          nlls_accum[...,update_start:end] += nlls[..., -update_window:]
          valid_tokens_accum[..., update_start:end] += valid_tokens[..., -update_window:]
          
      # gen ppl
      avg_nll = (nlls_accum * valid_tokens_accum).sum() / valid_tokens_accum.sum()
      self.gen_ppls.append(avg_nll.exp().detach().cpu().item())
      self.gen_ppl.update(nlls_accum, valid_tokens_accum)

      # entropy
      entropy_full = 0
      for i in range(samples_batch.shape[0]):
        _, counts = torch.unique(samples_batch[i], return_counts=True, sorted=False)
        entropy = torch.special.entr(counts.float() / counts.sum()).sum()
        entropy_full += entropy
      self.gen_entropies.append(entropy_full.detach().cpu().item())
      self.gen_entropy.update(entropy_full, samples_batch.shape[0])

      # record sample length
      self.gen_lengths.append(valid_tokens_accum.sum().detach().cpu().item())
      
  @torch.no_grad()
  def record_mauve_score(
      self,
      generated_text: typing.List[str],
      reference_text: typing.List[str],
      prompts: typing.Optional[typing.List[str]] = None,
      max_text_length: int = 256,
      device_id: int = 0,
      featurize_model_name: str = "gpt2-large",
      verbose: bool = False,
  ) -> float:
      """Compute MAUVE score between generated and reference text distributions.
      
      Args:
          generated_text: List of generated samples (continuations or full text).
          reference_text: List of reference/gold samples (same length as generated_text).
          prompts: Optional list of prompts. If provided, prepends to both generated 
                  and reference text. Length must match generated_text.
          max_text_length: Max tokens for MAUVE computation (truncates longer seqs).
          device_id: GPU device for MAUVE featurization.
          featurize_model_name: Model for computing embeddings (default: gpt2-large).
          verbose: Print MAUVE internals.
      
      Returns:
          mauve_score: float in [0, 1], higher = better distributional match.
      """
      
      assert len(generated_text) == len(reference_text), \
          f"Length mismatch: {len(generated_text)} generated vs {len(reference_text)} reference"
      
      # Prepend prompts if provided
      if prompts is not None:
          assert len(prompts) == len(generated_text), \
              f"Length mismatch: {len(prompts)} prompts vs {len(generated_text)} generated"
          machine_text = [f"{p}{g}" for p, g in zip(prompts, generated_text)]
          human_text = [f"{p}{r}" for p, r in zip(prompts, reference_text)]
      else:
          machine_text = generated_text
          human_text = reference_text
      
      # Compute MAUVE
      # P = human (reference), Q = machine (generated)
      out = mauve.compute_mauve(
          p_text=human_text,
          q_text=machine_text,
          device_id=device_id,
          max_text_length=max_text_length,
          featurize_model_name=featurize_model_name,
          verbose=verbose,
      )
      
      score = out.mauve
      self.mauve_scores.append(score)
      self.mauve_score_mean.update(torch.tensor(score), 1)
      
      return score
