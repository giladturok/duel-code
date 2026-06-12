"""
BD3-LM ELBO (Evidence Lower Bound) Evaluation for lm-eval-harness.

Key changes from v1:
- GPT-2 tokenizer (GPT2TokenizerFast)
- Mask token = 50257 (vocab_size, the extra token)
- Pad token = 50256 (eos_token, standard GPT-2 practice)
- No attention_mask passed to model (BD3LM.forward doesn't support it)
- Antithetic + stratified sampling for variance reduction

Usage:
    accelerate launch eval_bd3lm_elbo.py \
        --tasks hellaswag \
        --model bd3lm_elbo \
        --batch_size 4 \
        --model_args model_path='kuleshov-group/bd3lm-owt-block_size4',block_size=4,mc_num=128
"""
import warnings
warnings.filterwarnings('ignore', category=Warning, module='dill')

import math
from typing import Optional, Tuple, List
import random

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import wandb
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import GPT2TokenizerFast, AutoModelForMaskedLM


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LogLinearSchedule:
    """
    Loglinear noise schedule for masked diffusion.
    
    σ(t) = σ_min * (σ_max/σ_min)^t
    α(t) = 1 - exp(-σ(t))  # mask probability
    
    At t=0: α ≈ 0 (no masking)
    At t=1: α ≈ 1 (fully masked)
    """
    
    def __init__(self, sigma_min: float = 1e-4, sigma_max: float = 20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_ratio = math.log(sigma_max / sigma_min)
    
    def sigma(self, t: Tensor) -> Tensor:
        return self.sigma_min * torch.exp(self.log_ratio * t)
    
    def alpha(self, t: Tensor) -> Tensor:
        """Mask probability at time t."""
        return 1.0 - torch.exp(-self.sigma(t))
    
    def dsigma_dt(self, t: Tensor) -> Tensor:
        """Derivative dσ/dt for importance weighting."""
        return self.sigma(t) * self.log_ratio


@register_model("bd3lm_elbo")
class BD3LMELBOEvalHarness(LM):
    """
    BD3-LM ELBO evaluation for lm-eval-harness.
    
    Computes variational lower bound via Monte Carlo over noise levels,
    respecting BD3-LM's block structure where each block has independent t.
    """
    
    # GPT-2 vocab size = 50257, BD3-LM adds mask token at index 50257
    GPT2_VOCAB_SIZE = 50257
    DEFAULT_MASK_ID = 50257  # The added mask token
    DEFAULT_PAD_ID = 50256   # GPT-2 eos_token
    
    def __init__(
        self,
        model_path: str = '',
        mask_token_id: int = DEFAULT_MASK_ID,
        pad_token_id: int = DEFAULT_PAD_ID,
        max_length: int = 1024,
        batch_size: int = 32,
        mc_num: int = 128,
        block_size: int = 4,
        antithetic_sampling: bool = True,
        sampling_eps: float = 1e-3,
        device: str = "cuda",
        **kwargs,
    ):
        """
        Args:
            model_path: HuggingFace model path (e.g., 'kuleshov-group/bd3lm-owt-block_size4')
            mask_token_id: Mask token ID (50257 for BD3-LM)
            pad_token_id: Pad token ID (50256 = GPT-2 eos)
            max_length: Maximum sequence length
            batch_size: Mini-batch size for MC samples (must divide mc_num)
            mc_num: Total Monte Carlo samples for ELBO
            block_size: BD3-LM block size L' (must match trained model)
            antithetic_sampling: Use antithetic pairs for variance reduction
            sampling_eps: Minimum t value (avoid t=0 singularity)
            device: Compute device
        """
        super().__init__()
        
        # Accelerator setup
        accelerator = accelerate.Accelerator()
        self.accelerator = accelerator if accelerator.num_processes > 1 else None
        
        # Model loading
        model_kwargs = {'trust_remote_code': True, 'dtype': torch.bfloat16}
        if self.accelerator is not None:
            model_kwargs['device_map'] = {'': f'{self.accelerator.device}'}
        
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()
        
        # Device setup
        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1
        
        # GPT-2 Fast Tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Standard practice
        
        # Config
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.batch_size = int(batch_size)
        self.mc_num = int(mc_num)
        self.block_size = int(block_size)
        self.antithetic_sampling = antithetic_sampling
        self.sampling_eps = sampling_eps
        
        # Noise schedule
        self.schedule = LogLinearSchedule()
        
        # Validation
        assert self.mc_num % self.batch_size == 0, f"mc_num ({self.mc_num}) must be divisible by batch_size ({self.batch_size})"
        
        # Logging
        try:
            wandb.init(project="exact-ll", tags=["bd3lm", "elbo"], mode="disabled")
        except:
            pass
    
    @property
    def rank(self) -> int:
        return self._rank
    
    @property
    def world_size(self) -> int:
        return self._world_size
    
    def _sample_times_stratified_antithetic(
        self,
        num_samples: int,
        num_blocks: int,
        device: torch.device
    ) -> Tensor:
        """
        Generate low-discrepancy time samples using stratified + antithetic sampling.
        
        This matches the variance reduction strategy in MDLM/BD3-LM.
        
        Args:
            num_samples: Number of MC samples
            num_blocks: Number of blocks per sequence
            device: Compute device
            
        Returns:
            t: [num_samples, num_blocks] in [eps, 1]
        """
        if self.antithetic_sampling and num_samples >= 2:
            half = num_samples // 2
            
            # Stratified base samples
            u = torch.rand(half, num_blocks, device=device)
            strata = torch.arange(half, device=device).float().view(-1, 1)
            u_stratified = (u + strata) / half
            
            # Antithetic pairs
            u_antithetic = 1.0 - u_stratified
            
            t_raw = torch.cat([u_stratified, u_antithetic], dim=0)
        else:
            t_raw = torch.rand(num_samples, num_blocks, device=device)
        
        # Scale to [eps, 1]
        t = t_raw * (1.0 - self.sampling_eps) + self.sampling_eps
        return t
    
    def _mask_sequence_blockwise(
        self,
        x: Tensor,
        t: Tensor,
        prompt_lens: Tensor,
        lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply block-wise masking.
        
        Each block b has independent noise level t_b.
        Within block, each token masked i.i.d. with prob α(t_b).
        
        Args:
            x: [batch_size, seq_len] ground truth
            t: [batch_size, num_blocks] per-block noise levels
            prompt_lens: [batch_size] prompt lengths (not masked)
            lengths: [batch_size] total lengths
            
        Returns:
            z_t: [batch_size, seq_len] masked sequence
            alpha_per_token: [batch_size, seq_len] mask prob per position
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        alpha = self.schedule.alpha(t)  # [batch_size, num_blocks]
        
        z_t = x.clone()
        alpha_per_token = torch.zeros(batch_size, seq_len, device=device)
        
        for b in range(batch_size):
            prompt_len = prompt_lens[b].item()
            length = lengths[b].item()
            answer_start = prompt_len
            answer_len = length - prompt_len
            
            if answer_len <= 0:
                continue
            
            num_blocks = (answer_len + self.block_size - 1) // self.block_size
            
            for block_idx in range(num_blocks):
                block_start = answer_start + block_idx * self.block_size
                block_end = min(answer_start + (block_idx + 1) * self.block_size, length)
                block_len = block_end - block_start
                
                alpha_b = alpha[b, block_idx]
                alpha_per_token[b, block_start:block_end] = alpha_b
                
                # Sample mask
                mask = torch.rand(block_len, device=device) < alpha_b
                z_t[b, block_start:block_end] = torch.where(
                    mask,
                    self.mask_token_id,
                    x[b, block_start:block_end]
                )
        
        return z_t, alpha_per_token
    
    @torch.no_grad()
    def get_loglikelihood(
        self,
        tokens: Tensor,
        prompt_lens: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """
        Compute ELBO via Monte Carlo.
        
        ELBO = E_{t, z_t}[ Σ_i (1/α(t_i)) * (-log p(x_i | z_t)) * 1[masked] ]
        
        Args:
            tokens: [batch_size, seq_len] ground truth
            prompt_lens: [batch_size] prompt lengths
            lengths: [batch_size] sequence lengths
            
        Returns:
            ll: [batch_size] log-likelihood lower bound (= -ELBO)
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # Number of blocks for longest answer
        max_answer_len = (lengths - prompt_lens).max().item()
        num_blocks = max((max_answer_len + self.block_size - 1) // self.block_size, 1)
        
        # Generate all MC time samples
        all_t = self._sample_times_stratified_antithetic(self.mc_num, num_blocks, device)
        
        # Accumulate NLL (float64 for precision)
        nll_accum = torch.zeros(batch_size, device=device, dtype=torch.float64)
        
        num_mc_batches = self.mc_num // self.batch_size
        
        for mc_batch_idx in range(num_mc_batches):
            start_idx = mc_batch_idx * self.batch_size
            end_idx = (mc_batch_idx + 1) * self.batch_size
            
            # Process each example
            for ex_idx in range(batch_size):
                ex_tokens = tokens[ex_idx:ex_idx+1].expand(self.batch_size, -1)
                ex_prompt_len = prompt_lens[ex_idx:ex_idx+1].expand(self.batch_size)
                ex_length = lengths[ex_idx:ex_idx+1].expand(self.batch_size)
                
                t_batch = all_t[start_idx:end_idx]
                
                # Mask
                z_t, alpha_per_token = self._mask_sequence_blockwise(
                    ex_tokens, t_batch, ex_prompt_len, ex_length
                )
                
                # Forward (no attention_mask - BD3LM doesn't support it)
                sigma = self.schedule.sigma(t_batch)
                logits = self.model(z_t, sigma).logits
                
                # Log probs
                log_probs = F.log_softmax(logits.float(), dim=-1)
                true_log_probs = torch.gather(
                    log_probs, dim=-1, index=ex_tokens.unsqueeze(-1)
                ).squeeze(-1)
                
                # Masks
                is_masked = (z_t == self.mask_token_id)
                is_answer = torch.zeros_like(is_masked)
                prompt_len = ex_prompt_len[0].item()
                length = ex_length[0].item()
                is_answer[:, prompt_len:length] = True
                
                valid_mask = is_masked & is_answer
                
                # Importance weight: 1/α(t)
                weights = 1.0 / (alpha_per_token + 1e-8)
                
                # Weighted NLL
                weighted_nll = -true_log_probs * weights * valid_mask.float()
                nll_per_mc = weighted_nll.sum(dim=-1)
                
                nll_accum[ex_idx] += nll_per_mc.sum().double()
        
        # Average over MC samples
        nll_avg = nll_accum / self.mc_num
        ll = -nll_avg
        
        return ll.float()
    
    def _encode_pair(self, context: str, continuation: str) -> Tuple[List[int], List[int]]:
        """Encode context and continuation with proper whitespace handling."""
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        
        whole_enc = self.tokenizer.encode(context + continuation)
        context_enc = self.tokenizer.encode(context)
        continuation_enc = whole_enc[len(context_enc):]
        
        return context_enc, continuation_enc
    
    def _create_batch(
        self,
        elements: List[dict],
        include_target: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Create padded batch."""
        sequences = []
        prompt_lens = []
        lengths = []
        
        for elem in elements:
            if include_target:
                seq = elem["prefix"] + elem["target"]
                prompt_len = len(elem["prefix"])
            else:
                seq = elem["tokens"]
                prompt_len = 0
            
            sequences.append(seq)
            prompt_lens.append(prompt_len)
            lengths.append(len(seq))
        
        max_len = min(max(lengths), self.max_length)
        batch_size = len(elements)
        
        # Right-pad with pad_token_id
        tokens = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), max_len)
            tokens[i, :seq_len] = torch.tensor(seq[:seq_len], device=self.device)
            lengths[i] = seq_len
            prompt_lens[i] = min(prompt_lens[i], seq_len)
        
        prompt_lens = torch.tensor(prompt_lens, dtype=torch.long, device=self.device)
        lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        
        return tokens, prompt_lens, lengths
    
    def loglikelihood(self, requests: list) -> list:
        """Compute log-likelihood for (context, continuation) pairs."""
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {"prefix": prefix, "target": target}
        
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        
        out = []
        # Reduce eval batch size since we expand for MC samples
        eval_batch_size = max(1, self.batch_size // (self.mc_num // self.batch_size))
        num_batches = math.ceil(len(ds) / eval_batch_size)
        
        for batch_idx in tqdm(range(num_batches), desc="Computing ELBO..."):
            batch_start = batch_idx * eval_batch_size
            batch_end = min((batch_idx + 1) * eval_batch_size, len(ds))
            batch_elements = [ds[i] for i in range(batch_start, batch_end)]
            
            tokens, prompt_lens, lengths = self._create_batch(batch_elements, include_target=True)
            ll_batch = self.get_loglikelihood(tokens, prompt_lens, lengths)
            
            for i in range(len(batch_elements)):
                # lm-eval-harness expects (ll, is_greedy)
                out.append((ll_batch[i].item(), False))
        
        torch.cuda.empty_cache()
        return out
    
    def loglikelihood_rolling(self, requests: list) -> list:
        """Compute rolling log-likelihood for documents."""
        all_tokens = [
            self.tokenizer.encode(req.args[0])
            for req in tqdm(requests, desc="Tokenizing...")
        ]
        
        all_chunks = []
        doc_chunk_counts = []
        
        for tokens in all_tokens:
            num_chunks = max(1, math.ceil(len(tokens) / self.max_length))
            doc_chunk_counts.append(num_chunks)
            
            for start_idx in range(0, len(tokens), self.max_length):
                chunk = tokens[start_idx:start_idx + self.max_length]
                all_chunks.append({"tokens": chunk})
        
        chunk_lls = []
        eval_batch_size = max(1, self.batch_size // (self.mc_num // self.batch_size))
        num_batches = math.ceil(len(all_chunks) / eval_batch_size)
        
        for batch_idx in tqdm(range(num_batches), desc="Computing rolling ELBO..."):
            batch_start = batch_idx * eval_batch_size
            batch_end = min((batch_idx + 1) * eval_batch_size, len(all_chunks))
            batch_elements = all_chunks[batch_start:batch_end]
            
            tokens, prompt_lens, lengths = self._create_batch(batch_elements, include_target=False)
            ll_batch = self.get_loglikelihood(tokens, prompt_lens, lengths)
            chunk_lls.extend([ll.item() for ll in ll_batch])
        
        # Aggregate to documents
        out = []
        chunk_idx = 0
        for num_chunks in doc_chunk_counts:
            doc_ll = sum(chunk_lls[chunk_idx:chunk_idx + num_chunks])
            out.append(doc_ll)
            chunk_idx += num_chunks
        
        torch.cuda.empty_cache()
        return out
    
    def generate_until(self, requests: list) -> list:
        raise NotImplementedError("ELBO evaluator does not support generation.")


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()