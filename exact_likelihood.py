"""
Exact NLL computation for masked diffusion models (BD3-LM compatible).
"""
import warnings
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor


def compute_exact_loglikelihood(
    x0: Tensor,  # [batch_size, seq_len]
    model_forward_fn,  # Function that takes (x, sigma) and returns logits
    mask_token_id: int,
    strategy: object,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute exact log-likelihood using deterministic decoding.
    
    Args:
        x0: Full sequence [B, L]
        model_forward_fn: Callable that takes (x, sigma) -> logits
        mask_token_id: Token ID for masking
        strategy: Decoding strategy (e.g., BlockGreedyConfidenceStrategy)
        attention_mask: Attention mask [B, L]
    
    Returns:
        ll_total: [batch_size] total log-likelihood (not per-token)
    """
    batch_size, seq_len = x0.shape
    device = x0.device

    lengths = (
        attention_mask.sum(dim=1)
        if attention_mask is not None
        else torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    ) # per-example lengths [B]
    
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    is_valid_answer = positions < lengths.unsqueeze(1) # account for padding [B, L]

    mask_fill = x0.new_full(x0.shape, mask_token_id)
    z_k = torch.where(is_valid_answer, mask_fill, x0)

    num_unmasked = torch.zeros(batch_size, dtype=torch.long, device=device) # [B]
    ll_total = torch.zeros(batch_size, device=device) # [B]
    max_steps = int(lengths.max().item())

    steps = 0
    while (num_unmasked < lengths).any():
        if steps >= max_steps:
            warnings.warn("Exact LL loop exhausted max answer length", stacklevel=2)
            break
        steps += 1

        done = num_unmasked >= lengths
        logits = model_forward_fn(z_k)  # [B, L, V]
        log_probs = _log_softmax_inplace(logits, dim=-1)

        is_maskable = (z_k == mask_token_id) & is_valid_answer  # [B, L]
        positions = strategy.select_positions(
            log_probs,
            is_maskable,
            num_unmasked=num_unmasked,
            lengths=lengths,
        )

        ll_total += _compute_ll_for_positions(log_probs, x0, positions, ~done)

        z_k = _unmask_positions(z_k, x0, positions)
        num_unmasked += (positions >= 0).sum(dim=1)

    return ll_total


def _compute_ll_for_positions(
    log_probs: Tensor, x0: Tensor, positions: Tensor, is_active: Tensor
) -> Tensor:
    true_log_probs = log_probs.gather(2, x0.unsqueeze(-1)).squeeze(-1)

    valid = positions >= 0
    safe_positions = positions.clamp(min=0)
    gathered = true_log_probs.gather(1, safe_positions)
    ll = (gathered * valid).sum(dim=1)
    return ll * is_active


def _unmask_positions(z_k: Tensor, x0: Tensor, positions: Tensor) -> Tensor:
    valid = positions >= 0
    if not valid.any():
        return z_k

    safe_positions = positions.clamp(min=0)
    updates = x0.gather(1, safe_positions)
    current = z_k.gather(1, safe_positions)
    final = torch.where(valid, updates, current)
    return z_k.scatter(1, safe_positions, final)


def _log_softmax_inplace(x: Tensor, dim: int = -1) -> Tensor:
    """
    Numerically stable in-place log_softmax.
    
    Computes: x_i = x_i - log(sum_j(exp(x_j)))
    """
    x.sub_(torch.logsumexp(x, dim=dim, keepdim=True))
    return x
