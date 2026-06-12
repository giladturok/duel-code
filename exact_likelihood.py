"""
Exact NLL computation for masked diffusion models (BD3-LM compatible).
"""
from itertools import permutations
import math
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

    return ll_total, steps

def compute_exact_loglikelihood_cached(
    x0: Tensor,  # [B, L]
    model_forward_fn,  # Callable: (x_block, commit=False) -> logits_block
    mask_token_id: int,
    strategy: object,
    attention_mask: Optional[Tensor] = None,  # [B, L]
    block_size: Optional[int] = None,
) -> Tensor:  # [B] - total log-likelihood per sequence
    """
    Compute exact log-likelihood using block-wise KV caching.
    
    Args:
        x0: True sequence
        model_forward_fn: Callable(x_block, commit=False) -> logits_block
            - Takes current block [B, block_size]
            - Returns logits for that block [B, block_size, V]
            - commit=True advances the cache pointer
        mask_token_id: Token ID for masked positions
        strategy: Selection strategy for unmasking
        attention_mask: Valid token mask
        block_size: Block size for processing
    
    Cache mechanism:
    - For block b, model attends to:
      * Cached blocks 0..b-1 (fully unmasked, KV values stored)
      * Current block b (partially unmasked, computed on-the-fly)
    - After block b is fully unmasked, commit it to cache
    """
    if block_size is None:
        raise ValueError("block_size must be provided for cached LL computation.")

    batch_size, seq_len = x0.shape
    device = x0.device
    steps = 0

    # Compute valid positions (account for padding)
    lengths = (
        attention_mask.sum(dim=1)
        if attention_mask is not None
        else torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    )  # [B]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, L]
    is_valid = positions < lengths.unsqueeze(1)  # [B, L]

    # Initialize: all valid positions masked
    mask_fill = x0.new_full(x0.shape, mask_token_id)
    z_k = torch.where(is_valid, mask_fill, x0)  # [B, L]
    
    num_unmasked = torch.zeros(batch_size, dtype=torch.long, device=device)  # [B]
    ll_total = torch.zeros(batch_size, device=device)  # [B]

    # Process each block sequentially
    num_blocks = (seq_len + block_size - 1) // block_size
    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, seq_len)
        block_slice = slice(block_start, block_end)

        # Skip if no valid tokens in this block
        block_valid = is_valid[:, block_slice]  # [B, block_size]
        if not block_valid.any().item():
            continue

        # Iteratively unmask current block
        while _has_masked_tokens_in_block(
            z_k[:, block_slice], mask_token_id, block_valid
        ):
            # Forward pass: model attends to cached blocks + current block
            logits_block = model_forward_fn(z_k[:, block_slice], commit=False)  # [B, block_size, V]
            log_probs_block = F.log_softmax(logits_block, dim=-1)  # [B, block_size, V]

            # Pad to full sequence for strategy interface
            log_probs = _pad_block_logits_to_full(log_probs_block, block_start, seq_len)  # [B, L, V]

            # Select positions to unmask
            is_maskable = (z_k == mask_token_id) & is_valid  # [B, L]
            positions = strategy.select_positions(
                log_probs,
                is_maskable,
                num_unmasked=num_unmasked,
                lengths=lengths,
            )  # [B, k]
            
            # Verify progress
            newly_selected = (positions >= 0).sum(dim=1)  # [B]
            if not newly_selected.any().item():
                raise RuntimeError(
                    "Strategy failed to select new positions while maskable tokens remain."
                )
                
            # Accumulate log-likelihood
            active = num_unmasked < lengths  # [B]
            ll_total += _compute_ll_for_positions(log_probs, x0, positions, active)

            # Unmask selected positions
            z_k = _unmask_positions(z_k, x0, positions)
            num_unmasked += newly_selected
            steps += 1
            
        # Commit block to cache (advances cache_idx for next block)
        model_forward_fn(z_k[:, block_slice], commit=True)

    return ll_total, steps


def compute_exact_loglikelihood_cached_permutations(
    x0: Tensor,  # [B, L]
    model_forward_fn,  # Callable: (x_block, commit=False) -> logits_block
    mask_token_id: int,
    strategy: Optional[object] = None,
    attention_mask: Optional[Tensor] = None,  # [B, L]
    block_size: Optional[int] = None,
):  # -> (ll_total: [B], steps: int)
    """
    Compute exact log-likelihood using block-wise KV caching and all permutations.
    
    
    Args:
        x0: True sequence
        model_forward_fn: Callable(x_block, commit=False) -> logits_block
            - Takes current block [B, block_size]
            - Returns logits for that block [B, block_size, V]
            - commit=True advances the cache pointer
        mask_token_id: Token ID for masked positions
        strategy: Selection strategy for unmasking
        attention_mask: Valid token mask
        block_size: Block size for processing
    
    Cache mechanism:
    - For block b, model attends to:
      * Cached blocks 0..b-1 (fully unmasked, KV values stored)
      * Current block b (partially unmasked, computed on-the-fly)
    - After block b is fully unmasked, commit it to cache
    """
    if block_size is None:
        raise ValueError("block_size must be provided for cached LL computation.")

    batch_size, seq_len = x0.shape
    device = x0.device
    steps = 0

    # Compute valid positions (account for padding)
    lengths = (
        attention_mask.sum(dim=1)
        if attention_mask is not None
        else torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    )  # [B]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, L]
    is_valid = positions < lengths.unsqueeze(1)  # [B, L]

    # Initialize: all valid positions masked
    mask_fill = x0.new_full(x0.shape, mask_token_id)
    z_k = torch.where(is_valid, mask_fill, x0)  # [B, L]

    num_unmasked = torch.zeros(batch_size, dtype=torch.long, device=device)  # [B]
    ll_total = torch.zeros(batch_size, device=device)  # [B]

    # Process each block sequentially
    num_blocks = (seq_len + block_size - 1) // block_size
    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, seq_len)
        block_slice = slice(block_start, block_end)

        # Skip if no valid tokens in this block
        block_valid = is_valid[:, block_slice]  # [B, block_size]
        if not block_valid.any().item():
            continue

        # Freeze current z_k for permutation trials
        z_k_freeze = z_k.clone()

        # Track best permutation for this block
        ll_block_best = torch.full((batch_size,), float('-inf'), device=device)
        z_k_best = z_k.clone()
        num_unmasked_best = num_unmasked.clone()

        # Try all permutations of positions in the block
        actual_block_size = block_end - block_start
        block_permutations = permutations(range(actual_block_size))
        for perm in block_permutations:

            # Reset z_k and num_unmasked for each permutation
            z_k_perm = z_k_freeze.clone()
            num_unmasked_perm = num_unmasked.clone()
            ll_block_perm = torch.zeros(batch_size, device=device)  # [B] - LL for this block only

            # Process current block with given permutation
            z_k_perm, num_unmasked_perm, ll_block_perm = _loop_fn(
                perm, z_k_perm, num_unmasked_perm, ll_block_perm, block_start, block_slice, model_forward_fn, mask_token_id, lengths, x0, is_valid
            )

            # Compare to best permutation found for this block so far
            ll_better = (ll_block_perm > ll_block_best)  # [B]

            # Update best for this block if this permutation is better
            z_k_best = torch.where(ll_better.unsqueeze(1), z_k_perm, z_k_best)
            num_unmasked_best = torch.where(ll_better, num_unmasked_perm, num_unmasked_best)
            ll_block_best = torch.where(ll_better, ll_block_perm, ll_block_best)

        # After trying all permutations, update global state with best permutation
        z_k = z_k_best
        num_unmasked = num_unmasked_best
        ll_total += ll_block_best  # Add best block LL to cumulative total

        # NFE accounting: each permutation does `actual_block_size` forwards
        # (one per position via _loop_fn), and there are actual_block_size!
        # permutations evaluated per block.
        steps += math.factorial(actual_block_size) * actual_block_size

        # Commit block to cache (advances cache_idx for next block)
        model_forward_fn(z_k[:, block_slice], commit=True)

    return ll_total, steps


def _loop_fn(
    perm, 
    z_k_perm, 
    num_unmasked_perm, 
    ll_total_perm, 
    block_start, 
    block_slice, 
    model_forward_fn, 
    mask_token_id, 
    lengths, 
    x0, 
    is_valid
):
    batch_size, seq_len = z_k_perm.shape
    device = z_k_perm.device

    # Iteratively unmask current block
    for pos in perm:
        
        # Forward pass: model attends to cached blocks + current block
        logits_block = model_forward_fn(z_k_perm[:, block_slice], commit=False)  # [B, block_size, V]
        log_probs_block = F.log_softmax(logits_block, dim=-1)  # [B, block_size, V]

        # Pad to full sequence for strategy interface
        log_probs = _pad_block_logits_to_full(log_probs_block, block_start, seq_len)  # [B, L, V]

        # Select position to unmask via pos and without strategy
        is_maskable = (z_k_perm == mask_token_id) & is_valid  # [B, L]
        positions = torch.full((batch_size, 1), -1, dtype=torch.long, device=device)  # [B, 1]
        for b in range(batch_size):
            if is_maskable[b, block_start + pos]:
                positions[b, 0] = block_start + pos
                
        # Accumulate log-likelihood
        active = num_unmasked_perm < lengths  # [B]
        ll_total_perm += _compute_ll_for_positions(log_probs, x0, positions, active)
        
        # Unmask selected positions
        z_k_perm = _unmask_positions(z_k_perm, x0, positions)
        num_unmasked_perm += (positions >= 0).sum(dim=1)
        
    return z_k_perm, num_unmasked_perm, ll_total_perm


def _compute_ll_for_positions(
    log_probs: Tensor, x0: Tensor, positions: Tensor, is_active: Tensor
) -> Tensor:
    true_log_probs = log_probs.gather(2, x0.unsqueeze(-1)).squeeze(-1)

    valid = positions >= 0 # TODO: revert back to 0
    safe_positions = positions.clamp(min=0)
    gathered = true_log_probs.gather(1, safe_positions)
    
    # Replace invalid entries with 0 BEFORE multiplication
    gathered_safe = torch.where(valid, gathered, torch.zeros_like(gathered))
    ll = gathered_safe.sum(dim=1)
    return torch.where(is_active, ll, torch.zeros_like(ll))


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


def _pad_block_logits_to_full(
    logits_block: Tensor,  # [B, block_size, V]
    block_start: int,
    seq_len: int,
) -> Tensor:  # [B, L, V]
    """Embed block logits into full sequence, padding with -inf."""
    batch_size, block_size, vocab_size = logits_block.shape
    log_probs = logits_block.new_full(
        (batch_size, seq_len, vocab_size),
        float("-inf"),
    )
    log_probs[:, block_start:block_start + block_size] = logits_block
    return log_probs


def _has_masked_tokens_in_block(
    z_block: Tensor,  # [B, block_size]
    mask_token_id: int,
    valid_mask: Tensor,  # [B, block_size]
) -> bool:
    """Check if block has any masked tokens in valid positions."""
    maskable = (z_block == mask_token_id) & valid_mask
    return maskable.any().item()