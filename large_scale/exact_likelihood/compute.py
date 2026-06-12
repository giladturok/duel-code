"""
Deterministic and stochastic NLL computation for masked diffusion models.

Key design principles:
1. Separate deterministic vs stochastic (NELBO) computation paths
2. Strategies return positions to unmask (variable k handled naturally)
3. Use is_valid_answer mask to exclude prompt/padding from NLL
4. Log-scale convention: -inf to exclude from argmax, 0.0 to exclude from sum
"""
from typing import Optional
import itertools

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from exact_likelihood.strategies import BlockPermutationStrategy

# ============================================================================
# Main API
# ============================================================================

def compute_deterministic_loglikelihood(
    model: nn.Module,
    x: Tensor,  # [batch_size, seq_len] full sequence
    prompt_lens: Tensor,  # [batch_size]
    lengths: Tensor,  # [batch_size] - total length (prompt + answer)
    mask_token_id: int,
    strategy: object,  # Any strategy with select_positions method
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute LL with given decoding strategy.
    
    Deterministic unmasking: maintain state z_k, progressively unmask tokens.
   
    Args:
        x: Full sequence (prompt + answer)
        prompt_lens: Length of prompt per sample (NLL computed on answer only)
        lengths: Total length per sample (prompt + answer, excluding padding)
        normalize: Divide by answer length (True by default)
    
    Returns:
        ll: [batch_size] total LL (not normalized)
    """
    batch_size, seq_len = x.shape
    device = x.device
    
    # Valid answer positions (where we compute NLL)
    is_valid_answer = _get_valid_answer_mask(
        batch_size, seq_len, prompt_lens, lengths, device
    )
    answer_lens = lengths - prompt_lens
    
    # Initialize: mask all answer positions, keep prompt and padding unchanged
    mask_fill = torch.full_like(x, mask_token_id)
    z_k = torch.where(is_valid_answer, mask_fill, x)
    
    # Track progress
    num_unmasked = torch.zeros(batch_size, dtype=torch.long, device=device)
    ll_total = torch.zeros(batch_size, device=device)
    
    # Continue until all samples done
    max_iterations = answer_lens.max().item() + 100  # Safety limit
    
    for iteration in range(max_iterations):
    
        # Which samples still have masked tokens?
        is_active = num_unmasked < answer_lens
        if not is_active.any():
            break
        
        # Which positions can be unmasked?
        is_maskable = (z_k == mask_token_id) & is_valid_answer
        
        # Get model predictions
        logits = model(z_k, attention_mask=attention_mask).logits  # [batch_size, seq_len, vocab_size]
        
        # Select positions to unmask
        positions = strategy.select_positions(
            logits, is_maskable, num_unmasked=num_unmasked, prompt_lens=prompt_lens
        )# positions: [batch_size, max_k] with -1 padding
        
        # Compute LL for selected positions
        ll_step = _compute_ll_for_positions(
            logits, x, positions, is_active
        ) # [batch_size]
        ll_total += ll_step
        del logits # Free memory
        
        # Unmask selected positions
        z_k = _unmask_positions(z_k, x, positions)
        
        # Update unmasked count per sample
        k_unmasked = (positions >= 0).sum(dim=1)  # [batch_size]
        num_unmasked += k_unmasked
    
    return ll_total


# ============================================================================
# Block Permutation Search
# ============================================================================

def compute_block_permutation_loglikelihood(
    model: nn.Module,
    x: Tensor,  # [batch_size, seq_len] full sequence
    prompt_lens: Tensor,  # [batch_size]
    lengths: Tensor,  # [batch_size] - total length (prompt + answer)
    mask_token_id: int,
    block_size: int = 4,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute LL using exhaustive block permutation search.
    
    For each block of size `block_size`, evaluates all k! permutations
    of unmasking order. Within each permutation, unmasks greedily k=1
    token at a time. Selects the permutation with highest likelihood.
    
    This is an oracle method that requires access to ground truth x.
    
    Args:
        x: Full sequence (prompt + answer)
        prompt_lens: Length of prompt per sample (NLL computed on answer only)
        lengths: Total length per sample (prompt + answer, excluding padding)
        mask_token_id: Token ID representing [MASK]
        block_size: Size of blocks (4 means evaluate all 4! = 24 permutations)
        attention_mask: Optional attention mask
    
    Returns:
        ll: [batch_size] total log-likelihood (not normalized)
    
    Example:
        For block_size=4, each block will evaluate 24 permutations:
        [0,1,2,3], [0,1,3,2], [0,2,1,3], ..., [3,2,1,0]
        
        Each permutation unmasks one token at a time:
        - Permutation [0,2,1,3]: unmask pos 0, then pos 2, then pos 1, then pos 3
        - Permutation [3,1,2,0]: unmask pos 3, then pos 1, then pos 2, then pos 0
    """
    import itertools
    
    batch_size, seq_len = x.shape
    device = x.device
    
    # Generate all permutations once
    all_perms = list(itertools.permutations(range(block_size)))
    num_perms = len(all_perms)  # 24 for block_size=4
    
    # Valid answer positions (where we compute NLL)
    is_valid_answer = _get_valid_answer_mask(
        batch_size, seq_len, prompt_lens, lengths, device
    )
    answer_lens = lengths - prompt_lens
    
    # Initialize: mask all answer positions, keep prompt and padding unchanged
    mask_fill = torch.full_like(x, mask_token_id)
    z_k = torch.where(is_valid_answer, mask_fill, x)
    
    ll_total = torch.zeros(batch_size, device=device)
    
    # Process each sample independently (batch across permutations for each sample)
    for b in range(batch_size):
        answer_len = answer_lens[b].item()
        prompt_len = prompt_lens[b].item()
        num_blocks = (answer_len + block_size - 1) // block_size
        
        # Current state for this sample [1, seq_len]
        z_sample = z_k[b:b+1].clone()
        attn_sample = attention_mask[b:b+1] if attention_mask is not None else None
        
        for block_idx in range(num_blocks):
            # Block boundaries (absolute positions in sequence)
            block_start = prompt_len + block_idx * block_size
            block_end = min(block_start + block_size, prompt_len + answer_len)
            actual_block_size = block_end - block_start
            
            if actual_block_size == 0:
                continue
            
            # Evaluate all valid permutations for this block
            # (fewer than k! for partial blocks at the end)
            if actual_block_size < block_size:
                perms_to_eval = list(itertools.permutations(range(actual_block_size)))
            else:
                perms_to_eval = all_perms
            
            num_valid_perms = len(perms_to_eval)
            
            # Batch evaluate all permutations: [num_perms, seq_len]
            ll_all_perms = _evaluate_block_permutations_batched(
                model=model,
                x=x[b:b+1],  # Ground truth for this sample
                z_current=z_sample,  # Current state with block masked
                block_start=block_start,
                block_end=block_end,
                permutations=perms_to_eval,
                attention_mask=attn_sample,
            )  # [num_valid_perms]
            
            # Select best permutation
            best_perm_idx = ll_all_perms.argmax().item()
            best_ll = ll_all_perms[best_perm_idx].item()
            best_perm = perms_to_eval[best_perm_idx]
            
            # Unmask using best permutation (update z_sample in place)
            for step_idx in best_perm:
                pos = block_start + step_idx
                z_sample[0, pos] = x[b, pos]
            
            # Accumulate likelihood
            ll_total[b] += best_ll
    
    return ll_total


def _evaluate_block_permutations_batched(
    model: nn.Module,
    x: Tensor,  # [1, seq_len] ground truth (single sample)
    z_current: Tensor,  # [1, seq_len] current state with block masked
    block_start: int,  # Absolute position where block starts
    block_end: int,  # Absolute position where block ends
    permutations: list,  # List of permutation tuples, e.g., [(0,1,2,3), (0,1,3,2), ...]
    attention_mask: Optional[Tensor] = None,  # [1, seq_len]
) -> Tensor:
    """
    Evaluate all permutations for a single block in parallel.
    
    This function batches across permutations: if there are P permutations,
    we create a batch of size P and evaluate them all simultaneously.
    
    Args:
        x: Ground truth sequence (single sample, batch_size=1)
        z_current: Current state with the block masked
        block_start, block_end: Absolute positions defining the block
        permutations: List of orderings, e.g., [(0,1,2,3), (0,2,1,3), ...]
    
    Returns:
        ll_perms: [num_perms] log-likelihood for each permutation
    """
    device = x.device
    seq_len = x.shape[1]
    num_perms = len(permutations)
    actual_block_size = block_end - block_start
    
    # Replicate current state for each permutation: [num_perms, seq_len]
    z_batch = z_current.expand(num_perms, -1).clone()
    x_batch = x.expand(num_perms, -1)  # Ground truth (no clone needed, read-only)
    
    if attention_mask is not None:
        attn_batch = attention_mask.expand(num_perms, -1)
    else:
        attn_batch = None
    
    # Track log-likelihood for each permutation
    ll_perms = torch.zeros(num_perms, device=device)
    
    # Unmask tokens step-by-step according to each permutation
    for step in range(actual_block_size):
        # Get model predictions for all permutations: [num_perms, seq_len, vocab_size]
        with torch.no_grad():
            logits = model(z_batch, attention_mask=attn_batch).logits
            log_probs = F.log_softmax(logits, dim=-1)  # [num_perms, seq_len, vocab_size]
        
        # For each permutation, unmask the position specified by perm[step]
        for p_idx, perm in enumerate(permutations):
            pos = block_start + perm[step]  # Absolute position to unmask
            true_token = x[0, pos].item()
            
            # Add log prob of true token at this position
            ll_perms[p_idx] += log_probs[p_idx, pos, true_token]
            
            # Unmask this position for subsequent steps
            z_batch[p_idx, pos] = true_token
        
        del logits, log_probs  # Free memory
    
    return ll_perms


def generate_with_strategy(
    model: nn.Module,
    prompt: Tensor,  # [batch_size, prompt_len] 
    gen_length: int,  # Number of tokens to generate
    strategy: object,  # Any strategy with select_positions method
    mask_token_id: int,
    attention_mask: Optional[Tensor] = None,
    block_size: Optional[int] = None,  # For block permutation strategy
) -> Tensor:
    """Generate text using a decoding strategy."""
    
    batch_size, prompt_len = prompt.shape
    device = prompt.device
    seq_len = prompt_len + gen_length
    
    # Initialize: prompt unmasked, generation area masked
    z_k = torch.full((batch_size, seq_len), mask_token_id, dtype=torch.long, device=device)
    z_k[:, :prompt_len] = prompt
    
    if attention_mask is not None:
        gen_mask = torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=device)
        attention_mask_full = torch.cat([attention_mask, gen_mask], dim=1)
    else:
        attention_mask_full = None
    
    # Setup tracking
    prompt_lens = torch.full((batch_size,), prompt_len, dtype=torch.long, device=device)
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    num_unmasked = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # Special handling for BlockPermutationStrategy
    is_block_perm = isinstance(strategy, BlockPermutationStrategy)
    if is_block_perm:
        if block_size is None:
            block_size = strategy.block_size
        strategy.reset_state(batch_size, device)
        
        if gen_length % block_size != 0:
            raise ValueError(
                f"gen_length={gen_length} must be divisible by block_size={block_size}"
            )
        
        num_blocks = gen_length // block_size
    else:
        num_blocks = None
    
    # Generation loop
    if is_block_perm:
        # NEW: Cache for greedy tokens per block
        cached_block_tokens = {}  # block_num -> [batch_size, block_size]
        
        # Block-wise generation with permutation search
        for block_num in range(num_blocks):
            block_idx = torch.full((batch_size,), block_num, dtype=torch.long, device=device)
            
            # Evaluate permutations and cache greedy tokens
            best_perm, best_greedy = _find_best_permutation_for_generation(  # ← NEW: receives both
                model, z_k, block_idx, prompt_lens, block_size, 
                mask_token_id, attention_mask_full
            )
            
            # Cache both permutation and tokens
            strategy.set_block_permutation(block_idx, best_perm)
            cached_block_tokens[block_num] = best_greedy  # ← NEW: Store cached tokens
            
            # NEW: Unmask block using cached tokens (NO forward passes needed!)
            for step in range(block_size):
                positions = strategy.select_positions(
                    None,  # ← logits not needed since we follow cached permutation
                    (z_k == mask_token_id),  # is_maskable
                    num_unmasked=num_unmasked,
                    prompt_lens=prompt_lens
                )  # [batch_size, 1]
                
                # Unmask using cached greedy predictions
                for b in range(batch_size):
                    if positions[b, 0] >= 0:
                        pos = positions[b, 0]
                        z_k[b, pos] = cached_block_tokens[block_num][b, step]  # ← Use cache!
                
                num_unmasked += (positions[:, 0] >= 0).long()
    else:
        # Standard iterative generation (unchanged)
        max_iterations = gen_length + 10
        
        for iteration in range(max_iterations):
            is_active = num_unmasked < gen_length
            if not is_active.any():
                break
            
            is_maskable = (z_k == mask_token_id)
            
            logits = model(z_k, attention_mask=attention_mask_full).logits
            predictions = logits.argmax(dim=-1)
            
            positions = strategy.select_positions(
                logits, is_maskable,
                num_unmasked=num_unmasked,
                prompt_lens=prompt_lens
            )
            
            for b in range(batch_size):
                for k in range(positions.shape[1]):
                    if positions[b, k] >= 0:
                        pos = positions[b, k]
                        z_k[b, pos] = predictions[b, pos]
            
            num_unmasked += (positions >= 0).sum(dim=1)
    
    return z_k


def _find_best_permutation_for_generation(
    model: nn.Module,
    z_k: Tensor,  # [batch_size, seq_len]
    block_idx: Tensor,  # [batch_size]
    prompt_lens: Tensor,  # [batch_size]
    block_size: int,
    mask_token_id: int,
    attention_mask: Optional[Tensor] = None,
    permutation_batch_size: int = 24,  # NEW: Split permutations into chunks
) -> tuple[Tensor, Tensor]:
    """Find best permutation using chunked permutation batching."""
    
    batch_size, seq_len = z_k.shape
    device = z_k.device
    
    all_perms = list(itertools.permutations(range(block_size)))
    num_perms = len(all_perms)
    
    # Block boundaries
    block_start = prompt_lens + block_idx * block_size
    
    # NEW: Process permutations in chunks
    num_chunks = (num_perms + permutation_batch_size - 1) // permutation_batch_size
    
    # Storage for all permutation results
    all_confidence = []
    all_greedy_tokens = []
    
    for chunk_idx in range(num_chunks):
        start_perm = chunk_idx * permutation_batch_size
        end_perm = min(start_perm + permutation_batch_size, num_perms)
        chunk_perms = all_perms[start_perm:end_perm]
        chunk_size = len(chunk_perms)
        
        # Expand for this chunk: [batch_size * chunk_size, seq_len]
        z_expanded = z_k.unsqueeze(1).repeat(1, chunk_size, 1).reshape(
            batch_size * chunk_size, seq_len
        )
        
        if attention_mask is not None:
            attn_expanded = attention_mask.unsqueeze(1).repeat(
                1, chunk_size, 1
            ).reshape(batch_size * chunk_size, seq_len)
        else:
            attn_expanded = None
        
        block_start_expanded = block_start.unsqueeze(1).repeat(
            1, chunk_size
        ).reshape(batch_size * chunk_size)
        
        # Track confidence and cache for this chunk
        confidence_chunk = torch.zeros(batch_size * chunk_size, device=device)
        greedy_tokens_chunk = torch.zeros(
            batch_size * chunk_size, block_size,
            dtype=torch.long, device=device
        )
        
        # Simulate unmasking for this chunk of permutations
        for step in range(block_size):
            positions_in_block = torch.tensor(
                [chunk_perms[p % chunk_size][step] 
                 for p in range(batch_size * chunk_size)],
                dtype=torch.long, device=device
            )
            abs_positions = block_start_expanded + positions_in_block
            
            # Forward pass (smaller batch now!)
            with torch.no_grad():
                logits = model(z_expanded, attention_mask=attn_expanded).logits
                log_probs = F.log_softmax(logits, dim=-1)
            
            # Greedy prediction
            batch_indices = torch.arange(batch_size * chunk_size, device=device)
            greedy_tokens = logits[batch_indices, abs_positions].argmax(dim=-1)
            greedy_log_probs = log_probs[batch_indices, abs_positions, greedy_tokens]
            
            # Cache and accumulate
            greedy_tokens_chunk[:, step] = greedy_tokens
            confidence_chunk += greedy_log_probs
            z_expanded[batch_indices, abs_positions] = greedy_tokens
        
        # Store chunk results
        all_confidence.append(confidence_chunk.reshape(batch_size, chunk_size))
        all_greedy_tokens.append(greedy_tokens_chunk.reshape(
            batch_size, chunk_size, block_size
        ))
    
    # Concatenate all chunks: [batch_size, num_perms]
    confidence_per_perm = torch.cat(all_confidence, dim=1)
    greedy_tokens_all = torch.cat(all_greedy_tokens, dim=1)
    
    # Select best across all permutations
    best_perm_idx = confidence_per_perm.argmax(dim=1)  # [batch_size]
    best_permutation = torch.tensor(
        [all_perms[idx.item()] for idx in best_perm_idx],
        dtype=torch.long, device=device
    )
    
    # Extract best greedy tokens
    best_greedy_tokens = torch.zeros(
        batch_size, block_size, dtype=torch.long, device=device
    )
    for b in range(batch_size):
        best_greedy_tokens[b] = greedy_tokens_all[b, best_perm_idx[b]]
    
    return best_permutation, best_greedy_tokens


# ============================================================================
# Helper Functions
# ============================================================================

def _get_valid_answer_mask(
    batch_size: int,
    seq_len: int,
    prompt_lens: Tensor,  # [batch_size]
    lengths: Tensor,  # [batch_size]
    device: torch.device
) -> Tensor:
    """
    Boolean mask: True for positions in [prompt_len, length).
    
    Returns: [batch_size, seq_len]
    """
    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
    
    is_valid = (
        (positions >= prompt_lens.unsqueeze(1)) &
        (positions < lengths.unsqueeze(1))
    )
    
    return is_valid


def _compute_ll_for_positions(
    logits: Tensor,  # [batch_size, seq_len, vocab_size]
    x: Tensor,  # [batch_size, seq_len] ground truth
    positions: Tensor,  # [batch_size, max_k] positions to evaluate (-1 = padding)
    is_active: Tensor,  # [batch_size] which samples to compute for
) -> Tensor:
    """
    Compute LL contribution for specified positions.
    
    Returns: [batch_size] NLL (0 for inactive samples)
    
    Convention: Use 0.0 to zero out contributions in sum (not -inf)
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device
    
    # Get log probs
    log_probs = F.log_softmax(logits, dim=-1)  # [B, L, V]
    
    # Extract log probs for ground truth tokens
    true_log_probs = torch.gather(
        log_probs, dim=2, index=x.unsqueeze(-1)
    ).squeeze(-1)  # [B, L]
    
    # Sum log probs at specified positions
    ll = torch.zeros(batch_size, device=device)
    
    for b in range(batch_size):
        if is_active[b]:
            valid_pos = positions[b][positions[b] >= 0]
            if len(valid_pos) > 0:
                ll[b] = true_log_probs[b, valid_pos].sum() # Note: Use sum to accumulate log probs
    
    return ll


def _unmask_positions(
    z_k: Tensor,  # [batch_size, seq_len] current state
    x: Tensor,  # [batch_size, seq_len] ground truth
    positions: Tensor,  # [batch_size, max_k] positions to unmask (-1 = padding)
) -> Tensor:
    """Unmask specified positions using ground truth values."""
    z_k = z_k.clone()
    batch_size = z_k.shape[0]
    
    for b in range(batch_size):
        valid_pos = positions[b][positions[b] >= 0]
        if len(valid_pos) > 0:
            z_k[b, valid_pos] = x[b, valid_pos]
    
    return z_k