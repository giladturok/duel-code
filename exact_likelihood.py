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
    valid_mask: Optional[Tensor] = None,  # [B, L] bool
    lengths_override: Optional[Tensor] = None,  # [B]
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

    # Optional explicit control over *which* positions are scored, decoupled
    # from the "first `lengths` positions" prefix convention. `lengths` is what
    # the strategy uses to locate the current block and what `active` compares
    # against, so it stays the *coordinate* extent; `valid_mask` says which of
    # those positions are masked-and-scored. Positions inside `lengths` that are
    # not in `valid_mask` are given as context and counted as already unmasked,
    # which keeps `num_unmasked // block_size` (the strategy's block pointer)
    # aligned with the generation-time sampler.
    if lengths_override is not None:
        lengths = lengths_override.to(device=device, dtype=torch.long)
    if valid_mask is not None:
        is_valid = valid_mask.to(device=device, dtype=torch.bool)

    # Initialize: all valid positions masked
    mask_fill = x0.new_full(x0.shape, mask_token_id)
    z_k = torch.where(is_valid, mask_fill, x0)  # [B, L]

    num_unmasked = (lengths - is_valid.sum(dim=1)).clamp(min=0).to(torch.long)  # [B]
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
    return_per_order: bool = False,
):  # -> (ll_total: [B], steps: int) or (ll_total, steps, extras)
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
        return_per_order: If True, additionally return the uniform-order and
            normalized-mixture reductions of the same per-permutation values.

    Cache mechanism:
    - For block b, model attends to:
      * Cached blocks 0..b-1 (fully unmasked, KV values stored)
      * Current block b (partially unmasked, computed on-the-fly)
    - After block b is fully unmasked, commit it to cache

    Why three reductions are free: every permutation ends the block fully
    revealed to ground truth, so the post-block state `z_k` is identical across
    permutations. Block contributions are therefore independent and *all* of
    oracle / uniform-order / mixture decompose as a sum over blocks of a
    reduction of the same [B, n_perms] table.
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
    # Running totals for the two extra reductions (same forwards, same tokens).
    ll_total_uniform = torch.zeros(batch_size, device=device)  # [B]
    ll_total_mixture = torch.zeros(batch_size, device=device)  # [B]

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

        # Per-permutation block log-likelihoods, kept for the extra reductions.
        # NOTE: this list is indexed by permutation of *positions*, not of
        # orderable positions. When a block has fewer than `actual_block_size`
        # valid positions (see the `ignore_bos` off-by-one), each distinct order
        # of the valid subset appears the same number of times, so a uniform
        # mean / logsumexp over this list is still a uniform mean / logsumexp
        # over the distinct orders.
        ll_perms_list = []

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

            ll_perms_list.append(ll_block_perm)

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

        if return_per_order:
            ll_perms = torch.stack(ll_perms_list, dim=1)  # [B, n_perms]
            n_perms = ll_perms.shape[1]
            # Oracle == max, and must agree with the incumbent-tracking above.
            ll_total_uniform += ll_perms.mean(dim=1)
            ll_total_mixture += (
                torch.logsumexp(ll_perms, dim=1) - math.log(n_perms)
            )
        del ll_perms_list

        # NFE accounting: each permutation does `actual_block_size` forwards
        # (one per position via _loop_fn), and there are actual_block_size!
        # permutations evaluated per block.
        steps += math.factorial(actual_block_size) * actual_block_size

        # Commit block to cache (advances cache_idx for next block)
        model_forward_fn(z_k[:, block_slice], commit=True)

    if return_per_order:
        return ll_total, steps, {
            'll_oracle': ll_total,
            'll_uniform_order': ll_total_uniform,
            'll_mixture': ll_total_mixture,
        }
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


# One-time warning flag for ragged batches in the subset-lattice DP.
_SUBSET_DP_RAGGED_WARNED = False


def compute_exact_loglikelihood_cached_subset_dp(
    x0: Tensor,  # [B, L]
    model_forward_fn,  # Callable: (x_block, commit=False) -> logits_block
    mask_token_id: int,
    strategy: Optional[object] = None,
    attention_mask: Optional[Tensor] = None,  # [B, L]
    block_size: Optional[int] = None,
    return_per_order: bool = False,
):  # -> (ll_total: [B], steps: int) or (ll_total, steps, extras)
    """
    Exact block-wise LL via a subset-lattice dynamic program.

    Drop-in replacement for `compute_exact_loglikelihood_cached_permutations`:
    same signature, same returns, same three reductions (oracle / uniform-order /
    mixture), but 2**m - 1 forwards per block instead of m! * m.

    Why the DP is exact
    -------------------
    For the SUBS parameterization the backbone receives sigma = zeros, so a
    forward pass is a pure function of the token input. The block input is
    determined entirely by *which* positions hold ground truth and which hold
    MASK -- not by the order in which they were revealed. Hence

        w(S, j) = log p(x0_j | z(S)),   z(S) = block with S (plus the invalid /
                                        padding positions) set to x0, rest MASK

    depends only on the revealed *set* S. A permutation's block log-likelihood
    is the sum of w along a maximal chain 0 -> ... -> P_b of the subset lattice,
    so all m! chains share their prefixes and one forward at state S yields
    w(S, j) for every j not in S simultaneously.

    Three reductions over the same lattice, all computed in one sweep:
      * ORACLE   V(0)=0,  V(S|{j}) = max(V(S|{j}), V(S) + w(S,j));  read V(P_b)
      * MIXTURE  M(0)=0,  M(S|{j}) = logaddexp(M(S|{j}), M(S) + w(S,j));
                 read M(P_b) - log(m!)
      * UNIFORM  no path DP: edge (S, j) is traversed by a uniformly random
                 order with probability 1 / ((m-k) * C(m,k)) where k = |S|, so
                 the expectation is a flat weighted sum over all edges.

    States are visited as integers 0, 1, ..., 2**K - 1, which is automatically a
    valid topological order (S \\ {j} < S numerically), so V[:, S] / M[:, S] are
    final when S is dequeued and all K outgoing edges can be relaxed at once
    (PUSH). No w table is ever materialized.

    The oracle result should be *bitwise* identical to the permutation path:
    fp32 addition is monotone under round-to-nearest, and both accumulate the
    same per-position log-probs left-to-right along the same maximizing chain.

    Args:
        x0: True sequence [B, L]
        model_forward_fn: Callable(x_block, commit=False) -> logits_block
            - Takes current block [B, block_size]
            - Returns logits for that block [B, block_size, V]
            - commit=True advances the cache pointer
        mask_token_id: Token ID for masked positions
        strategy: Marker only (BlockSubsetDPStrategy); unused.
        attention_mask: Valid token mask [B, L]
        block_size: Block size for processing; must divide seq_len.
        return_per_order: If True, additionally return the uniform-order and
            normalized-mixture reductions.

    NFE accounting:
        `steps` counts executed commit=False forwards only (commits excluded).
        This is NOT comparable to the permutation path's `K! * K` convention --
        it is the whole point of this function that the count is far smaller.

    Ragged batches (per-example valid sets differing within a batch) are handled
    correctly but cost extra forwards; a one-time warning is emitted.
    """
    global _SUBSET_DP_RAGGED_WARNED

    if block_size is None:
        raise ValueError("block_size must be provided for cached LL computation.")

    batch_size, seq_len = x0.shape
    device = x0.device
    steps = 0

    assert seq_len % block_size == 0, (
        f"subset-DP requires block_size to divide seq_len; got seq_len={seq_len}, "
        f"block_size={block_size}. The KV-cache write in the backbone is a "
        f"fixed-width slice and would shape-error on a short final block."
    )
    K = block_size
    n_states = 1 << K
    neg_inf = float('-inf')

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
    ll_total_uniform = torch.zeros(batch_size, device=device)  # [B]
    ll_total_mixture = torch.zeros(batch_size, device=device)  # [B]

    # ---- Lattice tables, built once on device ----
    arange_K = torch.arange(K, device=device)  # [K]
    bits_table = (
        (torch.arange(n_states, device=device).unsqueeze(1) >> arange_K.unsqueeze(0)) & 1
    ).bool()  # [2**K, K]
    popc_table = bits_table.sum(dim=1)  # [2**K] int64
    pow2 = 2 ** arange_K  # [K] int64

    # coef[m, k] = 1 / ((m - k) * C(m, k)) = probability that a uniformly random
    # order of m items traverses a given edge out of a given level-k subset.
    coef = torch.zeros((K + 1, K + 1), dtype=torch.float64)
    for m_ in range(K + 1):
        for k_ in range(K + 1):
            if k_ < m_:
                coef[m_, k_] = 1.0 / ((m_ - k_) * math.comb(m_, k_))
    coef_table = coef.float().to(device)  # [K+1, K+1]

    logfact_table = torch.tensor(
        [math.lgamma(i + 1) for i in range(K + 1)], dtype=torch.float64
    ).float().to(device)  # [K+1], log(m!)

    # Process each block sequentially
    num_blocks = (seq_len + block_size - 1) // block_size
    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, seq_len)
        block_slice = slice(block_start, block_end)

        # Skip if no valid tokens in this block.
        # NOTE: this replicates the reference quirk (lines 256-257 of the
        # permutation version): the commit is skipped too, leaving cache_idx
        # unadvanced. Deliberate -- the two paths must agree exactly.
        block_valid = is_valid[:, block_slice]  # [B, K]
        if not block_valid.any().item():
            continue

        x0_blk = x0[:, block_slice]  # [B, K]
        mask_blk = x0_blk.new_full(x0_blk.shape, mask_token_id)  # [B, K]
        m = block_valid.sum(dim=1)  # [B] int64, |P_b| per example

        valid_bits = (block_valid.long() * pow2).sum(dim=1)  # [B] int64 bitmask
        valid_bits_list = valid_bits.tolist()
        union_bits = 0
        for v in valid_bits_list:
            union_bits |= int(v)

        if not _SUBSET_DP_RAGGED_WARNED and len(set(valid_bits_list)) > 1:
            _SUBSET_DP_RAGGED_WARNED = True
            warnings.warn(
                "subset-DP: valid-position mask is not constant across the batch. "
                "Results are still exact, but the lattice sweep visits the union "
                "of per-example subsets and therefore costs extra forwards.",
                stacklevel=2,
            )

        V = torch.full((batch_size, n_states), neg_inf, device=device)  # oracle
        V[:, 0] = 0.0
        M = torch.full((batch_size, n_states), neg_inf, device=device)  # mixture
        M[:, 0] = 0.0
        level_acc = torch.zeros(batch_size, K, device=device)  # uniform-order

        for S in range(n_states):
            if S & ~union_bits:
                continue  # unreachable for every example
            subset_ok = (S & ~valid_bits) == 0  # [B] bool: S subset-of P_b
            if not (subset_ok & (valid_bits != S)).any():
                continue  # S is terminal (or unreachable) for every example

            s_bits = bits_table[S]  # [K] bool
            # Invalid positions are permanent ground-truth context.
            reveal = s_bits.unsqueeze(0) | (~block_valid)  # [B, K]
            z_blk = torch.where(reveal, x0_blk, mask_blk)  # [B, K]

            logits = model_forward_fn(z_blk, commit=False)  # [B, K, V]
            # MUST be F.log_softmax (not _log_softmax_inplace, which mutates its
            # input) to match the reference's arithmetic bit-for-bit.
            lp = F.log_softmax(logits, dim=-1)  # [B, K, V]
            true_lp = lp.gather(2, x0_blk.unsqueeze(-1)).squeeze(-1).float()  # [B, K]
            steps += 1

            edge_ok = block_valid & (~s_bits).unsqueeze(0)  # [B, K]: j in P_b \ S
            k = int(popc_table[S])

            # UNIFORM: every edge out of a level-k subset carries coef[m, k].
            rows = torch.where(
                edge_ok, true_lp, torch.zeros_like(true_lp)
            ).sum(dim=1)  # [B]
            level_acc[:, k] += coef_table[m, k] * subset_ok.float() * rows

            # ORACLE / MIXTURE: push along all outgoing edges. -inf sentinels are
            # safe here: only maximum, logaddexp and finite + (-inf) occur, and
            # torch.logaddexp(-inf, -inf) == -inf without NaN.
            w = torch.where(
                edge_ok, true_lp, torch.full_like(true_lp, neg_inf)
            )  # [B, K]
            for j in range(K):
                if (S >> j) & 1 or not ((union_bits >> j) & 1):
                    continue
                T = S | (1 << j)
                V[:, T] = torch.maximum(V[:, T], V[:, S] + w[:, j])
                M[:, T] = torch.logaddexp(M[:, T], M[:, S] + w[:, j])

        # Read the answers at the per-example full set, which is always reached.
        gi = valid_bits.unsqueeze(1)  # [B, 1]
        ll_total += V.gather(1, gi).squeeze(1)
        ll_total_mixture += M.gather(1, gi).squeeze(1) - logfact_table[m]
        ll_total_uniform += level_acc.sum(dim=1)

        # Every order ends with the block fully revealed, so the post-block state
        # is order-independent.
        z_k = z_k.clone()
        z_k[:, block_slice] = x0_blk
        num_unmasked += m

        # Commit block to cache (advances cache_idx for next block)
        model_forward_fn(z_k[:, block_slice], commit=True)

    if return_per_order:
        return ll_total, steps, {
            'll_oracle': ll_total,
            'll_uniform_order': ll_total_uniform,
            'll_mixture': ll_total_mixture,
        }
    return ll_total, steps


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