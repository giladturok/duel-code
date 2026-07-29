"""
Selection strategies for exact likelihood computation.
"""
import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor


@dataclass
class GreedyConfidenceStrategy:
    """
    Selects the top-k maskable tokens with the highest confidence (max log-prob).
    """

    k: int = 1

    def select_positions(
        self,
        log_probs: Tensor,
        is_maskable: Tensor,
        k: Optional[Union[int, Tensor]] = None,
        num_unmasked: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        del num_unmasked  # Greedy strategy does not rely on mask counts.
        per_sample_k = _normalize_k(k, self.k, log_probs.size(0), log_probs.device)
        confidence = _masked_confidence(log_probs, is_maskable)
        return _select_topk_per_sample(confidence, per_sample_k)


@dataclass
class BlockGreedyConfidenceStrategy:
    """
    Greedy selection restricted to the current block determined by num_unmasked.
    """

    block_size: int
    k: int = 1

    def select_positions(
        self,
        log_probs: Tensor,
        is_maskable: Tensor,
        k: Optional[Union[int, Tensor]] = None,
        num_unmasked: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        if num_unmasked is None or lengths is None:
            raise ValueError("BlockGreedy requires num_unmasked and lengths tensors.")

        batch_size, seq_len = log_probs.shape[:2]
        device = log_probs.device

        block_idx = num_unmasked // self.block_size # [B]
        block_start = block_idx * self.block_size # [B]
        block_end = torch.minimum(block_start + self.block_size, lengths) # [B]

        positions = torch.arange(seq_len, device=device)
        is_in_block = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
        )

        maskable_in_block = is_maskable & is_in_block
        confidence = _masked_confidence(log_probs, maskable_in_block)
        per_sample_k = _normalize_k(k, self.k, batch_size, device)
        return _select_topk_per_sample(confidence, per_sample_k)
    
@dataclass
class LeftToRightStrategy:
    """
    Selects the next k maskable tokens in left-to-right order.
    """

    k: int = 1

    def select_positions(
        self,
        log_probs: Tensor,
        is_maskable: Tensor,
        k: Optional[Union[int, Tensor]] = None,
        num_unmasked: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        del num_unmasked  # Left-to-right strategy only relies on is_maskable.
        batch_size, seq_len = log_probs.shape[:2]
        device = log_probs.device
        per_sample_k = _normalize_k(k, self.k, batch_size, device)

        # Create position indices: [B, S]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Priority score: lower position = higher priority (leftmost first)
        # Maskable positions get their position index, non-maskable get inf
        priority = torch.where(is_maskable, positions.float(), float("inf"))

        # Select k positions with lowest priority (use negative to leverage topk)
        return _select_topk_per_sample(-priority, per_sample_k)
    

@dataclass
class BlockLeftToRightStrategy:
    """
    Left-to-right selection restricted to the current block determined by num_unmasked.
    """

    block_size: int
    k: int = 1

    def select_positions(
        self,
        log_probs: Tensor,
        is_maskable: Tensor,
        k: Optional[Union[int, Tensor]] = None,
        num_unmasked: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        if num_unmasked is None or lengths is None:
            raise ValueError("BlockLeftToRight requires num_unmasked and lengths tensors.")

        batch_size, seq_len = log_probs.shape[:2]
        device = log_probs.device

        # Determine current block bounds
        block_idx = num_unmasked // self.block_size  # [B]
        block_start = block_idx * self.block_size  # [B]
        block_end = torch.minimum(block_start + self.block_size, lengths)  # [B]

        # Create mask for positions in current block
        positions = torch.arange(seq_len, device=device)
        is_in_block = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
        )

        # Only consider maskable positions within the block
        maskable_in_block = is_maskable & is_in_block

        # Priority: lower position = higher priority (leftmost first)
        positions_expanded = positions.unsqueeze(0).expand(batch_size, -1)
        priority = torch.where(maskable_in_block, positions_expanded.float(), float("inf"))

        # Select k positions with lowest priority (use negative to leverage topk)
        per_sample_k = _normalize_k(k, self.k, batch_size, device)
        return _select_topk_per_sample(-priority, per_sample_k)


@dataclass
class ProbabilityMarginStrategy:
    """
    Selects the top-k maskable tokens with the highest probability margin.
    
    Margin is defined as the absolute difference between the top two probabilities
    over the vocabulary: |p(x_i = j1|x_t) - p(x_i = j2|x_t)| where j1 and j2 are
    the two most probable values.
    
    Higher margin indicates higher certainty (one value dominates).
    """

    k: int = 1

    def select_positions(
        self,
        log_probs: Tensor,
        is_maskable: Tensor,
        k: Optional[Union[int, Tensor]] = None,
        num_unmasked: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        del num_unmasked  # Probability margin strategy does not rely on mask counts.
        per_sample_k = _normalize_k(k, self.k, log_probs.size(0), log_probs.device)
        
        margin = _compute_probability_margin(log_probs)  # [B, L]
        
        # Mask out non-maskable positions
        margin = margin.masked_fill(~is_maskable, float("-inf"))
        
        return _select_topk_per_sample(margin, per_sample_k)


@dataclass
class BlockProbabilityMarginStrategy:
    """
    Probability margin selection restricted to the current block determined by num_unmasked.
    """

    block_size: int
    k: int = 1

    def select_positions(
        self,
        log_probs: Tensor,
        is_maskable: Tensor,
        k: Optional[Union[int, Tensor]] = None,
        num_unmasked: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        if num_unmasked is None or lengths is None:
            raise ValueError("BlockProbabilityMargin requires num_unmasked and lengths tensors.")

        batch_size, seq_len = log_probs.shape[:2]
        device = log_probs.device

        # Determine current block bounds
        block_idx = num_unmasked // self.block_size  # [B]
        block_start = block_idx * self.block_size  # [B]
        block_end = torch.minimum(block_start + self.block_size, lengths)  # [B]

        # Create mask for positions in current block
        positions = torch.arange(seq_len, device=device)
        is_in_block = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
        )

        # Only consider maskable positions within the block
        maskable_in_block = is_maskable & is_in_block

        margin = _compute_probability_margin(log_probs)  # [B, L]

        # Mask out positions outside block or not maskable
        margin = margin.masked_fill(~maskable_in_block, float("-inf"))

        per_sample_k = _normalize_k(k, self.k, batch_size, device)
        return _select_topk_per_sample(margin, per_sample_k)

    
@dataclass
class ConfidenceThresholdStrategy:
    """
    Unmasks all tokens with max probability >= k (used as threshold).
    Falls back to single most confident token if none meet threshold.

    From Fast-dLLM: "Only those with confidence exceeding a threshold are
    unmasked in the current step; the rest remain masked and are reconsidered
    in future steps. If no token's confidence exceeds the threshold, we always
    unmask the token with the highest confidence to ensure progress."

    Note: k is used as the threshold value (e.g., k=0.9 means 90% confidence threshold)
    to maintain API consistency with other strategies.
    """

    k: float = 0.9  # Minimum probability threshold in (0, 1]

    def select_positions(
        self,
        log_probs: Tensor,
        is_maskable: Tensor,
        num_unmasked: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        del num_unmasked  # Not used by threshold strategy

        # Compute confidence scores (returns -inf for non-maskable)
        log_confidence = log_probs.max(dim=-1).values  # [B, L]
        log_confidence_masked = log_confidence.masked_fill(~is_maskable, float("-inf"))

        # Compare in log-space for numerical stability
        # c_i >= τ ⟺ log(c_i) >= log(τ)
        log_threshold = math.log(self.k)
        meets_threshold = (log_confidence_masked > float("-inf")) & (log_confidence_masked >= log_threshold)

        # Count how many positions meet threshold per sample
        k_per_sample = meets_threshold.sum(dim=1)  # [B]

        # Fallback: if no positions meet threshold, select most confident (k=1)
        k_per_sample = torch.clamp(k_per_sample, min=1)

        # Select top-k positions by confidence
        assert not torch.isnan(log_confidence_masked).any(), "NaNs in log_confidence_masked"
        return _select_topk_per_sample(log_confidence_masked, k_per_sample)


@dataclass
class BlockPermutationStrategy:
    """
    Oracle: exhaustive search over all block_size! permutations per block.

    Marker class. Dispatches diffusion._compute_exact_ll to the permutation
    code path (compute_exact_loglikelihood_cached_permutations), which
    enumerates permutations via itertools.permutations and selects positions
    inline. select_positions is not used.
    """
    block_size: int
    k: int = 1

    def select_positions(self, *args, **kwargs):
        raise NotImplementedError(
            "BlockPermutationStrategy is a marker; selection is performed "
            "inside compute_exact_loglikelihood_cached_permutations."
        )


@dataclass
class BlockSubsetDPStrategy:
    """
    Oracle via subset-lattice DP: identical answer to BlockPermutationStrategy
    in 2**block_size - 1 forwards per block instead of block_size! * block_size.

    Marker class -- deliberately NOT a subclass of BlockPermutationStrategy, so
    the isinstance dispatch in diffusion._compute_exact_ll keeps the two paths
    disjoint. select_positions is not used.
    """
    block_size: int
    k: int = 1

    def select_positions(self, *args, **kwargs):
        raise NotImplementedError(
            "BlockSubsetDPStrategy is a marker; selection is performed inside "
            "compute_exact_loglikelihood_cached_subset_dp."
        )


@dataclass
class BlockConfidenceThresholdStrategy:
    """
    Confidence threshold selection restricted to current block.

    Note: k is used as the threshold value (e.g., k=0.9 means 90% confidence threshold)
    to maintain API consistency with other strategies.
    """

    block_size: int
    k: float = 0.9  # Minimum probability threshold in (0, 1]

    def select_positions(
        self,
        log_probs: Tensor,
        is_maskable: Tensor,
        num_unmasked: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        
        if num_unmasked is None or lengths is None:
            raise ValueError("BlockConfidenceThreshold requires num_unmasked and lengths.")

        batch_size, seq_len = log_probs.shape[:2]
        device = log_probs.device

        # Determine current block bounds by enforcing synchronized blocks
        per_sample_block_idx = num_unmasked // self.block_size  # [B]
        cur_block = per_sample_block_idx.min().item() # []
        block_idx = torch.full_like(per_sample_block_idx, cur_block)  # [B]
        block_start = block_idx * self.block_size  # [B]
        block_end = torch.minimum(block_start + self.block_size, lengths)  # [B]

        # Mask for positions in current block
        positions = torch.arange(seq_len, device=device)
        is_in_block = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
        )

        # Only consider maskable positions within the block
        maskable_in_block = is_maskable & is_in_block

        # Compute confidence as max log-probability over vocabulary
        log_confidence = log_probs.max(dim=-1).values  # [B, L]
        log_confidence_masked = log_confidence.masked_fill(~maskable_in_block, float("-inf"))
        
        # Mask out non-maskable positions
        log_threshold = torch.log(torch.tensor(self.k, device=log_probs.device)) # Compare in log-space for numerical stability
        meets_threshold = (log_confidence_masked > float("-inf")) & (log_confidence_masked >= log_threshold)
        k_per_sample = meets_threshold.sum(dim=1)  # [B]

        # Fallback: if no positions meet threshold in block, select most confident
        has_maskable_in_block = maskable_in_block.any(dim=1) # [B]
        k_per_sample = torch.where(
            has_maskable_in_block, torch.clamp(k_per_sample, min=1), k_per_sample
        )
        
        # assert no nans in log_confidence_masked
        assert not torch.isnan(log_confidence_masked).any(), "NaNs in log_confidence_masked"
        return _select_topk_per_sample(log_confidence_masked, k_per_sample)


def _masked_confidence(log_probs: Tensor, mask: Tensor) -> Tensor:
    """Return per-position confidence masked with -inf outside the allowed region."""
    confidence = log_probs.max(dim=-1).values
    return confidence.masked_fill(~mask, float("-inf"))


def _normalize_k(
    k: Optional[Union[int, Tensor]],
    default_k: Optional[int],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """
    Convert the caller-provided k into a per-sample tensor on the target device.
    """
    chosen_k = default_k if k is None else k
    if chosen_k is None:
        raise ValueError("k must be provided either at init or call time.")
    if isinstance(chosen_k, int):
        if chosen_k < 0:
            raise ValueError("k must be non-negative.")
        return torch.full((batch_size,), chosen_k, device=device, dtype=torch.long)
    return chosen_k.to(device=device, dtype=torch.long)


def _select_topk_per_sample(scores: Tensor, k_per_sample: Tensor) -> Tensor:
    """
    Select the top-k indices per sample, padding unused slots with -1.
    """
    device = scores.device
    batch_size = scores.size(0)
    k_per_sample = k_per_sample.to(device=device, dtype=torch.long)
    max_k = int(k_per_sample.max().item()) if k_per_sample.numel() > 0 else 0

    if max_k == 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=device)

    topk_values, topk_indices = torch.topk(scores, k=max_k, dim=1, sorted=True)

    score_mask = topk_values > float("-inf")
    k_mask = torch.arange(max_k, device=device).unsqueeze(0) < k_per_sample.unsqueeze(1)
    valid_mask = score_mask & k_mask

    padding_value = torch.full_like(topk_indices, -1)
    positions = torch.where(valid_mask, topk_indices, padding_value)
    return positions


def _compute_probability_margin(log_probs: Tensor) -> Tensor:
    """
    Compute probability margin for each position in a numerically stable way.
    
    Margin is defined as the absolute difference between the top two probabilities:
    margin_i = |p(x_i = j1|x_t) - p(x_i = j2|x_t)|
    
    Uses expm1 to avoid catastrophic cancellation when p1 ≈ p2.
    
    Args:
        log_probs: [B, L, V] log-probabilities over vocabulary
        
    Returns:
        margin: [B, L] probability margin for each position
    """
    # Get top-2 log-probabilities for each position across vocabulary
    top2_log_probs, _ = torch.topk(log_probs, k=2, dim=-1, sorted=True)  # [B, L, 2]
    
    # Compute margin in a numerically stable way
    # margin = p1 - p2 = exp(l1) - exp(l2) = -exp(l1) * expm1(l2 - l1)
    l1 = top2_log_probs[..., 0]
    l2 = top2_log_probs[..., 1]
    margin = -torch.exp(l1) * torch.expm1(l2 - l1)  # [B, L]
    
    return margin


def _compute_confidence_threshold_mask(
    log_probs: Tensor, 
    threshold: float, 
    is_maskable: Tensor
) -> Tensor:
    """
    Compute confidence scores and identify positions meeting threshold.
    
    Confidence is defined as max probability over vocabulary: c_i = max_j p(x_i=j|·)
    
    Args:
        log_probs: [B, L, V] log-probabilities over vocabulary
        threshold: minimum probability threshold in (0, 1]
        is_maskable: [B, L] boolean mask of maskable positions
    
    Returns:
        confidence: [B, L] max log-probability (for ranking), -inf for non-maskable
    """
    # Compute confidence as max log-probability over vocabulary
    log_confidence = log_probs.max(dim=-1).values  # [B, L]
    
    # Mask out non-maskable positions
    confidence = log_confidence.masked_fill(~is_maskable, float("-inf"))
    
    return confidence
