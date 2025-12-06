"""
Selection strategies for exact likelihood computation.
"""
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
