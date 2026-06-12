"""
Deterministic and stochastic NLL computation for masked diffusion models.

Key design principles:
1. Separate deterministic vs stochastic (NELBO) computation paths
2. Strategies return positions to unmask (variable k handled naturally)
3. Use is_valid_answer mask to exclude prompt/padding from NLL
4. Log-scale convention: -inf to exclude from argmax, 0.0 to exclude from sum
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Decoding Strategies (no protocol needed - just implement select_positions)
# ============================================================================

class GreedyConfidenceStrategy:
    """Unmask position(s) with highest model confidence."""
    
    def __init__(self, k: int = 1):
        """
        Args:
            k: Number of positions to unmask per step (can be overridden per call)
        """
        self.k = k
    
    def select_positions(
        self,
        logits: Tensor,  # [batch_size, seq_len, vocab_size] - raw model output
        is_maskable: Tensor,  # [batch_size, seq_len] - which positions can be selected
        k: Optional[Tensor] = None,  # [batch_size] - override k per sample
        num_unmasked: Optional[Tensor] = None,
        prompt_lens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Returns: positions [batch_size, max_k] with -1 padding
        
        Note: Use -inf to mask out positions on log scale (for argmax/topk)
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Compute confidence (max log prob at each position)
        log_probs = F.log_softmax(logits, dim=-1)  # [B, L, V]
        confidence = log_probs.max(dim=-1).values  # [B, L]
              
        # Mask out non-maskable positions with -inf (log scale)
        confidence = torch.where(is_maskable, confidence, torch.full_like(confidence, float('-inf')))
        
        # Select top-k per sample
        if k is None:
            k = torch.full((batch_size,), self.k, device=logits.device)
        
        return _select_topk_per_sample(confidence, k)

class ProbabilityMarginStrategy:
    """Unmask position(s) with highest margin between top-2 predictions."""
    
    def __init__(self, k: int = 1):
        self.k = k
    
    def select_positions(self, logits, is_maskable, k=None, num_unmasked=None, prompt_lens=None):
        batch_size = logits.shape[0]
        
        log_probs = F.log_softmax(logits, dim=-1)
        top2 = torch.topk(log_probs, k=2, dim=-1).values  # [B, L, 2]
        margin = top2[..., 0] - top2[..., 1]  # [B, L]
        
        margin = torch.where(
            is_maskable,
            margin,
            torch.tensor(float('-inf'), device=margin.device)
        )
        
        if k is None:
            k = torch.full((batch_size,), self.k, device=logits.device)
        
        return _select_topk_per_sample(margin, k)


class ConfidenceThresholdStrategy:
    """Unmask ALL positions above confidence threshold (variable k per sample)."""
    
    def __init__(self, threshold: float = 0.8, max_k: int = 10):
        """
        Args:
            threshold: Minimum confidence (probability) to unmask
            max_k: Maximum tokens to unmask per step (prevents runaway)
        """
        self.threshold = threshold
        self.max_k = max_k
    
    def select_positions(self, logits, is_maskable, k=None, num_unmasked=None, prompt_lens=None):
        """
        Note: k parameter is ignored - we select all positions above threshold
        (up to max_k)
        """
        batch_size, seq_len = logits.shape[:2]
        
        # Get max probability at each position (NOT log prob for threshold)
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values  # [B, L]
        
        # Positions above threshold AND maskable
        is_confident = (max_probs > self.threshold) & is_maskable
        
        # Use confidence as scores for top-k selection
        scores = torch.where(
            is_confident,
            max_probs,
            torch.tensor(float('-inf'), device=max_probs.device)
        )
        
        # Select up to max_k positions per sample
        k_per_sample = torch.full((batch_size,), self.max_k, device=logits.device)
        return _select_topk_per_sample(scores, k_per_sample)


class BlockGreedyConfidenceStrategy:
    """Greedy unmasking within fixed-size blocks."""
    
    def __init__(self, block_size: int, k: int = 1):
        """
        Args:
            block_size: Size of each block (in answer tokens, not total seq)
            k: Tokens to unmask per step within block
        """
        self.block_size = block_size
        self.k = k
    
    def select_positions(
        self, 
        logits, 
        is_maskable, 
        k=None,
        num_unmasked: Optional[Tensor] = None,  # [batch_size] - for block tracking
        prompt_lens: Optional[Tensor] = None,   # [batch_size]
    ):
        """
        Additional context needed:
        - num_unmasked: tracks which block we're in
        - prompt_lens: to compute absolute position bounds
        """
        batch_size, seq_len = logits.shape[:2]
        device = logits.device
        
        if num_unmasked is None or prompt_lens is None:
            raise ValueError("BlockGreedyStrategy needs num_unmasked and prompt_lens")
        
        # Which block are we in? (answer-relative)
        block_idx = num_unmasked // self.block_size  # [batch_size]
        
        # Absolute block boundaries
        block_start = prompt_lens + block_idx * self.block_size  # [batch_size]
        block_end = block_start + self.block_size  # [batch_size]
        
        # Create block mask
        positions = torch.arange(seq_len, device=device)  # [seq_len]
        is_in_block = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1)) &
            (positions.unsqueeze(0) < block_end.unsqueeze(1))
        )  # [batch_size, seq_len]
        
        # Only consider maskable positions within current block
        is_maskable_in_block = is_maskable & is_in_block
        
        # Standard greedy selection within block
        log_probs = F.log_softmax(logits, dim=-1)
        confidence = log_probs.max(dim=-1).values
        
        confidence = torch.where(
            is_maskable_in_block,
            confidence,
            torch.tensor(float('-inf'), device=device)
        )
        
        if k is None:
            k = torch.full((batch_size,), self.k, device=device)
        
        return _select_topk_per_sample(confidence, k)

class BlockProbabilityMarginStrategy:
    """Unmask position(s) with highest probability margin within fixed-size blocks.

    Uses the margin between top-2 log-probabilities as the selection criterion,
    as proposed in arXiv:2502.06768.
    """

    def __init__(self, block_size: int, k: int = 1):
        """
        Args:
            block_size: Size of each block (in answer tokens, not total seq)
            k: Tokens to unmask per step within block
        """
        self.block_size = block_size
        self.k = k

    def select_positions(
        self,
        logits,
        is_maskable,
        k=None,
        num_unmasked: Optional[Tensor] = None,
        prompt_lens: Optional[Tensor] = None,
    ):
        batch_size, seq_len = logits.shape[:2]
        device = logits.device

        if num_unmasked is None or prompt_lens is None:
            raise ValueError("BlockProbabilityMarginStrategy needs num_unmasked and prompt_lens")

        # Which block are we in? (answer-relative)
        block_idx = num_unmasked // self.block_size

        # Absolute block boundaries
        block_start = prompt_lens + block_idx * self.block_size
        block_end = block_start + self.block_size

        # Create block mask
        positions = torch.arange(seq_len, device=device)
        is_in_block = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1)) &
            (positions.unsqueeze(0) < block_end.unsqueeze(1))
        )

        # Only consider maskable positions within current block
        is_maskable_in_block = is_maskable & is_in_block

        # Probability margin: difference between top-2 log-probs
        log_probs = F.log_softmax(logits, dim=-1)
        top2 = torch.topk(log_probs, k=2, dim=-1).values  # [B, L, 2]
        margin = top2[..., 0] - top2[..., 1]  # [B, L]

        margin = torch.where(
            is_maskable_in_block,
            margin,
            torch.tensor(float('-inf'), device=device)
        )

        if k is None:
            k = torch.full((batch_size,), self.k, device=device)

        return _select_topk_per_sample(margin, k)


class BlockLeftToRightStrategy:
    """Left-to-right unmasking within fixed-size blocks (autoregressive-like)."""
    
    def __init__(self, block_size: int, k: int = 1):
        """
        Args:
            block_size: Size of each block (in answer tokens, not total seq)
            k: Tokens to unmask per step within block
        """
        self.block_size = block_size
        self.k = k
    
    def select_positions(
        self, 
        logits, 
        is_maskable, 
        k=None,
        num_unmasked: Optional[Tensor] = None,  # [batch_size]
        prompt_lens: Optional[Tensor] = None,   # [batch_size]
    ):
        """
        Select k leftmost masked positions in current block.
        Ignores model confidence - purely positional.
        """
        batch_size, seq_len = logits.shape[:2]
        device = logits.device
        
        if num_unmasked is None or prompt_lens is None:
            raise ValueError("BlockLeftToRightStrategy needs num_unmasked and prompt_lens")
        
        # Which block are we in? (answer-relative)
        block_idx = num_unmasked // self.block_size  # [batch_size]
        
        # Absolute block boundaries
        block_start = prompt_lens + block_idx * self.block_size  # [batch_size]
        block_end = block_start + self.block_size  # [batch_size]
        
        # Create block mask
        positions = torch.arange(seq_len, device=device)  # [seq_len]
        is_in_block = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1)) &
            (positions.unsqueeze(0) < block_end.unsqueeze(1))
        )  # [batch_size, seq_len]
        
        # Only consider maskable positions within current block
        is_maskable_in_block = is_maskable & is_in_block
        
        # Left-to-right: use negative position as score
        # (leftmost positions get highest priority in topk)
        scores = -positions.unsqueeze(0).float()  # [batch_size, seq_len]
        
        # Mask out non-maskable positions with -inf (log scale)
        scores = torch.where(
            is_maskable_in_block,
            scores,
            torch.tensor(float('-inf'), device=device)
        )
        
        if k is None:
            k = torch.full((batch_size,), self.k, device=device)
        
        return _select_topk_per_sample(scores, k)


class BlockPermutationStrategy:
    """
    Evaluate all K! permutations within each block and follow the best one.
    
    Strategy flow:
    1. At block start: Lookahead evaluation via evaluate_block_permutations()
    2. Cache the best permutation ordering
    3. Each select_positions() call returns next position from that ordering
    
    Usage:
        strategy = BlockPermutationStrategy(block_size=4)
        strategy.reset_state(batch_size, device)
        
        # At each block boundary:
        best_perm = evaluate_block_permutations(...)  # External function
        strategy.set_block_permutation(block_idx, best_perm)
        
        # Then iteratively:
        positions = strategy.select_positions(...)
    """
    
    def __init__(self, block_size: int = 4):
        """
        Args:
            block_size: Tokens per block (4 → 24 permutations, 5 → 120, etc.)
        """
        if block_size > 5:
            import warnings
            warnings.warn(
                f"block_size={block_size} yields {math.factorial(block_size)} "
                f"permutations. Consider smaller blocks for efficiency."
            )
        
        self.block_size = block_size
        
        # State: cached best permutation per sample
        self.cached_permutation = None  # [batch_size, block_size] 
        self.cached_block_idx = None    # [batch_size]
        self.permutation_pos = None     # [batch_size] - current position in ordering
    
    def reset_state(self, batch_size: int, device: torch.device):
        """Initialize state for a new batch."""
        self.cached_permutation = None
        self.cached_block_idx = None
        self.permutation_pos = None
       
    def set_block_permutation(
        self,
        block_idx: Tensor,  # [batch_size]
        best_permutation: Tensor,  # [batch_size, block_size] - block-relative indices
    ):
        """
        Cache the best permutation ordering for current block.
        
        Args:
            block_idx: Which block (answer-relative) for each sample
            best_permutation: Block-relative positions [0, block_size) in unmasking order
        """
        batch_size = block_idx.shape[0]
        device = block_idx.device
        
        # Reset position counter when starting new block
        if (self.cached_block_idx is None or 
            not torch.equal(self.cached_block_idx, block_idx)):
            self.permutation_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        self.cached_permutation = best_permutation.clone()
        self.cached_block_idx = block_idx.clone()
    
    def select_positions(
        self,
        logits: Optional[Tensor],  # [batch_size, seq_len, vocab_size] - not used, we follow cache
        is_maskable: Tensor,  # [batch_size, seq_len]
        k: Optional[Tensor] = None,  # Ignored (always k=1)
        num_unmasked: Optional[Tensor] = None,  # [batch_size]
        prompt_lens: Optional[Tensor] = None,  # [batch_size]
    ) -> Tensor:
        """
        Return next position from cached permutation.
        
        Returns: [batch_size, 1] positions with -1 for inactive samples
        """
        batch_size, seq_len = is_maskable.shape
        device = is_maskable.device
        
        if self.cached_permutation is None:
            raise RuntimeError(
                "BlockPermutationStrategy.select_positions() called before "
                "set_block_permutation(). Must evaluate permutations first."
            )
        
        if num_unmasked is None or prompt_lens is None:
            raise ValueError("BlockPermutationStrategy needs num_unmasked and prompt_lens")
        
        # Compute block boundaries
        block_idx = num_unmasked // self.block_size
        block_start = prompt_lens + block_idx * self.block_size  # [batch_size]
        
        # Get next position from cached permutation for each sample
        positions = torch.full((batch_size, 1), -1, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            if self.permutation_pos[b] < self.block_size:
                # Block-relative position from permutation
                pos_in_block = self.cached_permutation[b, self.permutation_pos[b]]
                
                # Convert to absolute sequence position
                abs_pos = block_start[b] + pos_in_block
                
                # Verify it's valid and maskable
                if 0 <= abs_pos < seq_len and is_maskable[b, abs_pos]:
                    positions[b, 0] = abs_pos
        
        # Advance position counter
        self.permutation_pos += 1
        
        return positions


def _select_topk_per_sample(
    scores: Tensor,  # [batch_size, seq_len]
    k_per_sample: Tensor,  # [batch_size]
) -> Tensor:
    """
    Select top-k positions per sample (variable k).
    
    Returns: [batch_size, max_k] positions with -1 padding
    
    Convention: scores should have -inf for positions not to select
    """
    device = scores.device
    
    # Get max k for tensor size
    max_k = k_per_sample.max()
    assert max_k > 0, "k_per_sample must have at least one positive values"
    topk_values, topk_indices = torch.topk(scores, k=max_k, dim=1, sorted=True)  # [B, max_k]
    
    # Create mask: valid if (1) score > -inf, (2) within k for this sample
    score_mask = topk_values > float('-inf') # [B, max_k]
    k_mask = torch.arange(max_k, device=device).unsqueeze(0) < k_per_sample.unsqueeze(1)  # [B, max_k]
    valid_mask = score_mask & k_mask
    
    # Set invalid positions to -1
    positions = torch.where(valid_mask, topk_indices, torch.full_like(topk_indices, -1))
    return positions
