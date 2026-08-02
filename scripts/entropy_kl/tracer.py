"""Instrumentation for the strategy-based sampler: exact H(P_F) and path log-prob.

Attached to a `diffusion.Diffusion` as `model._duel_trace`; `_unmask_block_with_strategy`
(diffusion.py, step 5 of the loop) calls `record()` once per unmasking step and is a
no-op when nothing is attached.

For a *deterministic* unmasking rule F the induced test-time distribution factorises
exactly over reveals,

    log P_F(x) = sum_{reveals (t,i)} log p_theta(x_i | state_t),

so two unbiased estimators of H(P_F) = E_{x~P_F}[-log P_F(x)] are accumulated:

  (a) sampled-token / path estimator:  -sum log p_theta(x_i | state_t)
  (b) Rao-Blackwellised estimator:      sum H(p_theta(. | state_t))
      using E_{x_i ~ p(.|state_t)}[-log p(x_i|state_t)] = H(p(.|state_t)).
      Same estimand, strictly lower variance (conditioning on state_t removes all
      within-step sampling noise).

Both are read off `full_logits`, which for the SUBS parameterisation is already
log-normalised (diffusion.Diffusion._subs_parameterization), NOT off the
nucleus-renormalised `p_x0`. At nucleus_p=1.0 the two coincide and `check_*` below
asserts exactly that.
"""
import torch


class DuelTrace:
    """Per-position entropy / log-prob accumulator for one generation batch."""

    def __init__(self, batch_size, seq_len, device):
        self.B, self.L = batch_size, seq_len
        f64 = dict(dtype=torch.float64, device=device)
        # entropy of the categorical position j was drawn from
        self.ent = torch.full((batch_size, seq_len), float('nan'), **f64)
        # log p(sampled token) at position j
        self.lp = torch.full((batch_size, seq_len), float('nan'), **f64)
        # within-block step index at which position j was revealed
        self.step_in_block = torch.full((batch_size, seq_len), -1,
                                        dtype=torch.int16, device=device)
        self.block_of = torch.full((batch_size, seq_len), -1,
                                   dtype=torch.int16, device=device)
        self.n_written = torch.zeros((batch_size, seq_len), dtype=torch.int32,
                                     device=device)
        # one-time correctness checks (populated on the first record call)
        self.checks = {}

    @torch.no_grad()
    def record(self, full_logits, p_x0, positions, x_after, x_before,
               block_slice, step_in_block):
        valid = positions >= 0                      # [B, k]
        if not valid.any():
            return
        V = full_logits.shape[-1]
        safe = positions.clamp(min=0)               # [B, k]
        idx = safe.unsqueeze(-1).expand(-1, -1, V)

        lp_sel = full_logits.gather(1, idx)         # [B, k, V] log-probs
        p_sel = lp_sel.exp()

        if not self.checks:
            self._run_checks(full_logits, p_x0, block_slice, lp_sel, p_sel, idx)

        # H = -sum_v p log p, with the 0 * -inf entries (mask token, and any
        # position pinned to a one-hot) contributing exactly 0.
        prod = torch.where(p_sel > 0, p_sel * lp_sel, torch.zeros_like(p_sel))
        H = -prod.sum(dim=-1, dtype=torch.float64)  # [B, k]

        tok = x_after.gather(1, safe)               # [B, k] the token just sampled
        lp_tok = lp_sel.gather(-1, tok.unsqueeze(-1)).squeeze(-1).double()

        b_idx = torch.arange(self.B, device=safe.device).unsqueeze(1).expand_as(safe)
        bi, pi = b_idx[valid], safe[valid]
        # every position must be revealed exactly once
        assert (self.n_written[bi, pi] == 0).all(), 'position revealed twice'
        del x_before  # only used for debugging; positions are masked by construction
        self.n_written[bi, pi] = 1
        self.ent[bi, pi] = H[valid]
        self.lp[bi, pi] = lp_tok[valid]
        self.step_in_block[bi, pi] = int(step_in_block)
        self.block_of[bi, pi] = int(block_slice.start) // (block_slice.stop
                                                           - block_slice.start)

    def _run_checks(self, full_logits, p_x0, block_slice, lp_sel, p_sel, idx):
        """One-off assertions, run on the first unmasking step of a batch."""
        blk = full_logits[:, block_slice]
        # (1) full_logits is already log-normalised: log_softmax is a no-op.
        renorm = blk.log_softmax(dim=-1)
        d_logsoftmax = (renorm - blk)[torch.isfinite(blk)].abs().max().item()
        # (2) nucleus at p=1.0 is the identity, so p_x0 == exp(full_logits).
        p_gathered = p_x0.gather(1, idx)
        d_nucleus = (p_gathered - p_sel).abs().max().item()
        # (3) the categorical actually sums to one
        d_norm = (p_sel.sum(-1) - 1.0).abs().max().item()
        self.checks = {'max_abs_log_softmax_minus_full_logits': d_logsoftmax,
                       'max_abs_p_x0_minus_exp_full_logits': d_nucleus,
                       'max_abs_prob_sum_minus_one': d_norm}
        assert d_logsoftmax < 1e-4, f'full_logits not log-normalised: {d_logsoftmax}'
        assert d_nucleus < 1e-6, f'nucleus_p != 1.0 identity: {d_nucleus}'
        assert d_norm < 1e-4, f'categorical not normalised: {d_norm}'

    def summary(self, skip_bos=True):
        """Per-sequence totals. Position 0 (BOS) is context, never sampled."""
        sl = slice(1, None) if skip_bos else slice(None)
        ent, lp = self.ent[:, sl], self.lp[:, sl]
        written = self.n_written[:, sl] > 0
        n = written.sum(dim=1)
        H_rb = torch.where(written, ent, torch.zeros_like(ent)).sum(dim=1)
        H_path = -torch.where(written, lp, torch.zeros_like(lp)).sum(dim=1)
        return {'n_revealed': n, 'H_rb_total': H_rb, 'H_path_total': H_path}
