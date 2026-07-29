"""
Equivalence test: subset-lattice DP vs. brute-force permutation enumeration.

`compute_exact_loglikelihood_cached_permutations` (the reference) enumerates all
`block_size!` unmasking orders per block, at a cost of `block_size! * block_size`
forwards per block.  `compute_exact_loglikelihood_cached_subset_dp` is supposed to
compute the *same* three quantities (`ll_oracle`, `ll_uniform_order`,
`ll_mixture`) with `2**m - 1` forwards per block, where `m` is the number of valid
positions in the block.  This file proves the two agree.

Pure math: CPU only, no model checkpoint, no GPU, milliseconds per config.

Run as either:
    pytest tests/test_subset_dp_equivalence.py
    python tests/test_subset_dp_equivalence.py

The K=8 config (2**8 = 256-cell lattice, 8! * 8 = 322,560 reference forwards) is
gated behind `SUBSET_DP_SLOW=1` / `-m slow`; a K=6 config covering a lattice
larger than K=4 runs by default.
"""
import math
import os
import sys
import time

import torch

# These are microscopic tensors ([2..3, 1..8, 11]); with the box's default 32
# intra-op threads the OpenMP barrier dominates and the suite runs ~11x slower.
torch.set_num_threads(1)

# The modules under test live at the repo root; there is no conftest.py.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exact_likelihood import compute_exact_loglikelihood_cached_permutations  # noqa: E402

try:  # the DP is being written concurrently; the file must still import without it
    from exact_likelihood import compute_exact_loglikelihood_cached_subset_dp
    HAVE_DP = True
    DP_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on sibling agent's progress
    compute_exact_loglikelihood_cached_subset_dp = None
    HAVE_DP = False
    DP_IMPORT_ERROR = str(exc)


try:
    import pytest
except ImportError:  # pragma: no cover - allow `python tests/...` without pytest
    class _Mark:
        def __getattr__(self, _name):
            def _decorator(fn):
                return fn
            return _decorator

    class _PytestStub:
        mark = _Mark()

        @staticmethod
        def skip(reason=""):
            raise RuntimeError("skip: " + reason)

    pytest = _PytestStub()  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Fake forward
# --------------------------------------------------------------------------
def make_fake(V, D, K, seed=0):
    """A model whose block logits depend on the current block's tokens and the
    committed prefix, and on *nothing else* -- in particular not on the order in
    which the block's positions were revealed.  That order-independence is the
    DP's only assumption about the model, so it is exactly what the fake encodes.

    Returns (fwd, state).  `state` also carries the commit=False call counter and
    a `reset()` used to hand a clean model to each implementation in turn.
    """
    g = torch.Generator().manual_seed(seed)
    Wemb = torch.randn(V, D, generator=g)
    Wout = torch.randn(D, V, generator=g)
    state = {'prefix': None, 'n_forward': 0, 'n_commit': 0}

    def fwd(x_blk, commit=False):            # x_blk [B, K] -> [B, K, V]
        h = Wemb[x_blk]                                  # [B, K, D]
        h = h + h.sum(1, keepdim=True)                   # bidirectional mixing in block
        if state['prefix'] is not None:
            h = h + state['prefix'].unsqueeze(1)
        if commit:
            state['n_commit'] += 1
            p = h.mean(1)
            state['prefix'] = p if state['prefix'] is None else state['prefix'] + p
            return None                                  # reference ignores this
        state['n_forward'] += 1
        return h @ Wout

    def reset():
        state['prefix'] = None
        state['n_forward'] = 0
        state['n_commit'] = 0

    state['reset'] = reset
    return fwd, state


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------
V = 11          # small vocab: keeps the reference's K! enumeration cheap
D = 8
MASK_TOKEN_ID = V - 1


class Config:
    def __init__(self, name, L, K, B, lengths, slow=False, seed=0):
        self.name = name
        self.L = L
        self.K = K
        self.B = B
        self.lengths = list(lengths)
        self.slow = slow
        self.seed = seed
        assert len(self.lengths) == B
        assert all(0 <= n <= L for n in self.lengths)

    @property
    def homogeneous(self):
        return len(set(self.lengths)) == 1

    def build(self):
        g = torch.Generator().manual_seed(1234 + self.seed)
        # draw from [0, V-1) so x0 can never collide with mask_token_id = V-1
        x0 = torch.randint(0, V - 1, (self.B, self.L), generator=g)
        lengths = torch.tensor(self.lengths, dtype=torch.long)
        attention_mask = (
            torch.arange(self.L).unsqueeze(0) < lengths.unsqueeze(1)
        ).long()
        return x0, attention_mask

    def expected_dp_forwards(self):
        """sum over blocks of (2**m_block - 1), m_block = #valid positions.

        Only meaningful for homogeneous lengths, where every example in the batch
        has the same m in every block.
        """
        assert self.homogeneous
        n = self.lengths[0]
        total = 0
        for start in range(0, self.L, self.K):
            end = min(start + self.K, self.L)
            m = max(0, min(end, n) - start)
            if m == 0:
                continue
            total += 2 ** m - 1
        return total

    def reference_forwards(self):
        total = 0
        for start in range(0, self.L, self.K):
            end = min(start + self.K, self.L)
            width = end - start
            if not any(n > start for n in self.lengths):
                continue                      # reference skips all-invalid blocks
            total += math.factorial(width) * width
        return total


CONFIGS = [
    # 1. homogeneous, full blocks
    Config("c1_L16_K4_full", L=16, K=4, B=3, lengths=[16, 16, 16]),
    # 2. homogeneous, final block has m=3 -- the production case, since
    #    `ignore_bos` makes lengths = L - 1
    Config("c2_L16_K4_len15", L=16, K=4, B=3, lengths=[15, 15, 15]),
    # 3. RAGGED: exercises m=1, m=0 and the per-example subset gate at once
    Config("c3_L16_K4_ragged", L=16, K=4, B=3, lengths=[16, 13, 6]),
    # 4-fast. lattice bigger than K=4 that still runs by default (6! * 6 = 4320
    #   reference forwards vs. 2**6 - 1 = 63 for the DP).  L=6 rather than L=8
    #   because the DP requires block_size to divide seq_len (its KV-cache write
    #   is a fixed-width slice), so a short final block is not a legal input.
    Config("c4fast_L6_K6", L=6, K=6, B=2, lengths=[6, 6]),
    # 4. single full 2**8 production lattice (8! * 8 = 322,560 reference forwards)
    Config("c4_L8_K8_slow", L=8, K=8, B=2, lengths=[8, 8], slow=True),
    # 5. degenerate m=1 blocks
    Config("c5a_L8_K2", L=8, K=2, B=2, lengths=[8, 8]),
    Config("c5b_L4_K1", L=4, K=1, B=2, lengths=[4, 4]),
]

SLOW_ENABLED = os.environ.get("SUBSET_DP_SLOW", "") == "1"

# Tolerance for the two *reassociating* reductions (mean / logsumexp).  Unlike
# the oracle (max), these sum the whole [B, n_perms] table, and the two
# implementations sum it in different orders, so they differ by fp32 rounding.
#
# The nominal spec for this test was rtol=0, atol=1e-5.  That is below one fp32
# ULP for the quantity being compared: these are block-summed log-likelihoods of
# magnitude ~2e2 to ~9e2, where one ULP is 1.5e-5 to 6.1e-5.  Measured deviation
# is *exactly* 1 ULP on every config -- i.e. the tightest a reassociated fp32 sum
# can possibly be -- so atol=1e-5 is unsatisfiable by any correct implementation
# rather than a real signal.  We therefore scale the tolerance to the magnitude:
# `max(1e-5, N_ULP * ulp(max|ref|))`.  This keeps the spec'd 1e-5 for
# small-magnitude cases and stays ~4 orders of magnitude tighter than any genuine
# structural bug, which perturbs these numbers by O(0.1-10) nats, not O(1e-4).
#
# The oracle assertion below is BITWISE and is NOT covered by this.
N_ULP = 8


def _ulp_atol(ref: torch.Tensor, n_ulp: int = N_ULP, floor: float = 1e-5) -> float:
    scale = ref.abs().max().item()
    if not math.isfinite(scale) or scale == 0.0:
        return floor
    exp = math.frexp(scale)[1]                 # scale = m * 2**exp, m in [0.5, 1)
    ulp = math.ldexp(1.0, max(exp - 24, -149))  # fp32 has a 24-bit significand
    return max(floor, n_ulp * ulp)


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------
def run_impl(fn, cfg, x0, attention_mask, fwd, state):
    """Run one implementation on a clean model state; return (extras, ll_total,
    steps, n_forward, wall_seconds)."""
    state['reset']()
    t0 = time.perf_counter()
    ll_total, steps, extras = fn(
        x0,
        fwd,
        MASK_TOKEN_ID,
        strategy=None,
        attention_mask=attention_mask,
        block_size=cfg.K,
        return_per_order=True,
    )
    dt = time.perf_counter() - t0
    return ll_total, steps, extras, state['n_forward'], dt


def compare(cfg):
    """Run both implementations on the same fake model and return a result dict."""
    x0, attention_mask = cfg.build()
    fwd, state = make_fake(V, D, cfg.K, seed=cfg.seed)

    ll_ref, steps_ref, ex_ref, nf_ref, t_ref = run_impl(
        compute_exact_loglikelihood_cached_permutations, cfg, x0, attention_mask, fwd, state
    )
    ll_dp, steps_dp, ex_dp, nf_dp, t_dp = run_impl(
        compute_exact_loglikelihood_cached_subset_dp, cfg, x0, attention_mask, fwd, state
    )

    dev = {
        k: (ex_ref[k] - ex_dp[k]).abs().max().item()
        for k in ('ll_oracle', 'll_uniform_order', 'll_mixture')
    }
    # same deviation expressed in fp32 ULPs at the magnitude of the values
    ulps = {
        k: dev[k] / (_ulp_atol(ex_ref[k], n_ulp=1, floor=0.0) or 1.0)
        for k in dev
    }
    return dict(
        cfg=cfg, x0=x0, attention_mask=attention_mask,
        ll_ref=ll_ref, ll_dp=ll_dp, ex_ref=ex_ref, ex_dp=ex_dp,
        steps_ref=steps_ref, steps_dp=steps_dp,
        nf_ref=nf_ref, nf_dp=nf_dp, t_ref=t_ref, t_dp=t_dp,
        dev=dev, ulps=ulps,
        oracle_bitwise=torch.equal(ex_ref['ll_oracle'], ex_dp['ll_oracle']),
    )


def check(res):
    """Assert everything we require of a comparison. Raises AssertionError."""
    cfg = res['cfg']
    ex_ref, ex_dp = res['ex_ref'], res['ex_dp']

    # ll_total is the oracle reduction within each implementation (in the
    # reference they are literally the same object).
    assert torch.equal(res['ll_ref'], ex_ref['ll_oracle']), \
        f"{cfg.name}: reference ll_total != extras['ll_oracle']"
    assert torch.equal(res['ll_dp'], ex_dp['ll_oracle']), \
        f"{cfg.name}: DP ll_total != extras['ll_oracle']"

    for k in ('ll_oracle', 'll_uniform_order', 'll_mixture'):
        assert ex_dp[k].shape == (cfg.B,), f"{cfg.name}: DP {k} has shape {tuple(ex_dp[k].shape)}, want ({cfg.B},)"
        assert torch.isfinite(ex_dp[k]).all(), f"{cfg.name}: DP {k} is not finite: {ex_dp[k]}"

    # BITWISE, deliberately. fp32 addition is monotone non-decreasing in each
    # argument under round-to-nearest, and both implementations accumulate a path
    # total left-to-right in the same association order, so
    #   max_p (sum_p + w) == (max_p sum_p) + w
    # holds exactly; and the per-state log-probs are bitwise identical because
    # they come from the same deterministic function of the same input. A failure
    # here is a STRUCTURAL bug -- wrong association order, a stale DP cell, or `w`
    # built from the wrong input -- and must be reported, not papered over with a
    # tolerance.
    assert torch.equal(ex_ref['ll_oracle'], ex_dp['ll_oracle']), (
        f"{cfg.name}: ll_oracle is not BITWISE equal.\n"
        f"  ref = {ex_ref['ll_oracle']}\n"
        f"  dp  = {ex_dp['ll_oracle']}\n"
        f"  max|diff| = {res['dev']['ll_oracle']:.3e}\n"
        "  Do NOT relax this to a tolerance: it indicates a structural bug."
    )

    # See N_ULP / _ulp_atol above for why the tolerance tracks fp32 resolution
    # instead of being a flat 1e-5.
    for key in ('ll_uniform_order', 'll_mixture'):
        atol = _ulp_atol(ex_ref[key])
        torch.testing.assert_close(
            ex_ref[key], ex_dp[key], rtol=0, atol=atol,
            msg=lambda m, key=key, atol=atol: (
                f"{cfg.name}: {key} mismatch (atol={atol:.3e} = "
                f"{N_ULP} fp32 ULP at |ll|~{ex_ref[key].abs().max().item():.1f})\n{m}"
            ),
        )

    # Forward-count budget. Catches a DP that over-enumerates and still lands on
    # the right numbers. Only asserted for homogeneous lengths, where every
    # example shares the same per-block m.
    # NOTE: we deliberately do NOT compare the returned `steps` between the two
    # implementations -- they use different NFE conventions (the reference counts
    # K! * K per block).
    if cfg.homogeneous:
        want = cfg.expected_dp_forwards()
        assert res['nf_dp'] == want, (
            f"{cfg.name}: DP did {res['nf_dp']} commit=False forwards, "
            f"expected sum_b (2**m_b - 1) = {want}"
        )


# --------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------
def _maybe_skip(cfg):
    if cfg.slow and not SLOW_ENABLED:
        pytest.skip(f"{cfg.name} is slow; set SUBSET_DP_SLOW=1 (or -m slow) to run")
    if not HAVE_DP:
        pytest.skip(
            "compute_exact_loglikelihood_cached_subset_dp is not implemented yet "
            f"({DP_IMPORT_ERROR})"
        )


@pytest.mark.parametrize("cfg", [c for c in CONFIGS if not c.slow], ids=lambda c: c.name)
def test_subset_dp_matches_permutations(cfg):
    _maybe_skip(cfg)
    check(compare(cfg))


@pytest.mark.slow
@pytest.mark.parametrize("cfg", [c for c in CONFIGS if c.slow], ids=lambda c: c.name)
def test_subset_dp_matches_permutations_slow(cfg):
    _maybe_skip(cfg)
    check(compare(cfg))


@pytest.mark.parametrize("cfg", [c for c in CONFIGS if not c.slow], ids=lambda c: c.name)
def test_reference_is_self_consistent(cfg):
    """Validates the fake forward and the harness independently of the DP: the
    reference run twice against the same (reset) model must be bitwise identical.
    If this fails, the fake is leaking state across runs and no equivalence claim
    below it means anything."""
    x0, attention_mask = cfg.build()
    fwd, state = make_fake(V, D, cfg.K, seed=cfg.seed)
    _, _, a, nf_a, _ = run_impl(
        compute_exact_loglikelihood_cached_permutations, cfg, x0, attention_mask, fwd, state
    )
    _, _, b, nf_b, _ = run_impl(
        compute_exact_loglikelihood_cached_permutations, cfg, x0, attention_mask, fwd, state
    )
    assert nf_a == nf_b == cfg.reference_forwards()
    for k in ('ll_oracle', 'll_uniform_order', 'll_mixture'):
        assert torch.equal(a[k], b[k]), f"{cfg.name}: reference not deterministic in {k}"


# --------------------------------------------------------------------------
# __main__ : compact PASS/FAIL table
# --------------------------------------------------------------------------
def _main():
    torch.set_printoptions(precision=8)
    header = (
        f"{'config':<20} {'B':>2} {'L':>3} {'K':>2} {'lengths':<14} "
        f"{'ref_fwd':>8} {'dp_fwd':>7} {'want':>6} "
        f"{'d_oracle':>13} {'d_unif':>13} {'d_mix':>13}  {'bitwise':<8} {'status'}"
    )

    if not HAVE_DP:
        print("=" * 118)
        print("compute_exact_loglikelihood_cached_subset_dp NOT FOUND in exact_likelihood.py")
        print(f"  ({DP_IMPORT_ERROR})")
        print("Falling back to REFERENCE-ONLY validation: running the reference twice")
        print("per config and checking it agrees with itself bitwise. This validates the")
        print("fake forward, the state reset and the harness, but proves nothing about the DP.")
        print("=" * 118)
        print(f"{'config':<20} {'B':>2} {'L':>3} {'K':>2} {'lengths':<14} "
              f"{'ref_fwd':>8} {'want_fwd':>9} {'dp_fwd_budget':>14}  {'status'}")
        n_fail = 0
        for cfg in CONFIGS:
            if cfg.slow and not SLOW_ENABLED:
                print(f"{cfg.name:<20} {cfg.B:>2} {cfg.L:>3} {cfg.K:>2} "
                      f"{str(cfg.lengths):<14} {'-':>8} {'-':>9} {'-':>14}  SKIP (SUBSET_DP_SLOW=1 to run)")
                continue
            try:
                x0, am = cfg.build()
                fwd, state = make_fake(V, D, cfg.K, seed=cfg.seed)
                _, _, a, nf_a, t_a = run_impl(
                    compute_exact_loglikelihood_cached_permutations, cfg, x0, am, fwd, state)
                _, _, b, nf_b, _ = run_impl(
                    compute_exact_loglikelihood_cached_permutations, cfg, x0, am, fwd, state)
                ok = (nf_a == nf_b == cfg.reference_forwards()) and all(
                    torch.equal(a[k], b[k])
                    for k in ('ll_oracle', 'll_uniform_order', 'll_mixture'))
                budget = cfg.expected_dp_forwards() if cfg.homogeneous else -1
                status = f"PASS ({t_a:.2f}s ref)" if ok else "FAIL"
                n_fail += 0 if ok else 1
                print(f"{cfg.name:<20} {cfg.B:>2} {cfg.L:>3} {cfg.K:>2} "
                      f"{str(cfg.lengths):<14} {nf_a:>8} {cfg.reference_forwards():>9} "
                      f"{(budget if budget >= 0 else 'ragged'):>14}  {status}")
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                print(f"{cfg.name:<20} {cfg.B:>2} {cfg.L:>3} {cfg.K:>2} "
                      f"{str(cfg.lengths):<14} {'-':>8} {'-':>9} {'-':>14}  ERROR: {exc!r}")
        print("=" * 118)
        print("REFERENCE-ONLY: DP equivalence NOT tested (function missing).")
        return 1 if n_fail else 0

    print("=" * 130)
    print("subset-DP vs. permutation-enumeration equivalence")
    print("=" * 130)
    print(header)
    n_fail = 0
    for cfg in CONFIGS:
        if cfg.slow and not SLOW_ENABLED:
            print(f"{cfg.name:<20} {cfg.B:>2} {cfg.L:>3} {cfg.K:>2} {str(cfg.lengths):<14} "
                  f"{'-':>8} {'-':>7} {'-':>6} {'-':>13} {'-':>13} {'-':>13}  {'-':<8} "
                  f"SKIP (set SUBSET_DP_SLOW=1)")
            continue
        try:
            res = compare(cfg)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"{cfg.name:<20} {cfg.B:>2} {cfg.L:>3} {cfg.K:>2} {str(cfg.lengths):<14} "
                  f"{'-':>8} {'-':>7} {'-':>6} {'-':>13} {'-':>13} {'-':>13}  {'-':<8} "
                  f"ERROR: {exc!r}")
            continue
        want = cfg.expected_dp_forwards() if cfg.homogeneous else None
        try:
            check(res)
            status = f"PASS  (ref {res['t_ref']:.2f}s / dp {res['t_dp']:.3f}s)"
        except AssertionError as exc:
            n_fail += 1
            status = f"FAIL: {str(exc).splitlines()[0]}"
        d, u = res['dev'], res['ulps']
        fmt = lambda k: f"{d[k]:.2e}/{u[k]:.0f}u"  # noqa: E731
        print(f"{cfg.name:<20} {cfg.B:>2} {cfg.L:>3} {cfg.K:>2} {str(cfg.lengths):<14} "
              f"{res['nf_ref']:>8} {res['nf_dp']:>7} {(want if want is not None else 'ragg'):>6} "
              f"{fmt('ll_oracle'):>13} {fmt('ll_uniform_order'):>13} {fmt('ll_mixture'):>13}  "
              f"{str(res['oracle_bitwise']):<8} {status}")
    print("=" * 130)
    print("ALL PASS" if n_fail == 0 else f"{n_fail} CONFIG(S) FAILED")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(_main())
