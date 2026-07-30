"""Pin the scored token set of the AR baseline's get_loglikelihood.

Regression test for the off-by-one fixed on branch `fix-ar-baseline-shift`.

`token_log_probs[i, j] = log p(input_ids[i, j+1] | x_{<=j})`, so the log-prob of the
token at ORIGINAL index q lives at SHIFTED index q-1. `_create_batch` already folds the
left-pad offset into `prompt_lens` (it returns `pad_len + prompt_len`). The continuation
therefore occupies original `[prompt_lens, W)` and its log-probs occupy shifted
`[prompt_lens-1, W-1)`.

The pre-fix code started at `prompt_lens`, dropping the first continuation token. On
multiple-choice that is the first content word, i.e. exactly the token that discriminates
between choices sharing a "Question:...Answer:" prefix.

Trick used here: with vocab size 2 and every logit row equal to [0, 0.5413], the
log-softmax of token 0 is exactly -1. All input ids are 0, so every gathered per-token
log-prob is -1 and the returned log-likelihood equals -(number of scored positions).
That turns an opaque float into a direct count of the scored set.

Runs on CPU; imports the module by path so the test does not depend on cwd.
"""
import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

_MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "eval_llama3.py"

# log_softmax([0.0, 0.5413])[0] == -1.0 (p = 1/(1+e^0.5413) = e^-1)
_LOGIT_HI = 0.5413248546129181


def _load_harness_class():
    """Import eval_llama3 without executing its CLI entry point."""
    spec = importlib.util.spec_from_file_location("_eval_llama3_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"eval_llama3.py not importable in this environment: {exc}")
    for name in ("Llama3EvalHarness", "Llama3Harness", "LLaMA3EvalHarness"):
        if hasattr(module, name):
            return getattr(module, name)
    lm_classes = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type) and hasattr(obj, "get_loglikelihood")
    ]
    if not lm_classes:  # pragma: no cover
        pytest.skip("no class exposing get_loglikelihood found in eval_llama3.py")
    return lm_classes[0]


class _ConstantLogitModel:
    """Returns logits whose log-softmax at token 0 is exactly -1 at every position."""

    def __call__(self, input_ids, attention_mask=None):
        b, w = input_ids.shape
        logits = torch.zeros(b, w, 2)
        logits[..., 1] = _LOGIT_HI
        return type("Out", (), {"logits": logits})()


def _score(harness_cls, *, width, prompt_lens, lengths):
    """Call the real get_loglikelihood with a stub model; return scored-token counts."""
    harness = harness_cls.__new__(harness_cls)  # bypass __init__ (loads an 8B model)
    harness.model = _ConstantLogitModel()
    harness.device = torch.device("cpu")
    harness._rank = 1  # suppress the wandb logging branch

    b = len(prompt_lens)
    input_ids = torch.zeros(b, width, dtype=torch.long)
    attention_mask = torch.ones(b, width, dtype=torch.long)
    ll = harness.get_loglikelihood(
        input_ids,
        attention_mask,
        torch.tensor(prompt_lens, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )
    # every per-token log-prob is exactly -1, so -LL is the scored-token count
    return [round(-float(v)) for v in ll]


@pytest.fixture(scope="module")
def harness_cls():
    return _load_harness_class()


def test_unpadded_continuation_scores_every_target_token(harness_cls):
    """prefix=5, target=3, no padding: all 3 target tokens must be scored.

    The pre-fix code returned 2 here -- it dropped the first continuation token.
    """
    assert _score(harness_cls, width=8, prompt_lens=[5], lengths=[8]) == [3]


def test_single_token_continuation_is_not_dropped_entirely(harness_cls):
    """A 1-token continuation is the sharpest case: pre-fix this scored NOTHING."""
    assert _score(harness_cls, width=6, prompt_lens=[5], lengths=[6]) == [1]


def test_left_padded_batch_scores_each_row_independently(harness_cls):
    """Ragged rows: prompt_lens is already pad-adjusted, so counts are pad-invariant.

    Row 0: pad_len=0, prompt ends at 5, width 8 -> 3 target tokens.
    Row 1: pad_len=3, prompt ends at 6, width 8 -> 2 target tokens.
    """
    counts = _score(harness_cls, width=8, prompt_lens=[5, 6], lengths=[8, 5])
    assert counts == [3, 2]


def test_padding_never_enters_the_scored_set(harness_cls):
    """The same continuation scores identically however much padding precedes it."""
    unpadded = _score(harness_cls, width=4, prompt_lens=[2], lengths=[4])
    padded = _score(harness_cls, width=10, prompt_lens=[8], lengths=[4])
    assert unpadded == padded == [2]


def test_rolling_path_scores_all_but_the_first_token(harness_cls):
    """prompt_len=0 (rolling). The first token has no scoreable predecessor.

    Under left-padding its log-prob would sit at a PADDING position, so W-1 is correct
    and is NOT an off-by-one -- it is the cost of not prepending an EOT separator, which
    upstream lm-eval does and this harness does not. Pinned so a future 'fix' that scores
    W tokens (reading a pad position as context) fails loudly.
    """
    assert _score(harness_cls, width=8, prompt_lens=[0], lengths=[8]) == [7]


def test_rolling_path_is_padding_invariant(harness_cls):
    """A padded rolling chunk scores len-1 tokens, not width-1."""
    assert _score(harness_cls, width=12, prompt_lens=[4], lengths=[8]) == [7]
