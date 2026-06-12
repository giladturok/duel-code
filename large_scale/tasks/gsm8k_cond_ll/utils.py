import datasets
import numpy as np


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Process GSM8K dataset for conditional likelihood evaluation.

    Presents each example as a single-option "multiple choice" task where the
    sole choice is the reference answer (including chain-of-thought reasoning).
    This allows computing P(answer | question) via the loglikelihood method.
    """
    def _process_doc(doc):
        return {
            "question": doc["question"],
            "choices": [" " + doc["answer"]],  # single choice: the full reference answer
            "label": 0,  # always the first (only) choice
        }

    return dataset.map(_process_doc)


def process_results_cond_ll(doc, results):
    """
    Extract conditional likelihood metrics from the single-choice result.

    Args:
        doc: dict with 'question', 'choices', 'label'
        results: list of (loglikelihood, is_greedy) tuples (length 1)
    """
    ll = results[0][0]
    answer_text = doc["choices"][0]
    answer_len = float(len(answer_text))

    nll_per_token = -ll / answer_len
    ppl_per_token = np.exp(nll_per_token)

    return {
        "ppl_per_token": ppl_per_token,
        "nll_per_token": nll_per_token,
        "total_nll": -ll,
    }
