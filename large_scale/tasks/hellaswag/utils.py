import re

import datasets
import numpy as np


def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "label": int(doc["label"]),
        }
        return out_doc
    return dataset.map(_process_doc)


def softmax(x):
    """Compute softmax from log probabilities."""
    x = np.array(x)
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def expected_calibration_error(items, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.array([item[0] for item in items])
    correctness = np.array([item[1] for item in items])
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            avg_accuracy_in_bin = np.mean(correctness[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
    
    return ece


def micro_average_perplexity(items):
    """
    Compute micro-average perplexity across all (question, choice) pairs.
    
    Args:
        items: list of lists, where each inner list contains perplexities
               for all choices of one question
    
    Returns:
        Single averaged perplexity value
    """
    # Flatten the list of lists into a single list
    all_perplexities = [ppl for item in items for ppl in item]
    return np.mean(all_perplexities)

def process_results_multiple_choice(doc, results):
    """Process results for multiple choice task.
    
    Assumes tasks conform to the arguments 'label' and 'choices'.
    
    Args:
        doc: dict with 'label' and 'choices'
        results: list of (loglikelihood, is_greedy) tuples
    """
    # Extract only loglikelihoods
    results = [res[0] for res in results]
    assert "label" in doc, "Document must contain 'label' key."
    label = doc["label"]
    
    # Accuracy
    pred = np.argmax(results)
    acc = 1.0 if pred == label else 0.0
    
    # Length-normalized accuracy
    assert "choices" in doc, "Document must contain 'choices' key."
    completion_len = np.array([float(len(i)) for i in doc["choices"]])
    acc_norm = 1.0 if np.argmax(results / completion_len) == label else 0.0
    
    # Brier score
    prob_norm = softmax(results)
    
    # Per-character perplexity for correct answer only
    nll_per_char_correct = -results[label] / len(doc["choices"][label])
    ppl_correct_answer_norm = np.exp(nll_per_char_correct)
    
    
    # Per-character perplexity for ALL choices (return as list for micro-averaging)
    ppl_all_choices_norm_list = [
        np.exp(-results[i] / len(doc["choices"][i])) 
        for i in range(len(results))
    ]
    
    # Calibration
    predicted_confidence = prob_norm[pred]
    is_correct = 1.0 if pred == label else 0.0
    
    return {
        "acc": acc,
        "acc_norm": acc_norm,
        "brier_score": (label, prob_norm),
        "ppl_correct_answer_norm": ppl_correct_answer_norm,
        "ppl_all_choices_norm": ppl_all_choices_norm_list,  # Return list!
        "ece": (predicted_confidence, is_correct),
    }