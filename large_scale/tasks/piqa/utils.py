import datasets
import numpy as np

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Transform PIQA dataset to standard format.
    Converts sol1/sol2 into choices list.
    """
    def _process_doc(doc):
        out_doc = {
            "goal": doc["goal"],
            "choices": [doc["sol1"], doc["sol2"]],
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
    """Compute micro-average perplexity across all (question, choice) pairs."""
    all_perplexities = [ppl for item in items for ppl in item]
    return np.mean(all_perplexities)

def process_results_multiple_choice(doc, results):
    """Process results for multiple choice task."""
    results = [res[0] for res in results]
    
    assert "label" in doc, "Document must contain 'label' key."
    label = doc["label"]
    
    pred = np.argmax(results)
    acc = 1.0 if pred == label else 0.0
    
    assert "choices" in doc, "Document must contain 'choices' key."
    completion_len = np.array([float(len(i)) for i in doc["choices"]])
    acc_norm = 1.0 if np.argmax(results / completion_len) == label else 0.0
    
    prob_norm = softmax(results)
    
    nll_per_char_correct = -results[label] / len(doc["choices"][label])
    ppl_correct_answer_norm = np.exp(nll_per_char_correct)
    
    ppl_all_choices_norm_list = [
        np.exp(-results[i] / len(doc["choices"][i])) 
        for i in range(len(results))
    ]
    
    predicted_confidence = prob_norm[pred]
    is_correct = 1.0 if pred == label else 0.0
    
    return {
        "acc": acc,
        "acc_norm": acc_norm,
        "brier_score": (label, prob_norm),
        "ppl_correct_answer_norm": ppl_correct_answer_norm,
        "ppl_all_choices_norm": ppl_all_choices_norm_list,
        "ece": (predicted_confidence, is_correct),
    }