"""
Evaluation Metrics
==================
Comprehensive metrics for binary classification with statistical tests.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, balanced_accuracy_score
)
from scipy import stats


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels (0/1)
        y_prob: Predicted probabilities for class 1
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        # Primary metrics
        'auc': roc_auc_score(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        
        # Precision/Recall
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),  # Sensitivity
        'f1': f1_score(y_true, y_pred, zero_division=0),
        
        # Additional metrics
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative predictive value
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive predictive value
        
        # Counts
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        
        # Average precision
        'average_precision': average_precision_score(y_true, y_prob)
    }
    
    # Youden's J statistic (optimal threshold)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    metrics['optimal_threshold'] = thresholds[best_idx]
    metrics['youden_j'] = j_scores[best_idx]
    
    return metrics


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute AUC with bootstrap confidence interval.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        random_state: Random seed
        
    Returns:
        (auc, ci_lower, ci_upper)
    """
    np.random.seed(random_state)
    n = len(y_true)
    
    bootstrap_aucs = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        auc = roc_auc_score(y_true_boot, y_prob_boot)
        bootstrap_aucs.append(auc)
    
    bootstrap_aucs = np.array(bootstrap_aucs)
    
    # Percentile CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_aucs, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_aucs, (1 - alpha/2) * 100)
    
    auc = roc_auc_score(y_true, y_prob)
    
    return auc, ci_lower, ci_upper


def delong_test(
    y_true: np.ndarray,
    y_prob_1: np.ndarray,
    y_prob_2: np.ndarray
) -> Tuple[float, float]:
    """
    DeLong's test for comparing two AUC values.
    
    Tests whether two ROC curves are significantly different.
    
    Args:
        y_true: True labels
        y_prob_1: Predictions from model 1
        y_prob_2: Predictions from model 2
        
    Returns:
        (z_statistic, p_value)
    """
    # Compute AUCs
    auc1 = roc_auc_score(y_true, y_prob_1)
    auc2 = roc_auc_score(y_true, y_prob_2)
    
    # Compute variance using Hanley-McNeil approximation
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    
    q1 = auc1 / (2 - auc1)
    q2 = auc2 / (2 - auc2)
    
    var1 = (auc1 * (1 - auc1) + (n1 - 1) * (q1 - auc1**2) + 
            (n0 - 1) * (q2 - auc1**2)) / (n0 * n1)
    
    q1 = auc2 / (2 - auc2)
    q2 = auc2 / (2 - auc2)
    
    var2 = (auc2 * (1 - auc2) + (n1 - 1) * (q1 - auc2**2) + 
            (n0 - 1) * (q2 - auc2**2)) / (n0 * n1)
    
    # Compute covariance (simplified - assumes correlated predictions)
    cov = 0.5 * (var1 + var2)  # Rough estimate
    
    # Z-statistic
    se = np.sqrt(var1 + var2 - 2 * cov)
    if se == 0:
        return 0, 1.0
    
    z = (auc1 - auc2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p_value


def compute_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute detailed confusion matrix metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # True positive rate
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # True negative rate
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive predictive value
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative predictive value
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False positive rate
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,  # False negative rate
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'balanced_accuracy': 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if (tp + fn) > 0 and (tn + fp) > 0 else 0,
        'mcc': ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) 
               if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    }


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    n_bootstrap: int = 1000
) -> Dict[str, Dict]:
    """
    Compare multiple models with statistical tests.
    
    Args:
        y_true: True labels
        predictions: Dictionary of {model_name: y_prob}
        n_bootstrap: Number of bootstrap samples for CI
        
    Returns:
        Comparison results
    """
    results = {}
    model_names = list(predictions.keys())
    
    # Compute metrics for each model
    for name, y_prob in predictions.items():
        auc, ci_lower, ci_upper = bootstrap_auc_ci(y_true, y_prob, n_bootstrap)
        metrics = compute_all_metrics(y_true, y_prob)
        
        results[name] = {
            'auc': auc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            **metrics
        }
    
    # Pairwise comparisons
    comparisons = {}
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i+1:]:
            z, p = delong_test(y_true, predictions[name1], predictions[name2])
            comparisons[f"{name1}_vs_{name2}"] = {
                'z_statistic': z,
                'p_value': p,
                'significant_at_0.05': p < 0.05
            }
    
    results['pairwise_comparisons'] = comparisons
    
    return results
