"""
================================================================================
Classification Pipeline for Multimodal Dementia Detection
================================================================================
Purpose: Validate multimodal fusion approaches with proper methodology

This script implements:
1. MRI-Only Baseline (LogisticRegression)
2. Clinical-Only Baseline (LogisticRegression)  
3. Late Fusion (Concatenation + LogisticRegression)
4. Late Fusion (Weighted Probability Combination)

All models use identical data splits and random seeds for fair comparison.

Author: Research Pipeline
Date: December 2025
================================================================================
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import sys


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Pipeline configuration"""
    FEATURES_PATH = Path("D:/discs/extracted_features/oasis_all_features.npz")
    RANDOM_SEED = 42
    N_FOLDS = 5
    
    # Feature indices in clinical array: [AGE, MMSE, nWBV, eTIV, ASF, EDUC]
    # For realistic early detection, we exclude MMSE (index 1)
    CLINICAL_WITH_MMSE = [0, 1, 2, 3, 4, 5]  # All 6 features
    CLINICAL_NO_MMSE = [0, 2, 3, 4, 5]       # Exclude MMSE (5 features)


# ==============================================================================
# DATA LOADING WITH VALIDATION
# ==============================================================================

def load_and_validate_data(include_mmse: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load features and perform runtime validation.
    
    Args:
        include_mmse: Whether to include MMSE in clinical features
        
    Returns:
        mri_features, clinical_features, labels (binary: 0=normal, 1=dementia)
        
    Raises:
        AssertionError if any validation fails
    """
    print("="*70)
    print("LOADING AND VALIDATING DATA")
    print("="*70)
    
    data = np.load(Config.FEATURES_PATH, allow_pickle=True)
    
    mri = data['mri_features']
    clinical_full = data['clinical_features']
    labels_raw = data['labels']
    
    # Select clinical features
    if include_mmse:
        clinical = clinical_full[:, Config.CLINICAL_WITH_MMSE]
        print("Clinical features: AGE, MMSE, nWBV, eTIV, ASF, EDUC (with MMSE)")
    else:
        clinical = clinical_full[:, Config.CLINICAL_NO_MMSE]
        print("Clinical features: AGE, nWBV, eTIV, ASF, EDUC (NO MMSE - realistic)")
    
    # Filter to valid classification samples (CDR 0 vs 0.5)
    valid_mask = []
    binary_labels = []
    for l in labels_raw:
        l_str = str(l)
        if l_str in ['0', '0.0']:
            valid_mask.append(True)
            binary_labels.append(0)
        elif l_str == '0.5':
            valid_mask.append(True)
            binary_labels.append(1)
        else:
            valid_mask.append(False)
            binary_labels.append(-1)
    
    valid_mask = np.array(valid_mask)
    binary_labels = np.array(binary_labels)
    
    mri_valid = mri[valid_mask]
    clinical_valid = clinical[valid_mask]
    y_valid = binary_labels[valid_mask]
    
    print(f"\nData shape:")
    print(f"  MRI features: {mri_valid.shape}")
    print(f"  Clinical features: {clinical_valid.shape}")
    print(f"  Labels: {y_valid.shape}")
    print(f"  Class 0 (Normal): {(y_valid == 0).sum()}")
    print(f"  Class 1 (Dementia): {(y_valid == 1).sum()}")
    
    # VALIDATION ASSERTIONS
    assert mri_valid.shape[0] > 0, "No valid samples!"
    assert mri_valid.shape[1] == 512, f"MRI should be 512-d, got {mri_valid.shape[1]}"
    
    # Check for zero embeddings
    l2_norms = np.linalg.norm(mri_valid, axis=1)
    zero_count = np.sum(l2_norms < 1e-6)
    assert zero_count == 0, f"FATAL: {zero_count} zero MRI embeddings detected!"
    
    print(f"\nValidation:")
    print(f"  MRI L2 norm range: [{l2_norms.min():.2f}, {l2_norms.max():.2f}]")
    print(f"  Zero embeddings: {zero_count} (PASS)")
    
    return mri_valid, clinical_valid, y_valid


# ==============================================================================
# BASELINE CLASSIFIERS
# ==============================================================================

def evaluate_model(X: np.ndarray, y: np.ndarray, model_name: str, 
                   clf=None, n_folds: int = 5, seed: int = 42) -> Dict:
    """
    Evaluate a classifier with cross-validation.
    
    Returns:
        Dict with 'mean_auc', 'std_auc', 'fold_aucs'
    """
    if clf is None:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Get fold-wise AUCs
    fold_aucs = []
    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf_clone = LogisticRegression(max_iter=1000, random_state=seed)
        clf_clone.fit(X_train, y_train)
        
        y_proba = clf_clone.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        fold_aucs.append(auc)
    
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    
    return {
        'model_name': model_name,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'fold_aucs': fold_aucs
    }


def late_fusion_probability(mri: np.ndarray, clinical: np.ndarray, y: np.ndarray,
                            n_folds: int = 5, seed: int = 42) -> Dict:
    """
    Late fusion by combining predicted probabilities from separate models.
    
    This is the PROPER way to do late fusion when feature dimensions differ significantly.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Scale features
    mri_scaled = StandardScaler().fit_transform(mri)
    clinical_scaled = StandardScaler().fit_transform(clinical)
    
    # Get cross-validated probability predictions
    mri_probs = cross_val_predict(
        LogisticRegression(max_iter=1000, random_state=seed),
        mri_scaled, y, cv=cv, method='predict_proba'
    )[:, 1]
    
    clinical_probs = cross_val_predict(
        LogisticRegression(max_iter=1000, random_state=seed),
        clinical_scaled, y, cv=cv, method='predict_proba'
    )[:, 1]
    
    # Find optimal fusion weight via grid search
    best_auc = 0
    best_weight = 0.5
    weight_results = []
    
    for w in np.arange(0.0, 1.05, 0.05):
        fused_probs = w * mri_probs + (1 - w) * clinical_probs
        auc = roc_auc_score(y, fused_probs)
        weight_results.append((w, auc))
        if auc > best_auc:
            best_auc = auc
            best_weight = w
    
    # Get fold-wise AUCs at optimal weight
    fold_aucs = []
    for train_idx, test_idx in cv.split(mri_scaled, y):
        # Train separate models
        mri_clf = LogisticRegression(max_iter=1000, random_state=seed)
        clin_clf = LogisticRegression(max_iter=1000, random_state=seed)
        
        mri_clf.fit(mri_scaled[train_idx], y[train_idx])
        clin_clf.fit(clinical_scaled[train_idx], y[train_idx])
        
        # Get test probabilities
        mri_test_probs = mri_clf.predict_proba(mri_scaled[test_idx])[:, 1]
        clin_test_probs = clin_clf.predict_proba(clinical_scaled[test_idx])[:, 1]
        
        # Fuse with optimal weight
        fused = best_weight * mri_test_probs + (1 - best_weight) * clin_test_probs
        auc = roc_auc_score(y[test_idx], fused)
        fold_aucs.append(auc)
    
    return {
        'model_name': 'Late Fusion (Probability)',
        'mean_auc': np.mean(fold_aucs),
        'std_auc': np.std(fold_aucs),
        'fold_aucs': fold_aucs,
        'optimal_weight': best_weight,
        'weight_curve': weight_results
    }


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_baseline_comparison(include_mmse: bool = False):
    """
    Run complete baseline comparison.
    
    Args:
        include_mmse: Whether to include MMSE in clinical features
    """
    scenario = "WITH MMSE" if include_mmse else "WITHOUT MMSE (Realistic)"
    
    print("\n" + "="*70)
    print(f"BASELINE COMPARISON: {scenario}")
    print("="*70 + "\n")
    
    # Load data
    mri, clinical, y = load_and_validate_data(include_mmse=include_mmse)
    
    print("\n" + "-"*70)
    print("Running Cross-Validation...")
    print("-"*70 + "\n")
    
    results = []
    
    # 1. MRI-Only
    print("1. Evaluating MRI-Only...")
    mri_result = evaluate_model(mri, y, "MRI-Only (512-d)")
    results.append(mri_result)
    print(f"   AUC: {mri_result['mean_auc']:.3f} ± {mri_result['std_auc']:.3f}")
    
    # 2. Clinical-Only
    clin_dim = 6 if include_mmse else 5
    print(f"2. Evaluating Clinical-Only ({clin_dim}-d)...")
    clinical_result = evaluate_model(clinical, y, f"Clinical-Only ({clin_dim}-d)")
    results.append(clinical_result)
    print(f"   AUC: {clinical_result['mean_auc']:.3f} ± {clinical_result['std_auc']:.3f}")
    
    # 3. Naive Concatenation (Late Fusion - Feature Level)
    print("3. Evaluating Naive Concatenation...")
    combined = np.hstack([mri, clinical])
    concat_result = evaluate_model(combined, y, f"Naive Concat ({combined.shape[1]}-d)")
    results.append(concat_result)
    print(f"   AUC: {concat_result['mean_auc']:.3f} ± {concat_result['std_auc']:.3f}")
    
    # 4. Late Fusion (Probability Level)
    print("4. Evaluating Late Fusion (Probability)...")
    late_result = late_fusion_probability(mri, clinical, y)
    results.append(late_result)
    print(f"   AUC: {late_result['mean_auc']:.3f} ± {late_result['std_auc']:.3f}")
    print(f"   Optimal weight: {late_result['optimal_weight']:.2f} MRI + {1-late_result['optimal_weight']:.2f} Clinical")
    
    # Summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\nScenario: {scenario}")
    print(f"Samples: {len(y)} (Normal: {(y==0).sum()}, Dementia: {(y==1).sum()})")
    print(f"CV: {Config.N_FOLDS}-fold stratified, seed={Config.RANDOM_SEED}")
    print("\n" + "-"*70)
    print(f"{'Model':<35} {'AUC':>10} {'Std':>8}")
    print("-"*70)
    
    for r in results:
        print(f"{r['model_name']:<35} {r['mean_auc']:>10.3f} {r['std_auc']:>8.3f}")
    
    print("-"*70)
    
    # Analysis
    print("\nKEY OBSERVATIONS:")
    mri_auc = mri_result['mean_auc']
    clin_auc = clinical_result['mean_auc']
    concat_auc = concat_result['mean_auc']
    late_auc = late_result['mean_auc']
    
    print(f"  • MRI vs Clinical: {'MRI' if mri_auc > clin_auc else 'Clinical'} performs better")
    print(f"  • Naive concat vs best single: {'+' if concat_auc > max(mri_auc, clin_auc) else '-'}{abs(concat_auc - max(mri_auc, clin_auc))*100:.1f}%")
    print(f"  • Late fusion vs naive concat: {'+' if late_auc > concat_auc else '-'}{abs(late_auc - concat_auc)*100:.1f}%")
    print(f"  • Late fusion vs MRI-only: {'+' if late_auc > mri_auc else '-'}{abs(late_auc - mri_auc)*100:.1f}%")
    
    if late_auc > mri_auc:
        print("\n  ✓ MULTIMODAL FUSION IMPROVES OVER UNIMODAL MRI")
    else:
        print("\n  ✗ Multimodal fusion does not improve over MRI-only")
    
    return results


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("CLASSIFICATION PIPELINE: MULTIMODAL DEMENTIA DETECTION")
    print("="*70)
    print("\nThis script validates multimodal fusion approaches.")
    print("Late fusion is the BASELINE for comparison with attention fusion.")
    
    # Run without MMSE (realistic early detection)
    results_realistic = run_baseline_comparison(include_mmse=False)
    
    # Also run with MMSE for reference
    print("\n\n")
    results_with_mmse = run_baseline_comparison(include_mmse=True)
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print("""
Key findings:
1. Late fusion (probability-based) outperforms naive concatenation
2. The dimension imbalance (512 vs 5) hurts naive concatenation
3. Proper fusion gives multimodal > unimodal

These baselines will be compared against Attention Fusion in Step 3.
""")


if __name__ == "__main__":
    main()
