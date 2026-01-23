"""
Script 02: Train Baseline Models
=================================
Train traditional ML baselines for comparison.

Models: Logistic Regression, Random Forest, XGBoost, MLP
Feature configs: ResNet-only, Biomarker-only, Fusion (all)

Usage:
    python scripts/02_train_baselines.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import load_and_prepare_data
from src.models.baselines import train_all_baselines, train_sklearn_baseline
from src.evaluation.metrics import compute_all_metrics, bootstrap_auc_ci
from src.utils.helpers import set_seed, save_json
from config import METRICS_DIR, RANDOM_SEED


def main():
    """Train all baseline models."""
    print("\n" + "="*70)
    print("  LONGITUDINAL MULTIMODAL FUSION - BASELINE TRAINING")
    print("="*70 + "\n")
    
    # Set seed
    set_seed()
    
    # Load data
    print("Loading data...")
    train_data, test_data, scalers = load_and_prepare_data()
    
    print(f"Train: {len(train_data['labels'])} samples")
    print(f"Test: {len(test_data['labels'])} samples")
    
    # Train all baselines
    print("\n" + "="*60)
    print("TRAINING BASELINE MODELS")
    print("="*60)
    
    results = train_all_baselines(
        train_data=train_data,
        test_data=test_data,
        random_state=RANDOM_SEED
    )
    
    # Compute detailed metrics for best models
    print("\n" + "="*60)
    print("DETAILED EVALUATION")
    print("="*60)
    
    detailed_results = {}
    
    # Focus on key configurations
    key_configs = [
        ('logistic_regression', 'resnet_only', True, False),
        ('logistic_regression', 'bio_only', False, True),
        ('logistic_regression', 'fusion', True, True),
        ('random_forest', 'fusion', True, True),
        ('xgboost', 'fusion', True, True),
    ]
    
    for model_name, config_name, use_resnet, use_bio in key_configs:
        key = f"{model_name}_{config_name}"
        print(f"\n{key}:")
        
        # Get predictions
        model, _ = train_sklearn_baseline(
            model_name=model_name,
            train_data=train_data,
            test_data=test_data,
            include_resnet=use_resnet,
            include_bio=use_bio,
            random_state=RANDOM_SEED
        )
        
        y_prob = model.predict_proba(test_data, use_resnet, use_bio)
        y_true = test_data['labels']
        
        # Compute all metrics
        metrics = compute_all_metrics(y_true, y_prob)
        
        # Bootstrap CI
        auc, ci_lower, ci_upper = bootstrap_auc_ci(y_true, y_prob)
        metrics['ci_lower'] = ci_lower
        metrics['ci_upper'] = ci_upper
        
        detailed_results[key] = metrics
        
        print(f"  AUC: {metrics['auc']:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Sensitivity: {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
    
    # Find best model
    print("\n" + "="*60)
    print("BEST BASELINE MODEL")
    print("="*60)
    
    best_key = max(detailed_results.keys(), key=lambda k: detailed_results[k]['auc'])
    best_metrics = detailed_results[best_key]
    
    print(f"\n🏆 Best: {best_key}")
    print(f"   AUC: {best_metrics['auc']:.4f} (95% CI: [{best_metrics['ci_lower']:.4f}, {best_metrics['ci_upper']:.4f}])")
    print(f"   This is the baseline to beat with deep fusion!")
    
    # Save results
    save_json(detailed_results, METRICS_DIR / 'baseline_results.json')
    
    print("\n" + "="*60)
    print("✅ BASELINE TRAINING COMPLETE")
    print("="*60)
    print(f"\nNext step: python scripts/03_train_fusion.py")


if __name__ == "__main__":
    main()
