"""
Script 04: Evaluate All Models
==============================
Comprehensive evaluation comparing all models with statistical tests.

Usage:
    python scripts/04_evaluate_all.py
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import load_and_prepare_data
from src.data.dataset import create_dataloaders
from src.models.fusion_model import MultimodalTransformerFusion
from src.models.baselines import train_sklearn_baseline
from src.evaluation.metrics import (
    compute_all_metrics, bootstrap_auc_ci, delong_test, compare_models
)
from src.utils.helpers import set_seed, get_device, save_json, load_json
from config import CHECKPOINTS_DIR, METRICS_DIR, MODEL_CONFIG, RANDOM_SEED


def get_fusion_predictions(model, test_loader, device):
    """Get predictions from fusion model."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            labels = batch.pop('label')
            batch.pop('subject_id', None)  # Remove non-model keys
            
            outputs = model(**batch)
            probs = outputs['probabilities'][:, 1].cpu().numpy()
            
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


def main():
    """Complete evaluation pipeline."""
    print("\n" + "="*70)
    print("  LONGITUDINAL MULTIMODAL FUSION - COMPREHENSIVE EVALUATION")
    print("="*70 + "\n")
    
    # Set seed
    set_seed()
    device = get_device()
    
    # Load data
    print("Loading data...")
    train_data, test_data, scalers = load_and_prepare_data()
    _, test_loader = create_dataloaders(train_data, test_data, batch_size=32)
    
    y_true = test_data['labels']
    
    # Collect predictions from all models
    predictions = {}
    
    # 1. Baselines - ResNet only
    print("\n[1/6] Evaluating ResNet-only baseline...")
    model, _ = train_sklearn_baseline(
        'logistic_regression', train_data, test_data,
        include_resnet=True, include_bio=False,
        random_state=RANDOM_SEED
    )
    predictions['ResNet-only'] = model.predict_proba(test_data, True, False)
    
    # 2. Baselines - Biomarker only
    print("[2/6] Evaluating Biomarker-only baseline...")
    model, _ = train_sklearn_baseline(
        'logistic_regression', train_data, test_data,
        include_resnet=False, include_bio=True,
        random_state=RANDOM_SEED
    )
    predictions['Biomarker-only'] = model.predict_proba(test_data, False, True)
    
    # 3. Baselines - LogReg Fusion
    print("[3/6] Evaluating LogReg Fusion...")
    model, _ = train_sklearn_baseline(
        'logistic_regression', train_data, test_data,
        include_resnet=True, include_bio=True,
        random_state=RANDOM_SEED
    )
    predictions['LogReg-Fusion'] = model.predict_proba(test_data, True, True)
    
    # 4. Random Forest Fusion
    print("[4/6] Evaluating Random Forest Fusion...")
    model, _ = train_sklearn_baseline(
        'random_forest', train_data, test_data,
        include_resnet=True, include_bio=True,
        random_state=RANDOM_SEED
    )
    predictions['RF-Fusion'] = model.predict_proba(test_data, True, True)
    
    # 5. XGBoost Fusion
    print("[5/6] Evaluating XGBoost Fusion...")
    model, _ = train_sklearn_baseline(
        'xgboost', train_data, test_data,
        include_resnet=True, include_bio=True,
        random_state=RANDOM_SEED
    )
    predictions['XGBoost-Fusion'] = model.predict_proba(test_data, True, True)
    
    # 6. Deep Fusion Model
    print("[6/6] Evaluating Deep Fusion Model...")
    checkpoint_path = CHECKPOINTS_DIR / 'single_split' / 'best.pt'
    
    if checkpoint_path.exists():
        fusion_model = MultimodalTransformerFusion(
            resnet_dim=MODEL_CONFIG['resnet_dim'],
            baseline_bio_dim=9,
            followup_bio_dim=6,
            delta_bio_dim=6,
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_heads=MODEL_CONFIG['num_heads'],
            dropout=MODEL_CONFIG['dropout']
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        fusion_model.load_state_dict(checkpoint['model_state_dict'])
        
        fusion_preds, _ = get_fusion_predictions(fusion_model, test_loader, device)
        predictions['Deep-Fusion'] = fusion_preds
    else:
        print("  ⚠ Deep fusion checkpoint not found. Run 03_train_fusion.py first.")
    
    # Compare all models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    results = compare_models(y_true, predictions, n_bootstrap=1000)
    
    # Print results
    print("\n📊 RESULTS SUMMARY:\n")
    print(f"{'Model':<20} {'AUC':<8} {'95% CI':<20} {'Acc':<8} {'Sens':<8} {'Spec':<8}")
    print("-" * 80)
    
    for model_name in predictions.keys():
        if model_name in results and 'auc' in results[model_name]:
            r = results[model_name]
            ci = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            print(f"{model_name:<20} {r['auc']:<8.4f} {ci:<20} "
                  f"{r['accuracy']:<8.4f} {r['recall']:<8.4f} {r['specificity']:<8.4f}")
    
    # Statistical comparisons
    print("\n📈 STATISTICAL COMPARISONS (DeLong's Test):")
    print("-" * 60)
    
    if 'pairwise_comparisons' in results:
        for comp_name, comp_result in results['pairwise_comparisons'].items():
            sig = "✓" if comp_result['significant_at_0.05'] else "✗"
            print(f"{comp_name}: p={comp_result['p_value']:.4f} {sig}")
    
    # Identify best model
    best_model = max(
        [k for k in predictions.keys() if k in results],
        key=lambda k: results[k]['auc']
    )
    
    print("\n" + "="*60)
    print("🏆 BEST MODEL")
    print("="*60)
    print(f"\n{best_model}: AUC = {results[best_model]['auc']:.4f}")
    print(f"95% CI: [{results[best_model]['ci_lower']:.4f}, {results[best_model]['ci_upper']:.4f}]")
    
    # Check if we beat baseline
    baseline_auc = 0.83
    best_auc = results[best_model]['auc']
    
    if best_auc > baseline_auc:
        improvement = (best_auc - baseline_auc) * 100
        print(f"\n✅ SUCCESS! Beat baseline ({baseline_auc:.2f}) by +{improvement:.2f}%")
    else:
        gap = (baseline_auc - best_auc) * 100
        print(f"\n⚠ Did not beat baseline ({baseline_auc:.2f}). Gap: -{gap:.2f}%")
    
    # Save comprehensive results
    save_json(results, METRICS_DIR / 'comprehensive_evaluation.json')
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {METRICS_DIR / 'comprehensive_evaluation.json'}")
    print(f"\nNext step: python scripts/05_generate_figures.py")


if __name__ == "__main__":
    main()
