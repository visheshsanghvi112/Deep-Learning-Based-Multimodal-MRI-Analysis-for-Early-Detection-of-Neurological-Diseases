"""
Script 03: Train Deep Fusion Model
===================================
Train the multimodal transformer fusion model with 5-fold CV.

Usage:
    python scripts/03_train_fusion.py
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
from src.models.fusion_model import MultimodalTransformerFusion, get_model
from src.training.trainer import Trainer
from src.training.cross_validation import CrossValidator
from src.evaluation.metrics import compute_all_metrics, bootstrap_auc_ci
from src.utils.helpers import set_seed, get_device, save_json, print_model_summary, count_parameters
from config import (
    TRAINING_CONFIG, MODEL_CONFIG, CV_CONFIG,
    CHECKPOINTS_DIR, METRICS_DIR, RANDOM_SEED
)


def train_single_split():
    """Train on fixed train/test split."""
    print("\n" + "="*60)
    print("TRAINING ON FIXED SPLIT")
    print("="*60)
    
    # Load data
    train_data, test_data, scalers = load_and_prepare_data()
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        train_data, test_data,
        batch_size=TRAINING_CONFIG['batch_size']
    )
    
    # Create model
    model = MultimodalTransformerFusion(
        resnet_dim=MODEL_CONFIG['resnet_dim'],
        baseline_bio_dim=9,
        followup_bio_dim=6,
        delta_bio_dim=6,
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_heads=MODEL_CONFIG['num_heads'],
        dropout=MODEL_CONFIG['dropout']
    )
    
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # Compute class weights
    train_labels = train_data['labels']
    class_counts = np.bincount(train_labels)
    class_weights = torch.FloatTensor(
        len(train_labels) / (len(class_counts) * class_counts)
    )
    
    # Create trainer
    device = get_device()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        epochs=TRAINING_CONFIG['epochs'],
        patience=TRAINING_CONFIG['patience'],
        checkpoint_dir=CHECKPOINTS_DIR / 'single_split',
        class_weights=class_weights
    )
    
    # Train
    results = trainer.train(verbose=True)
    
    # Evaluate best model
    best_model = trainer.get_best_model()
    
    # Get predictions
    all_preds, all_labels = [], []
    best_model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            labels = batch.pop('label')
            batch.pop('subject_id', None)  # Remove non-model keys
            
            outputs = best_model(**batch)
            probs = outputs['probabilities'][:, 1].cpu().numpy()
            
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    y_true = np.array(all_labels)
    y_prob = np.array(all_preds)
    
    metrics = compute_all_metrics(y_true, y_prob)
    auc, ci_lower, ci_upper = bootstrap_auc_ci(y_true, y_prob)
    
    print("\n" + "="*60)
    print("SINGLE SPLIT RESULTS")
    print("="*60)
    print(f"AUC: {auc:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Sensitivity: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    
    results['final_metrics'] = metrics
    results['auc_ci'] = {'lower': ci_lower, 'upper': ci_upper}
    
    return results


def train_cross_validation():
    """Train with 5-fold cross-validation."""
    print("\n" + "="*60)
    print("TRAINING WITH 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    # Load all data
    train_data, test_data, scalers = load_and_prepare_data()
    
    # Combine train and test for CV
    combined_data = {
        k: np.concatenate([train_data[k], test_data[k]], axis=0)
        for k in train_data.keys()
    }
    
    print(f"\nTotal samples for CV: {len(combined_data['labels'])}")
    
    # Model factory
    def create_model():
        return MultimodalTransformerFusion(
            resnet_dim=MODEL_CONFIG['resnet_dim'],
            baseline_bio_dim=9,
            followup_bio_dim=6,
            delta_bio_dim=6,
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_heads=MODEL_CONFIG['num_heads'],
            dropout=MODEL_CONFIG['dropout']
        )
    
    # Cross-validator
    device = get_device()
    cv = CrossValidator(
        model_factory=create_model,
        n_folds=CV_CONFIG['n_folds'],
        batch_size=TRAINING_CONFIG['batch_size'],
        device=device,
        checkpoint_dir=CHECKPOINTS_DIR / 'cv',
        random_state=RANDOM_SEED
    )
    
    # Run CV
    cv_results = cv.run(
        data=combined_data,
        epochs=TRAINING_CONFIG['epochs'],
        patience=TRAINING_CONFIG['patience'],
        verbose=True
    )
    
    # Save CV results
    cv.save_results(METRICS_DIR / 'cv_results.json')
    
    return cv_results


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("  LONGITUDINAL MULTIMODAL FUSION - DEEP MODEL TRAINING")
    print("="*70 + "\n")
    
    # Set seed
    set_seed()
    
    # Check device
    device = get_device()
    
    # Print model architecture
    model = MultimodalTransformerFusion(
        resnet_dim=MODEL_CONFIG['resnet_dim'],
        baseline_bio_dim=9,
        followup_bio_dim=6,
        delta_bio_dim=6,
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_heads=MODEL_CONFIG['num_heads'],
        dropout=MODEL_CONFIG['dropout']
    )
    print_model_summary(model)
    
    # Train on single split first (faster)
    single_results = train_single_split()
    
    # Save single split results
    save_json({
        'best_val_auc': single_results['best_val_auc'],
        'best_epoch': single_results['best_epoch'],
        'final_metrics': single_results['final_metrics'],
        'auc_ci': single_results['auc_ci']
    }, METRICS_DIR / 'fusion_single_split.json')
    
    # Ask if user wants CV (takes longer)
    print("\n" + "="*60)
    print("5-FOLD CROSS-VALIDATION")
    print("="*60)
    print("\nRunning 5-fold cross-validation for robust evaluation...")
    
    cv_results = train_cross_validation()
    
    # Save CV results
    save_json({
        'mean_auc': cv_results['mean_auc'],
        'std_auc': cv_results['std_auc'],
        'ci_95_lower': cv_results['ci_95_lower'],
        'ci_95_upper': cv_results['ci_95_upper'],
        'fold_aucs': cv_results['fold_aucs'],
        'overall_auc': cv_results['overall_auc'],
        'overall_accuracy': cv_results['overall_accuracy']
    }, METRICS_DIR / 'fusion_cv_results.json')
    
    # Final summary
    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    
    print("\n📊 SINGLE SPLIT RESULTS:")
    print(f"   AUC: {single_results['final_metrics']['auc']:.4f}")
    print(f"   95% CI: [{single_results['auc_ci']['lower']:.4f}, {single_results['auc_ci']['upper']:.4f}]")
    
    print("\n📊 5-FOLD CV RESULTS:")
    print(f"   Mean AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
    print(f"   95% CI: [{cv_results['ci_95_lower']:.4f}, {cv_results['ci_95_upper']:.4f}]")
    print(f"   Fold AUCs: {[f'{x:.3f}' for x in cv_results['fold_aucs']]}")
    
    print("\n✅ Next step: python scripts/04_evaluate_all.py")


if __name__ == "__main__":
    main()
