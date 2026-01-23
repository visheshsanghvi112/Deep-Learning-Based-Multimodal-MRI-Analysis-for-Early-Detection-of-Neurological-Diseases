"""
Script 05: Generate Publication Figures
========================================
Create all publication-quality figures.

Usage:
    python scripts/05_generate_figures.py
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
from src.evaluation.visualization import (
    plot_roc_curves, 
    plot_confusion_matrix,
    plot_training_history,
    plot_attention_weights,
    create_results_summary_figure
)
from src.evaluation.metrics import compute_all_metrics, bootstrap_auc_ci
from src.utils.helpers import set_seed, get_device, load_json
from config import (
    CHECKPOINTS_DIR, METRICS_DIR, FIGURES_DIR,
    MODEL_CONFIG, RANDOM_SEED
)


def main():
    """Generate all figures."""
    print("\n" + "="*70)
    print("  LONGITUDINAL MULTIMODAL FUSION - FIGURE GENERATION")
    print("="*70 + "\n")
    
    # Set seed
    set_seed()
    device = get_device()
    
    # Create figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_data, test_data, scalers = load_and_prepare_data()
    _, test_loader = create_dataloaders(train_data, test_data, batch_size=32)
    
    y_true = test_data['labels']
    
    # Collect predictions
    print("\nCollecting predictions from all models...")
    predictions = {}
    
    # Baselines
    for name, config in [
        ('ResNet-only', (True, False)),
        ('Biomarker-only', (False, True)),
        ('LogReg-Fusion', (True, True))
    ]:
        model, _ = train_sklearn_baseline(
            'logistic_regression', train_data, test_data,
            include_resnet=config[0], include_bio=config[1],
            random_state=RANDOM_SEED
        )
        predictions[name] = model.predict_proba(test_data, config[0], config[1])
    
    # XGBoost
    model, _ = train_sklearn_baseline(
        'xgboost', train_data, test_data,
        include_resnet=True, include_bio=True,
        random_state=RANDOM_SEED
    )
    predictions['XGBoost-Fusion'] = model.predict_proba(test_data, True, True)
    
    # Deep fusion
    checkpoint_path = CHECKPOINTS_DIR / 'single_split' / 'best.pt'
    attention_weights = None
    
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
        fusion_model.eval()
        
        all_preds = []
        all_mri_attn = []
        all_bio_attn = []
        all_fusion_gates = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                labels = batch.pop('label')
                batch.pop('subject_id', None)  # Remove non-model keys
                
                outputs = fusion_model(**batch, return_attention=True)
                probs = outputs['probabilities'][:, 1].cpu().numpy()
                all_preds.extend(probs)
                
                if outputs.get('mri_temporal_attention') is not None:
                    all_mri_attn.append(outputs['mri_temporal_attention'].cpu().numpy())
                if outputs.get('bio_temporal_attention') is not None:
                    all_bio_attn.append(outputs['bio_temporal_attention'].cpu().numpy())
                if outputs.get('fusion_gates') is not None:
                    mri_g, bio_g = outputs['fusion_gates']
                    all_fusion_gates.append((mri_g.cpu().numpy(), bio_g.cpu().numpy()))
        
        predictions['Deep-Fusion'] = np.array(all_preds)
        
        # Aggregate attention weights
        if all_mri_attn:
            mri_attn = np.concatenate(all_mri_attn, axis=0)
            attention_weights = mri_attn
    
    # Figure 1: ROC Curves
    print("\n[1/5] Generating ROC curves...")
    plot_roc_curves(
        y_true, predictions,
        title="MCI-to-Dementia Progression Prediction",
        save_path=FIGURES_DIR / 'roc_curves.png'
    )
    
    # Figure 2: Confusion Matrix (best model)
    print("[2/5] Generating confusion matrix...")
    if 'Deep-Fusion' in predictions:
        y_pred = (predictions['Deep-Fusion'] > 0.5).astype(int)
        plot_confusion_matrix(
            y_true, y_pred,
            title="Deep Fusion Model - Confusion Matrix",
            save_path=FIGURES_DIR / 'confusion_matrix.png'
        )
    
    # Figure 3: Training History
    print("[3/5] Generating training history plot...")
    history_path = CHECKPOINTS_DIR / 'single_split' / 'best.pt'
    
    if history_path.exists():
        checkpoint = torch.load(history_path, map_location='cpu')
        if 'history' in checkpoint:
            plot_training_history(
                checkpoint['history'],
                title="Deep Fusion Model - Training History",
                save_path=FIGURES_DIR / 'training_history.png'
            )
    
    # Figure 4: Attention Weights
    print("[4/5] Generating attention weights visualization...")
    if attention_weights is not None:
        plot_attention_weights(
            attention_weights,
            title="Temporal Attention Weights",
            save_path=FIGURES_DIR / 'attention_weights.png'
        )
    
    # Figure 5: Results Summary
    print("[5/5] Generating results summary figure...")
    results = {}
    for name, preds in predictions.items():
        metrics = compute_all_metrics(y_true, preds)
        auc, ci_lower, ci_upper = bootstrap_auc_ci(y_true, preds)
        results[name] = {
            **metrics,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    create_results_summary_figure(
        results,
        title="Longitudinal Multimodal Fusion - Summary",
        save_path=FIGURES_DIR / 'results_summary.png'
    )
    
    # Summary
    print("\n" + "="*60)
    print("✅ FIGURE GENERATION COMPLETE")
    print("="*60)
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print("\nGenerated files:")
    for f in FIGURES_DIR.glob("*.png"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
