"""
================================================================================
Model Evaluation Pipeline
================================================================================
Phase 5: Evaluation & Validation
- Test set evaluation
- Multi-anchor validation (CDR, MMSE, Diagnosis)
- Cross-dataset validation (when ADNI available)
- Interpretability analysis

Part of: Master Research Plan - Phase 5
================================================================================
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Optional
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, r2_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from models.multimodal_fusion import HybridMultimodalModel, create_model
from training.trainer import MultimodalDataset, custom_collate_fn
from torch.utils.data import DataLoader


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model_path: Path, device: torch.device):
        self.device = device
        self.model = self.load_model(model_path)
        self.results = {}
    
    def load_model(self, model_path: Path) -> HybridMultimodalModel:
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        model = create_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Loaded model from: {model_path}")
        print(f"Config: {config}")
        
        return model
    
    def evaluate_dataset(self, 
                        dataset: MultimodalDataset,
                        data_loader: DataLoader,
                        dataset_name: str = "Test") -> Dict:
        """Evaluate model on a dataset"""
        print("=" * 80)
        print(f"EVALUATING ON {dataset_name.upper()} SET")
        print("=" * 80)
        
        all_predictions = {
            'cdr': [], 'mmse': [], 'diagnosis': [], 'binary': []
        }
        all_targets = {
            'cdr': [], 'mmse': [], 'diagnosis': [], 'binary': []
        }
        all_attention_weights = []
        all_embeddings = []
        
        with torch.no_grad():
            for batch in data_loader:
                anatomical = batch['anatomical'].to(self.device)
                clinical = batch['clinical'].to(self.device)
                cnn_emb = batch['cnn_embeddings'].to(self.device) if batch['cnn_embeddings'] is not None else None
                targets_list = batch['targets']
                
                # Get predictions
                predictions = self.model(
                    anatomical=anatomical,
                    clinical=clinical,
                    cnn_embeddings=cnn_emb
                )
                
                # Get embeddings for analysis
                embeddings = self.model.get_embeddings(
                    anatomical=anatomical,
                    clinical=clinical,
                    cnn_embeddings=cnn_emb
                )
                
                # Store predictions
                for task in ['cdr', 'mmse', 'diagnosis', 'binary']:
                    if task in predictions:
                        pred = predictions[task].cpu().numpy()
                        all_predictions[task].append(pred)
                
                # Store targets
                for target_dict in targets_list:
                    for task in ['cdr', 'mmse', 'diagnosis', 'binary']:
                        val = target_dict.get(task, -1)
                        if val >= 0:
                            all_targets[task].append(val)
                
                # Store attention weights
                if 'attention_weights' in predictions:
                    all_attention_weights.append(predictions['attention_weights'].cpu().numpy())
                
                # Store embeddings
                all_embeddings.append(embeddings['fused_embedding'].cpu().numpy())
        
        # Concatenate predictions
        for task in all_predictions:
            if all_predictions[task]:
                all_predictions[task] = np.concatenate(all_predictions[task], axis=0)
        
        # Compute metrics
        metrics = self.compute_metrics(all_predictions, all_targets)
        
        # Store results
        self.results[dataset_name] = {
            'predictions': all_predictions,
            'targets': all_targets,
            'metrics': metrics,
            'attention_weights': np.concatenate(all_attention_weights, axis=0) if all_attention_weights else None,
            'embeddings': np.concatenate(all_embeddings, axis=0) if all_embeddings else None
        }
        
        # Print metrics
        print(f"\n{dataset_name} Set Metrics:")
        print("-" * 80)
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print("=" * 80 + "\n")
        
        return metrics
    
    def compute_metrics(self, predictions: Dict, targets: Dict) -> Dict:
        """Compute comprehensive evaluation metrics"""
        metrics = {}
        
        # CDR regression
        if 'cdr' in predictions and len(targets['cdr']) > 0:
            cdr_pred = predictions['cdr'].squeeze()
            cdr_target = np.array(targets['cdr'])
            if len(cdr_pred) == len(cdr_target):
                metrics['cdr_mae'] = mean_absolute_error(cdr_target, cdr_pred)
                metrics['cdr_rmse'] = np.sqrt(np.mean((cdr_target - cdr_pred) ** 2))
                metrics['cdr_r2'] = r2_score(cdr_target, cdr_pred)
                metrics['cdr_correlation'] = np.corrcoef(cdr_target, cdr_pred)[0, 1]
        
        # MMSE regression
        if 'mmse' in predictions and len(targets['mmse']) > 0:
            mmse_pred = predictions['mmse'].squeeze()
            mmse_target = np.array(targets['mmse'])
            if len(mmse_pred) == len(mmse_target):
                metrics['mmse_mae'] = mean_absolute_error(mmse_target, mmse_pred)
                metrics['mmse_rmse'] = np.sqrt(np.mean((mmse_target - mmse_pred) ** 2))
                metrics['mmse_r2'] = r2_score(mmse_target, mmse_pred)
                metrics['mmse_correlation'] = np.corrcoef(mmse_target, mmse_pred)[0, 1]
        
        # Binary classification
        if 'binary' in predictions and len(targets['binary']) > 0:
            binary_pred = np.argmax(predictions['binary'], axis=1)
            binary_target = np.array(targets['binary'])
            if len(binary_pred) == len(binary_target):
                metrics['binary_accuracy'] = accuracy_score(binary_target, binary_pred)
                metrics['binary_precision'] = precision_score(binary_target, binary_pred, average='weighted', zero_division=0)
                metrics['binary_recall'] = recall_score(binary_target, binary_pred, average='weighted', zero_division=0)
                metrics['binary_f1'] = f1_score(binary_target, binary_pred, average='weighted', zero_division=0)
                
                # AUC-ROC
                try:
                    binary_probs = torch.softmax(torch.from_numpy(predictions['binary']), dim=1).numpy()[:, 1]
                    metrics['binary_auc'] = roc_auc_score(binary_target, binary_probs)
                except:
                    metrics['binary_auc'] = 0.0
        
        # Diagnosis classification
        if 'diagnosis' in predictions and len(targets['diagnosis']) > 0:
            diag_pred = np.argmax(predictions['diagnosis'], axis=1)
            diag_target = np.array(targets['diagnosis'])
            if len(diag_pred) == len(diag_target):
                metrics['diagnosis_accuracy'] = accuracy_score(diag_target, diag_pred)
                metrics['diagnosis_precision'] = precision_score(diag_target, diag_pred, average='weighted', zero_division=0)
                metrics['diagnosis_recall'] = recall_score(diag_target, diag_pred, average='weighted', zero_division=0)
                metrics['diagnosis_f1'] = f1_score(diag_target, diag_pred, average='weighted', zero_division=0)
        
        return metrics
    
    def analyze_attention_weights(self, dataset_name: str = "Test"):
        """Analyze attention weights for interpretability"""
        if dataset_name not in self.results:
            print(f"No results for {dataset_name}")
            return
        
        attention = self.results[dataset_name]['attention_weights']
        if attention is None:
            print("No attention weights available")
            return
        
        print("=" * 80)
        print("ATTENTION WEIGHTS ANALYSIS")
        print("=" * 80)
        
        # Average attention across samples
        avg_attention = attention.mean(axis=0)  # (3, 3) - modalities x modalities
        
        print("\nAverage Attention Weights (Modalities):")
        print("  MRI | Anatomical | Clinical")
        print("-" * 40)
        modality_names = ['MRI', 'Anatomical', 'Clinical']
        for i, name in enumerate(modality_names):
            print(f"{name:10} | {avg_attention[i, 0]:.3f} | {avg_attention[i, 1]:.3f} | {avg_attention[i, 2]:.3f}")
        
        # Which modality is most attended to?
        modality_importance = avg_attention.sum(axis=0)
        print(f"\nModality Importance (sum of attention):")
        for i, name in enumerate(modality_names):
            print(f"  {name}: {modality_importance[i]:.3f}")
        
        print("=" * 80 + "\n")
        
        return avg_attention
    
    def save_results(self, output_dir: Path):
        """Save evaluation results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = output_dir / "evaluation_metrics.json"
        metrics_dict = {}
        for k, v in self.results.items():
            if 'metrics' in v:
                metrics_dict[k] = {}
                for m, val in v['metrics'].items():
                    try:
                        metrics_dict[k][m] = float(val)
                    except (TypeError, ValueError):
                        metrics_dict[k][m] = str(val)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")
        
        # Save predictions
        predictions_file = output_dir / "predictions.npz"
        predictions_dict = {}
        for dataset_name, results in self.results.items():
            for task in ['cdr', 'mmse', 'diagnosis', 'binary']:
                if task in results['predictions']:
                    key = f"{dataset_name}_{task}_pred"
                    predictions_dict[key] = results['predictions'][task]
                if task in results['targets']:
                    key = f"{dataset_name}_{task}_target"
                    predictions_dict[key] = np.array(results['targets'][task])
        
        np.savez(predictions_file, **predictions_dict)
        print(f"Predictions saved to: {predictions_file}")


def main():
    """Main evaluation execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--output', type=str, default='project/results/evaluation', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    model_path = Path(args.model)
    data_path = Path(args.data)
    output_dir = Path(args.output)
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path, device)
    
    # Load test data
    df = pd.read_csv(data_path)
    
    # Create dataset (simplified - would need full setup)
    # For now, this is a template
    
    print("Evaluation pipeline ready!")
    print("Use this after training completes to evaluate the model")


if __name__ == "__main__":
    main()

