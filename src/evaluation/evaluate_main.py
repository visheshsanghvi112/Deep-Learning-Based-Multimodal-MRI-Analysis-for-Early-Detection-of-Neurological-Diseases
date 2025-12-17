"""
================================================================================
Main Evaluation Script
================================================================================
Complete evaluation pipeline for trained model
- Test set evaluation
- Multi-anchor validation
- Interpretability analysis

Part of: Master Research Plan - Phase 5
================================================================================
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
from datetime import datetime

BASE_DIR = Path("D:/discs")
sys.path.append(str(BASE_DIR / "project" / "src"))

from models.multimodal_fusion import HybridMultimodalModel, create_model
from training.trainer import MultimodalDataset, custom_collate_fn
from evaluation.evaluate import ModelEvaluator
from evaluation.interpretability import InterpretabilityAnalyzer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def main():
    """Main evaluation execution"""
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print("Part of Master Research Plan - Phase 5\n")
    
    # Configuration
    results_dir = BASE_DIR / "project" / "results"
    models_dir = results_dir / "models"
    data_dir = BASE_DIR / "project" / "data" / "processed"
    
    # Find latest model
    model_files = list(models_dir.glob("multimodal_model_*.pt"))
    if not model_files:
        print("ERROR: No trained model found!")
        print(f"Expected in: {models_dir}")
        return
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading model: {latest_model.name}\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    checkpoint = torch.load(latest_model, map_location=device)
    config = checkpoint['config']
    feature_cols = checkpoint.get('feature_cols', [])
    clinical_cols = checkpoint.get('clinical_cols', [])
    
    # Create evaluator
    evaluator = ModelEvaluator(latest_model, device)
    
    # Load data
    # Prefer the full deep feature matrix, fall back if needed
    features_file = data_dir / "oasis_complete_features_full.csv"
    if not features_file.exists():
        features_file = data_dir / "oasis_features_normalized.csv"
    if not features_file.exists():
        features_file = data_dir / "oasis_complete_features.csv"
    
    df = pd.read_csv(features_file)
    print(f"Loaded data: {df.shape}\n")
    print(f"Using feature file: {features_file}\n")
    
    # Phase 5 for Phase 4b model: disable CNN modality, use zero embeddings
    cnn_embeddings = np.zeros((len(df), config.get('mri_embedding_dim', 512)), dtype=np.float32)
    print(f"Using zero CNN embeddings placeholder for evaluation: {cnn_embeddings.shape}\n")
    
    # Identify columns
    exclude_cols = ['SUBJECT_ID', 'dataset', 'disc_folder', 'CDR', 'MMSE', 
                    'label_cdr', 'label_binary', 'label_diagnosis', 'AGE', 'Age', 'age']
    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    
    if not clinical_cols:
        clinical_cols = []
        for col in ['AGE', 'Age', 'age', 'GENDER', 'Gender', 'M/F', 'EDUC', 'Education']:
            if col in df.columns:
                clinical_cols.append(col)
    
    age_col = 'AGE' if 'AGE' in df.columns else ('Age' if 'Age' in df.columns else 'age')
    
    # Split data (same as training)
    stratify_col = None
    if 'CDR' in df.columns:
        df['stratify'] = (df['CDR'] > 0).astype(int)
        stratify_col = 'stratify'
    
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42,
        stratify=df[stratify_col] if stratify_col else None
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42,
        stratify=temp_df[stratify_col] if stratify_col else None
    )
    
    print(f"Test set: {len(test_df)} samples\n")
    
    # Create test dataset
    test_cnn = cnn_embeddings[test_df.index] if cnn_embeddings is not None else None
    test_dataset = MultimodalDataset(
        test_df, cnn_embeddings=test_cnn,
        feature_cols=feature_cols, clinical_cols=clinical_cols, age_col=age_col
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
    
    # Evaluate
    print("=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80 + "\n")
    
    test_metrics = evaluator.evaluate_dataset(test_dataset, test_loader, "Test")
    
    # Interpretability analysis
    print("=" * 80)
    print("INTERPRETABILITY ANALYSIS")
    print("=" * 80 + "\n")
    
    if 'Test' in evaluator.results:
        results = evaluator.results['Test']
        
        # Attention weights
        if results['attention_weights'] is not None:
            analyzer = InterpretabilityAnalyzer(evaluator.model, device)
            attention_path = results_dir / "figures" / "attention_weights.png"
            attention_path.parent.mkdir(parents=True, exist_ok=True)
            analyzer.analyze_attention_weights(results['attention_weights'], attention_path)
        
        # Embedding visualization
        if results['embeddings'] is not None and len(results['targets']['binary']) > 0:
            embeddings_path = results_dir / "figures" / "embeddings_tsne.png"
            labels = np.array(results['targets']['binary'])
            if len(labels) == len(results['embeddings']):
                analyzer.visualize_embeddings(
                    results['embeddings'],
                    labels,
                    embeddings_path,
                    method='tsne'
                )
    
    # Save results
    eval_output_dir = results_dir / "evaluation"
    evaluator.save_results(eval_output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nTest Set Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nResults saved to: {eval_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

