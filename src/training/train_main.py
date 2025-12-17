"""
================================================================================
Main Training Script for Hybrid Multimodal Model
================================================================================
Complete training pipeline for OASIS-1 dataset

Part of: Master Research Plan - Phase 3
================================================================================
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add paths
BASE_DIR = Path("D:/discs")
sys.path.append(str(BASE_DIR / "project" / "src"))

from models.multimodal_fusion import HybridMultimodalModel, create_model
from training.trainer import Trainer, MultimodalDataset, custom_collate_fn
from torch.utils.data import DataLoader


def load_cnn_embeddings(embeddings_file: Path) -> tuple:
    """Load pre-extracted CNN embeddings"""
    try:
        data = np.load(embeddings_file, allow_pickle=True)
        
        # Try different key names
        if 'combined_features' in data:
            embeddings = data['combined_features']
            # Extract CNN part (first 512 dims)
            if embeddings.shape[1] > 512:
                cnn_emb = embeddings[:, :512]
            else:
                cnn_emb = embeddings
        elif 'mri_features' in data:
            cnn_emb = data['mri_features']
        else:
            print("WARNING: Could not find CNN embeddings in file")
            return None, None
        
        # Get subject IDs if available
        subject_ids = None
        if 'subject_ids' in data:
            subject_ids = data['subject_ids']
        
        print(f"Loaded CNN embeddings: {cnn_emb.shape}")
        return cnn_emb, subject_ids
        
    except Exception as e:
        print(f"WARNING: Could not load CNN embeddings: {e}")
        return None, None


def main():
    """Main training execution"""
    print("=" * 80)
    print("HYBRID MULTIMODAL MODEL TRAINING")
    print("=" * 80)
    print("Part of Master Research Plan - Phase 3\n")
    
    # Configuration
    config = {
        'data_dir': BASE_DIR / "project" / "data" / "processed",
        'results_dir': BASE_DIR / "project" / "results",
        'models_dir': BASE_DIR / "project" / "results" / "models",
        
        # Model config
        'anatomical_dim': 50,  # Will be determined from data
        'clinical_dim': 6,  # Age, Gender, Education, etc.
        'mri_embedding_dim': 512,
        'anatomical_embedding_dim': 128,
        'clinical_embedding_dim': 64,
        'fusion_dim': 256,
        'num_attention_heads': 8,
        
        # Training config
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'early_stopping_patience': 15,
        
        # Loss weights
        'lambda_cdr': 1.0,
        'lambda_mmse': 1.0,
        'lambda_diagnosis': 1.0,
        'lambda_binary': 1.0,
        
        # Data split
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_seed': 42
    }
    
    # Create directories
    config['models_dir'].mkdir(parents=True, exist_ok=True)
    config['results_dir'].mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # Prefer the full deep feature matrix, fall back if needed
    features_file = config['data_dir'] / "oasis_complete_features_full.csv"
    if not features_file.exists():
        features_file = config['data_dir'] / "oasis_features_normalized.csv"
    if not features_file.exists():
        features_file = config['data_dir'] / "oasis_complete_features.csv"
    
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    df = pd.read_csv(features_file)
    print(f"Loaded features: {df.shape}")
    print(f"Using feature file: {features_file}")
    
    # For Phase 4b: disable CNN modality (anatomical + clinical only).
    # We still pass a tensor of zeros so the model interface stays unchanged,
    # but the CNN branch carries no information.
    cnn_embeddings = np.zeros((len(df), config['mri_embedding_dim']), dtype=np.float32)
    print(f"Using zero CNN embeddings placeholder: {cnn_embeddings.shape}")
    
    # Identify feature columns
    exclude_cols = ['SUBJECT_ID', 'dataset', 'disc_folder', 'CDR', 'MMSE', 
                    'label_cdr', 'label_binary', 'label_diagnosis', 'AGE', 'Age', 'age']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    
    # Clinical columns
    clinical_cols = []
    for col in ['AGE', 'Age', 'age', 'GENDER', 'Gender', 'M/F', 'EDUC', 'Education']:
        if col in df.columns:
            clinical_cols.append(col)
    
    age_col = 'AGE' if 'AGE' in df.columns else ('Age' if 'Age' in df.columns else 'age')
    
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Clinical columns: {clinical_cols}")
    print(f"Age column: {age_col}")
    
    # Update config
    config['anatomical_dim'] = len(feature_cols)
    config['clinical_dim'] = len(clinical_cols)
    
    # Data split
    print("\n" + "=" * 80)
    print("DATA SPLITTING")
    print("=" * 80)
    
    # Stratify by CDR if available
    stratify_col = None
    if 'CDR' in df.columns:
        # Create binary for stratification
        df['stratify'] = (df['CDR'] > 0).astype(int)
        stratify_col = 'stratify'
    
    train_df, temp_df = train_test_split(
        df, 
        test_size=1 - config['train_ratio'],
        random_state=config['random_seed'],
        stratify=df[stratify_col] if stratify_col else None
    )
    
    val_size = config['val_ratio'] / (config['val_ratio'] + config['test_ratio'])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_size,
        random_state=config['random_seed'],
        stratify=temp_df[stratify_col] if stratify_col else None
    )
    
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Create datasets
    print("\n" + "=" * 80)
    print("CREATING DATASETS")
    print("=" * 80)
    
    train_cnn = cnn_embeddings[train_df.index] if cnn_embeddings is not None else None
    val_cnn = cnn_embeddings[val_df.index] if cnn_embeddings is not None else None
    test_cnn = cnn_embeddings[test_df.index] if cnn_embeddings is not None else None
    
    train_dataset = MultimodalDataset(
        train_df, cnn_embeddings=train_cnn,
        feature_cols=feature_cols, clinical_cols=clinical_cols, age_col=age_col
    )
    val_dataset = MultimodalDataset(
        val_df, cnn_embeddings=val_cnn,
        feature_cols=feature_cols, clinical_cols=clinical_cols, age_col=age_col
    )
    test_dataset = MultimodalDataset(
        test_df, cnn_embeddings=test_cnn,
        feature_cols=feature_cols, clinical_cols=clinical_cols, age_col=age_col
    )
    
    # Data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate_fn)
    
    # Create model
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)
    
    model = create_model(config)
    print(f"Model created:")
    print(f"  Anatomical input: {config['anatomical_dim']}")
    print(f"  Clinical input: {config['clinical_dim']}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device, config)
    
    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")
    
    history = trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    # Save model
    model_save_path = config['models_dir'] / f"multimodal_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'feature_cols': feature_cols,
        'clinical_cols': clinical_cols
    }, model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    # Save training history
    history_file = config['results_dir'] / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_file, 'w') as f:
        json.dump({
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'val_metrics': [str(m) for m in history['val_metrics']]  # Convert to string for JSON
        }, f, indent=2)
    print(f"Training history saved to: {history_file}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nBest validation loss: {trainer.best_val_loss:.4f}")
    print(f"Model saved: {model_save_path}")


if __name__ == "__main__":
    main()

