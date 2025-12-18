"""
================================================================================
Training Pipeline for Hybrid Multimodal Model
================================================================================
Includes:
- Data loading and preprocessing
- Age-adjusted training
- Multi-task loss
- Evaluation metrics
- Model checkpointing

Part of: Master Research Plan - Phase 3
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.multimodal_fusion import HybridMultimodalModel


class MultimodalDataset(Dataset):
    """Dataset for multimodal MRI analysis"""
    
    def __init__(self, 
                 features_df: pd.DataFrame,
                 cnn_embeddings: Optional[np.ndarray] = None,
                 target_cols: List[str] = ['CDR', 'MMSE'],
                 feature_cols: Optional[List[str]] = None,
                 clinical_cols: Optional[List[str]] = None,
                 age_col: str = 'AGE'):
        self.features_df = features_df.reset_index(drop=True)
        self.cnn_embeddings = cnn_embeddings
        self.target_cols = target_cols
        self.age_col = age_col
        
        # Identify feature columns
        if feature_cols is None:
            exclude = ['SUBJECT_ID', 'dataset', 'disc_folder', 'CDR', 'MMSE', 
                      'label_cdr', 'label_binary', 'label_diagnosis', age_col]
            feature_cols = [c for c in features_df.columns if c not in exclude]
        
        if clinical_cols is None:
            clinical_cols = [age_col, 'GENDER', 'EDUC'] if age_col in features_df.columns else []
            clinical_cols = [c for c in clinical_cols if c in features_df.columns]
        
        self.feature_cols = feature_cols
        self.clinical_cols = clinical_cols
        
        # Remove rows with missing targets
        valid_mask = self.features_df[target_cols].notna().all(axis=1)
        self.valid_indices = self.features_df[valid_mask].index.tolist()
        
        print(f"Dataset: {len(self.valid_indices)} valid samples out of {len(features_df)}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Clinical: {len(self.clinical_cols)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.features_df.iloc[actual_idx]
        
        # Anatomical features
        anatomical = row[self.feature_cols].values.astype(np.float32)
        anatomical = np.nan_to_num(anatomical, nan=0.0)
        
        # Clinical features
        clinical = []
        if self.clinical_cols:
            for col in self.clinical_cols:
                val = row[col]
                if pd.isna(val):
                    clinical.append(0.0)
                elif col.upper() in ['GENDER', 'M/F', 'GENDER_CODE']:
                    # Encode gender: Male=1, Female=0
                    if isinstance(val, str):
                        clinical.append(1.0 if val.upper().startswith('M') else 0.0)
                    else:
                        clinical.append(float(val))
                else:
                    try:
                        clinical.append(float(val))
                    except (ValueError, TypeError):
                        clinical.append(0.0)
        clinical = np.array(clinical, dtype=np.float32)
        
        # CNN embeddings (if available)
        cnn_emb = None
        if self.cnn_embeddings is not None:
            cnn_emb = self.cnn_embeddings[actual_idx].astype(np.float32)
        
        # Targets
        targets = {}
        for col in self.target_cols:
            if col in row:
                val = row[col]
                if pd.isna(val):
                    targets[col] = -1  # Missing value marker
                else:
                    targets[col] = float(val)
            else:
                targets[col] = -1
        
        # Additional labels for multi-task learning
        if 'CDR' in row and not pd.isna(row['CDR']):
            targets['cdr'] = float(row['CDR'])
            targets['binary'] = int(row['CDR'] > 0)  # Binary: Normal (0) vs Impaired (>0)
        else:
            targets['cdr'] = -1
            targets['binary'] = -1
        
        if 'MMSE' in row and not pd.isna(row['MMSE']):
            targets['mmse'] = float(row['MMSE'])
        else:
            targets['mmse'] = -1
        
        # Diagnosis (if available)
        if 'label_diagnosis' in row:
            diag = row['label_diagnosis']
            if pd.isna(diag):
                targets['diagnosis'] = -1
            else:
                # Map: CN=0, MCI=1, AD=2
                diag_map = {'CN': 0, 'MCI': 1, 'AD': 2, 0: 0, 1: 1, 2: 2}
                targets['diagnosis'] = diag_map.get(diag, -1)
        else:
            targets['diagnosis'] = -1
        
        return {
            'anatomical': torch.FloatTensor(anatomical),
            'clinical': torch.FloatTensor(clinical),
            'cnn_embeddings': torch.FloatTensor(cnn_emb) if cnn_emb is not None else None,
            'targets': targets,  # This will be a dict, not a list
            'age': float(row[self.age_col]) if self.age_col in row and not pd.isna(row[self.age_col]) else 0.0
        }


def custom_collate_fn(batch):
    """Custom collate function to handle targets dictionary and None values"""
    # Separate targets and handle None values
    targets_list = []
    processed_batch = []
    
    for item in batch:
        targets_list.append(item.pop('targets'))
        
        # Handle None cnn_embeddings
        if item.get('cnn_embeddings') is None:
            # Create zero tensor if None
            if 'anatomical' in item:
                batch_size = 1
                item['cnn_embeddings'] = torch.zeros(512, dtype=torch.float32)
        
        processed_batch.append(item)
    
    # Use default collate for the rest
    try:
        collated = default_collate(processed_batch)
    except TypeError as e:
        # Handle None values manually
        collated = {}
        for key in processed_batch[0].keys():
            values = [item[key] for item in processed_batch]
            if all(v is not None for v in values):
                try:
                    collated[key] = default_collate(values)
                except:
                    # If collation fails, stack manually
                    if isinstance(values[0], torch.Tensor):
                        collated[key] = torch.stack(values)
                    else:
                        collated[key] = values
            else:
                # Handle None values - create zero tensors
                if key == 'cnn_embeddings':
                    # Get shape from first non-None or use default
                    collated[key] = torch.stack([v if v is not None else torch.zeros(512, dtype=torch.float32) for v in values])
                else:
                    collated[key] = values
    
    # Add targets as list (keep as list, not collated)
    collated['targets'] = targets_list
    
    return collated


class MultiTaskLoss(nn.Module):
    """Multi-task loss function"""
    
    def __init__(self, 
                 lambda_cdr: float = 1.0,
                 lambda_mmse: float = 1.0,
                 lambda_diagnosis: float = 1.0,
                 lambda_binary: float = 1.0,
                 binary_class_weights: Optional[torch.Tensor] = None):
        super(MultiTaskLoss, self).__init__()
        
        self.lambda_cdr = lambda_cdr
        self.lambda_mmse = lambda_mmse
        self.lambda_diagnosis = lambda_diagnosis
        self.lambda_binary = lambda_binary
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Use weighted loss for binary classification if class weights provided
        if binary_class_weights is not None:
            self.binary_ce_loss = nn.CrossEntropyLoss(weight=binary_class_weights)
        else:
            self.binary_ce_loss = self.ce_loss
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        Args:
            predictions: Dictionary of model predictions
            targets: List of target dictionaries (one per sample in batch)
        """
        losses = {}
        total_loss = 0.0
        
        batch_size = len(targets)
        
        # CDR regression
        if 'cdr' in predictions:
            # Ensure predictions are 1D (batch_size,) to support batch_size=1
            cdr_pred = predictions['cdr'].view(-1)
            cdr_target = torch.tensor([t.get('cdr', -1) for t in targets], 
                                     device=cdr_pred.device, dtype=torch.float32)
            valid_mask = cdr_target >= 0
            if valid_mask.sum() > 0:
                loss_cdr = self.mse_loss(cdr_pred[valid_mask], cdr_target[valid_mask])
                losses['cdr'] = loss_cdr
                total_loss += self.lambda_cdr * loss_cdr
        
        # MMSE regression
        if 'mmse' in predictions:
            mmse_pred = predictions['mmse'].view(-1)
            mmse_target = torch.tensor([t.get('mmse', -1) for t in targets],
                                      device=mmse_pred.device, dtype=torch.float32)
            valid_mask = mmse_target >= 0
            if valid_mask.sum() > 0:
                loss_mmse = self.mse_loss(mmse_pred[valid_mask], mmse_target[valid_mask])
                losses['mmse'] = loss_mmse
                total_loss += self.lambda_mmse * loss_mmse
        
        # Diagnosis classification
        if 'diagnosis' in predictions:
            diag_pred = predictions['diagnosis']
            diag_target = torch.tensor([t.get('diagnosis', -1) for t in targets],
                                      device=diag_pred.device, dtype=torch.long)
            valid_mask = diag_target >= 0
            if valid_mask.sum() > 0:
                loss_diag = self.ce_loss(diag_pred[valid_mask], diag_target[valid_mask])
                losses['diagnosis'] = loss_diag
                total_loss += self.lambda_diagnosis * loss_diag
        
        # Binary classification
        if 'binary' in predictions:
            binary_pred = predictions['binary']
            binary_target = torch.tensor([t.get('binary', -1) for t in targets],
                                        device=binary_pred.device, dtype=torch.long)
            valid_mask = binary_target >= 0
            if valid_mask.sum() > 0:
                # Use weighted loss if class weights are provided
                loss_binary = self.binary_ce_loss(binary_pred[valid_mask], binary_target[valid_mask])
                losses['binary'] = loss_binary
                total_loss += self.lambda_binary * loss_binary
        
        losses['total'] = total_loss
        
        return losses


class Trainer:
    """Training pipeline"""
    
    def __init__(self,
                 model: HybridMultimodalModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 config: Dict):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function
        binary_class_weights = config.get('binary_class_weights', None)
        self.criterion = MultiTaskLoss(
            lambda_cdr=config.get('lambda_cdr', 1.0),
            lambda_mmse=config.get('lambda_mmse', 1.0),
            lambda_diagnosis=config.get('lambda_diagnosis', 1.0),
            lambda_binary=config.get('lambda_binary', 1.0),
            binary_class_weights=binary_class_weights
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        batch_losses = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            anatomical = batch['anatomical'].to(self.device)
            clinical = batch['clinical'].to(self.device)
            cnn_emb = batch['cnn_embeddings'].to(self.device) if batch['cnn_embeddings'] is not None else None
            targets_list = batch['targets']  # This is a list of dicts from DataLoader
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(
                anatomical=anatomical,
                clinical=clinical,
                cnn_embeddings=cnn_emb
            )
            
            # Compute loss
            losses = self.criterion(predictions, targets_list)
            loss = losses['total']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_losses.append(loss.item())
        
        avg_loss = total_loss / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'batch_losses': batch_losses
        }
    
    def validate(self) -> Dict:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        all_predictions = {'cdr': [], 'mmse': [], 'diagnosis': [], 'binary': []}
        all_targets = {'cdr': [], 'mmse': [], 'diagnosis': [], 'binary': []}
        
        with torch.no_grad():
            for batch in self.val_loader:
                anatomical = batch['anatomical'].to(self.device)
                clinical = batch['clinical'].to(self.device)
                cnn_emb = batch['cnn_embeddings'].to(self.device) if batch['cnn_embeddings'] is not None else None
                targets_list = batch['targets']  # List of dicts
                
                predictions = self.model(
                    anatomical=anatomical,
                    clinical=clinical,
                    cnn_embeddings=cnn_emb
                )
                
                losses = self.criterion(predictions, targets_list)
                total_loss += losses['total'].item()
                
                # Store predictions and targets
                for task in ['cdr', 'mmse', 'diagnosis', 'binary']:
                    if task in predictions:
                        pred = predictions[task].cpu().numpy()
                        all_predictions[task].append(pred)
                        
                        target_vals = [t.get(task, -1) for t in targets_list if isinstance(t, dict) and t.get(task, -1) >= 0]
                        if target_vals:
                            all_targets[task].extend(target_vals)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute metrics
        metrics = self.compute_metrics(all_predictions, all_targets)
        
        return {
            'loss': avg_loss,
            'metrics': metrics
        }
    
    def compute_metrics(self, predictions: Dict, targets: Dict) -> Dict:
        """Compute evaluation metrics"""
        from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score
        
        metrics = {}
        
        # CDR regression
        if 'cdr' in predictions and len(targets['cdr']) > 0:
            cdr_pred = np.concatenate([p.squeeze() for p in predictions['cdr']])
            cdr_target = np.array(targets['cdr'])
            if len(cdr_pred) == len(cdr_target):
                metrics['cdr_mae'] = mean_absolute_error(cdr_target, cdr_pred)
                metrics['cdr_r2'] = r2_score(cdr_target, cdr_pred)
        
        # MMSE regression
        if 'mmse' in predictions and len(targets['mmse']) > 0:
            mmse_pred = np.concatenate([p.squeeze() for p in predictions['mmse']])
            mmse_target = np.array(targets['mmse'])
            if len(mmse_pred) == len(mmse_target):
                metrics['mmse_mae'] = mean_absolute_error(mmse_target, mmse_pred)
                metrics['mmse_r2'] = r2_score(mmse_target, mmse_pred)
        
        # Binary classification
        if 'binary' in predictions and len(targets['binary']) > 0:
            binary_pred = np.concatenate([np.argmax(p, axis=1) for p in predictions['binary']])
            binary_target = np.array(targets['binary'])
            if len(binary_pred) == len(binary_target):
                metrics['binary_acc'] = accuracy_score(binary_target, binary_pred)
                try:
                    binary_probs = np.concatenate([F.softmax(torch.from_numpy(p), dim=1).numpy()[:, 1] 
                                                  for p in predictions['binary']])
                    metrics['binary_auc'] = roc_auc_score(binary_target, binary_probs)
                except:
                    pass
        
        return metrics
    
    def train(self, num_epochs: int = 100, early_stopping_patience: int = 15):
        """Full training loop"""
        print("=" * 80)
        print("TRAINING STARTED")
        print("=" * 80)
        print(f"Epochs: {num_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_results = self.train_epoch()
            self.history['train_loss'].append(train_results['loss'])
            
            # Validate
            val_results = self.validate()
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_metrics'].append(val_results['metrics'])
            
            # Learning rate scheduling
            self.scheduler.step(val_results['loss'])
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_results['loss']:.4f}")
            print(f"  Val Loss: {val_results['loss']:.4f}")
            if val_results['metrics']:
                for metric, value in val_results['metrics'].items():
                    print(f"  {metric}: {value:.4f}")
            
            # Save best model
            if val_results['loss'] < self.best_val_loss:
                self.best_val_loss = val_results['loss']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"  [BEST] New best model (val_loss: {self.best_val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
            
            print()
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model (val_loss: {self.best_val_loss:.4f})")
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}\n")
        
        return self.history


if __name__ == "__main__":
    print("Training pipeline ready!")
    print("Use this module to train the hybrid multimodal model")

