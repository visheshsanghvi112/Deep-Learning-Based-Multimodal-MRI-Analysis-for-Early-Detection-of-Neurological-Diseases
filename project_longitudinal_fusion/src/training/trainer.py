"""
Model Trainer
=============
Professional training loop with:
- Early stopping
- Learning rate scheduling
- Gradient clipping
- Mixed precision (optional)
- Comprehensive logging
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
from pathlib import Path
import json

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TRAINING_CONFIG, DEVICE, RANDOM_SEED


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
        return self.should_stop


class Trainer:
    """
    Professional model trainer with comprehensive features.
    
    Features:
    - Early stopping based on validation AUC
    - Learning rate scheduling (ReduceLROnPlateau)
    - Gradient clipping
    - Model checkpointing
    - Training history logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = DEVICE,
        learning_rate: float = TRAINING_CONFIG['learning_rate'],
        weight_decay: float = TRAINING_CONFIG['weight_decay'],
        epochs: int = TRAINING_CONFIG['epochs'],
        patience: int = TRAINING_CONFIG['patience'],
        lr_patience: int = TRAINING_CONFIG['lr_patience'],
        lr_factor: float = TRAINING_CONFIG['lr_factor'],
        min_lr: float = TRAINING_CONFIG['min_lr'],
        gradient_clip: float = TRAINING_CONFIG['gradient_clip'],
        checkpoint_dir: Optional[Path] = None,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = checkpoint_dir
        
        # Loss with class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, mode='max')
        
        # History
        self.history = {
            'train_loss': [],
            'train_auc': [],
            'val_loss': [],
            'val_auc': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        self.best_val_auc = 0.0
        self.best_epoch = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in self.train_loader:
            # Move to device and filter model inputs
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            labels = batch.pop('label')
            # Remove non-model keys
            batch.pop('subject_id', None)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            logits = outputs['logits']
            
            loss = self.criterion(logits, labels)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            probs = outputs['probabilities'][:, 1].detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        from sklearn.metrics import roc_auc_score
        
        avg_loss = total_loss / len(self.train_loader)
        auc = roc_auc_score(all_labels, all_preds)
        
        return {'loss': avg_loss, 'auc': auc}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            labels = batch.pop('label')
            # Remove non-model keys
            batch.pop('subject_id', None)
            
            outputs = self.model(**batch)
            logits = outputs['logits']
            
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            probs = outputs['probabilities'][:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
        
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        avg_loss = total_loss / len(self.val_loader)
        preds_binary = (np.array(all_preds) > 0.5).astype(int)
        
        return {
            'loss': avg_loss,
            'auc': roc_auc_score(all_labels, all_preds),
            'accuracy': accuracy_score(all_labels, preds_binary)
        }
    
    def save_checkpoint(self, epoch: int, val_auc: float, is_best: bool = False):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
            
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_auc': val_auc,
            'history': self.history
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['val_auc']
    
    def train(
        self,
        verbose: bool = True,
        log_interval: int = 1
    ) -> Dict:
        """
        Full training loop.
        
        Returns:
            Dictionary with training results
        """
        print("="*60)
        print("TRAINING START")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['auc'])
            
            # Log history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(current_lr)
            
            # Check for best model
            is_best = val_metrics['auc'] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics['auc']
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics['auc'], is_best)
            
            # Logging
            if verbose and epoch % log_interval == 0:
                print(f"\nEpoch {epoch}/{self.epochs}")
                print(f"  Train: Loss={train_metrics['loss']:.4f}, AUC={train_metrics['auc']:.4f}")
                print(f"  Val:   Loss={val_metrics['loss']:.4f}, AUC={val_metrics['auc']:.4f}, "
                      f"Acc={val_metrics['accuracy']:.4f}")
                print(f"  LR: {current_lr:.2e}")
                
                if is_best:
                    print(f"  ★ New best AUC: {self.best_val_auc:.4f}")
            
            # Early stopping
            if self.early_stopping(val_metrics['auc']):
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Best AUC: {self.best_val_auc:.4f} (epoch {self.best_epoch})")
        print("="*60)
        
        return {
            'best_val_auc': self.best_val_auc,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'training_time': elapsed
        }
    
    def get_best_model(self) -> nn.Module:
        """Load and return the best model."""
        if self.checkpoint_dir is not None:
            best_path = self.checkpoint_dir / 'best.pt'
            if best_path.exists():
                checkpoint = torch.load(best_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.model
