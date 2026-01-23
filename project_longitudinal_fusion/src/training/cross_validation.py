"""
Cross-Validation Module
=======================
Stratified K-Fold cross-validation with proper subject-level splits.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CV_CONFIG, RANDOM_SEED, CHECKPOINTS_DIR

from ..data.dataset import MultimodalFusionDataset
from ..models.fusion_model import MultimodalTransformerFusion, get_model
from .trainer import Trainer


class StratifiedKFoldCV:
    """
    Stratified K-Fold Cross-Validation.
    
    Ensures:
    - Subject-level splits (no leakage)
    - Stratified by label
    - Reproducible with fixed seed
    """
    
    def __init__(
        self,
        n_folds: int = CV_CONFIG['n_folds'],
        shuffle: bool = CV_CONFIG['shuffle'],
        random_state: int = RANDOM_SEED
    ):
        self.n_folds = n_folds
        self.kfold = StratifiedKFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=random_state
        )
        
    def split(
        self,
        data: Dict[str, np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/val indices for each fold.
        
        Args:
            data: Dictionary with 'labels' key
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        labels = data['labels']
        indices = np.arange(len(labels))
        
        splits = []
        for train_idx, val_idx in self.kfold.split(indices, labels):
            splits.append((train_idx, val_idx))
            
        return splits


class CrossValidator:
    """
    Full cross-validation pipeline.
    
    Trains multiple folds and aggregates results.
    """
    
    def __init__(
        self,
        model_factory: Callable[[], torch.nn.Module],
        n_folds: int = CV_CONFIG['n_folds'],
        batch_size: int = 32,
        device: str = 'cuda',
        checkpoint_dir: Optional[Path] = None,
        random_state: int = RANDOM_SEED
    ):
        self.model_factory = model_factory
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.device = device
        self.checkpoint_dir = checkpoint_dir or CHECKPOINTS_DIR
        self.random_state = random_state
        
        self.cv_splitter = StratifiedKFoldCV(
            n_folds=n_folds,
            random_state=random_state
        )
        
        self.fold_results = []
        
    def run(
        self,
        data: Dict[str, np.ndarray],
        epochs: int = 100,
        patience: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Run full cross-validation.
        
        Args:
            data: Full dataset dictionary
            epochs: Max epochs per fold
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            Dictionary with CV results
        """
        print("="*60)
        print(f"CROSS-VALIDATION ({self.n_folds}-FOLD)")
        print("="*60)
        
        splits = self.cv_splitter.split(data)
        all_val_preds = []
        all_val_labels = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\n{'='*60}")
            print(f"FOLD {fold_idx + 1}/{self.n_folds}")
            print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
            print("="*60)
            
            # Create datasets
            train_data = {k: v[train_idx] for k, v in data.items()}
            val_data = {k: v[val_idx] for k, v in data.items()}
            
            train_dataset = MultimodalFusionDataset.from_dict(train_data)
            val_dataset = MultimodalFusionDataset.from_dict(val_data)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            
            # Create fresh model
            model = self.model_factory()
            
            # Compute class weights
            train_labels = train_data['labels']
            class_counts = np.bincount(train_labels)
            class_weights = torch.FloatTensor(
                len(train_labels) / (len(class_counts) * class_counts)
            )
            
            # Create trainer
            fold_checkpoint_dir = self.checkpoint_dir / f"fold_{fold_idx + 1}"
            
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                epochs=epochs,
                patience=patience,
                checkpoint_dir=fold_checkpoint_dir,
                class_weights=class_weights
            )
            
            # Train
            fold_result = trainer.train(verbose=verbose)
            
            # Get predictions on validation set
            best_model = trainer.get_best_model()
            val_preds, val_labels = self._get_predictions(best_model, val_loader)
            
            all_val_preds.extend(val_preds)
            all_val_labels.extend(val_labels)
            
            fold_result['fold'] = fold_idx + 1
            fold_result['val_predictions'] = val_preds
            fold_result['val_labels'] = val_labels
            self.fold_results.append(fold_result)
            
            print(f"\n✓ Fold {fold_idx + 1} complete: Val AUC = {fold_result['best_val_auc']:.4f}")
        
        # Aggregate results
        return self._aggregate_results(all_val_preds, all_val_labels)
    
    @torch.no_grad()
    def _get_predictions(
        self,
        model: torch.nn.Module,
        loader: DataLoader
    ) -> Tuple[List[float], List[int]]:
        """Get predictions for a dataloader."""
        model.eval()
        all_preds = []
        all_labels = []
        
        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            labels = batch.pop('label')
            # Remove non-model keys
            batch.pop('subject_id', None)
            
            outputs = model(**batch)
            probs = outputs['probabilities'][:, 1].cpu().numpy()
            
            all_preds.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
        
        return all_preds, all_labels
    
    def _aggregate_results(
        self,
        all_preds: List[float],
        all_labels: List[int]
    ) -> Dict:
        """Aggregate cross-validation results."""
        from sklearn.metrics import roc_auc_score, accuracy_score
        from scipy import stats
        
        # Fold-level AUCs
        fold_aucs = [r['best_val_auc'] for r in self.fold_results]
        
        # Overall AUC (all folds combined)
        overall_auc = roc_auc_score(all_labels, all_preds)
        
        # Predictions
        preds_binary = (np.array(all_preds) > 0.5).astype(int)
        overall_accuracy = accuracy_score(all_labels, preds_binary)
        
        # Statistics
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        
        # 95% CI using t-distribution
        ci = stats.t.interval(
            0.95,
            len(fold_aucs) - 1,
            loc=mean_auc,
            scale=stats.sem(fold_aucs)
        )
        
        results = {
            'fold_aucs': fold_aucs,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'ci_95_lower': ci[0],
            'ci_95_upper': ci[1],
            'overall_auc': overall_auc,
            'overall_accuracy': overall_accuracy,
            'n_folds': self.n_folds,
            'total_samples': len(all_labels),
            'fold_results': self.fold_results
        }
        
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS")
        print("="*60)
        print(f"Fold AUCs: {[f'{x:.4f}' for x in fold_aucs]}")
        print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"Overall AUC: {overall_auc:.4f}")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print("="*60)
        
        return results
    
    def save_results(self, path: Path):
        """Save CV results to JSON."""
        results = {
            'fold_aucs': [r['best_val_auc'] for r in self.fold_results],
            'fold_epochs': [r['best_epoch'] for r in self.fold_results],
            'n_folds': self.n_folds
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
