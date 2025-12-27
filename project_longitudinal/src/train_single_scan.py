"""
Longitudinal ADNI Experiment - Single Scan Model (Baseline Comparison)
=======================================================================
Uses only BASELINE scan per subject (like the cross-sectional approach).
This serves as the comparison point for multi-scan models.

Output: results/single_scan/
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import json

# Configuration
FEATURES_PATH = r"D:\discs\project_longitudinal\data\features\longitudinal_features.npz"
OUTPUT_DIR = r"D:\discs\project_longitudinal\results\single_scan"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleClassifier(nn.Module):
    """Simple MLP classifier for single-scan prediction."""
    def __init__(self, input_dim=512, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.net(x)

def load_baseline_data():
    """Load features and filter to first visit only (baseline)."""
    print("Loading features...")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    
    features = data['features']
    subject_ids = data['subject_ids']
    visit_nums = data['visit_nums']
    labels = data['labels']
    splits = data['splits']
    
    print(f"Total scans: {len(features)}")
    
    # Filter to baseline only (visit_num == 1)
    baseline_mask = visit_nums == 1
    
    features_bl = features[baseline_mask]
    subject_ids_bl = subject_ids[baseline_mask]
    labels_bl = labels[baseline_mask]
    splits_bl = splits[baseline_mask]
    
    print(f"Baseline scans: {len(features_bl)}")
    
    # Split train/test
    train_mask = splits_bl == 'train'
    test_mask = splits_bl == 'test'
    
    X_train = features_bl[train_mask]
    y_train = labels_bl[train_mask]
    X_test = features_bl[test_mask]
    y_test = labels_bl[test_mask]
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train label dist: {np.bincount(y_train.astype(int))}")
    print(f"Test label dist:  {np.bincount(y_test.astype(int))}")
    
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, X_test, y_test, epochs=100, lr=1e-3):
    """Train the single-scan classifier."""
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train.astype(int))
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test.astype(int))
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = SimpleClassifier(input_dim=X_train.shape[1]).to(DEVICE)
    
    # Class weights for imbalance
    class_counts = np.bincount(y_train.astype(int))
    class_weights = torch.FloatTensor(1.0 / class_counts).to(DEVICE)
    class_weights = class_weights / class_weights.sum()
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training loop
    best_auc = 0
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t.to(DEVICE))
            test_probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
            test_preds = test_outputs.argmax(dim=1).cpu().numpy()
        
        auc = roc_auc_score(y_test, test_probs)
        
        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - AUC: {auc:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t.to(DEVICE))
        test_probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
        test_preds = test_outputs.argmax(dim=1).cpu().numpy()
    
    auc = roc_auc_score(y_test, test_probs)
    auprc = average_precision_score(y_test, test_probs)
    acc = accuracy_score(y_test, test_preds)
    
    return model, {
        'auc': auc,
        'auprc': auprc,
        'accuracy': acc,
        'test_probs': test_probs.tolist(),
        'test_labels': y_test.tolist()
    }

def bootstrap_ci(y_true, y_probs, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence intervals for AUC."""
    aucs = []
    n = len(y_true)
    
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[indices], y_probs[indices]))
    
    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)
    
    return lower, upper

def main():
    print("="*60)
    print("SINGLE-SCAN MODEL (BASELINE COMPARISON)")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Load data
    X_train, y_train, X_test, y_test = load_baseline_data()
    
    # Train model
    print("\nTraining model...")
    model, metrics = train_model(X_train, y_train, X_test, y_test)
    
    # Bootstrap CI
    print("\nComputing bootstrap confidence intervals...")
    y_probs = np.array(metrics['test_probs'])
    y_true = np.array(metrics['test_labels'])
    ci_lower, ci_upper = bootstrap_ci(y_true, y_probs)
    
    metrics['auc_ci_lower'] = ci_lower
    metrics['auc_ci_upper'] = ci_upper
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump({k: v for k, v in metrics.items() if k not in ['test_probs', 'test_labels']}, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pt'))
    
    # Print summary
    print("\n" + "="*60)
    print("SINGLE-SCAN MODEL RESULTS")
    print("="*60)
    print(f"AUC:      {metrics['auc']:.4f} ({ci_lower:.4f} - {ci_upper:.4f})")
    print(f"AUPRC:    {metrics['auprc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("="*60)
    
    return metrics

if __name__ == "__main__":
    main()
