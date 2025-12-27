"""
Longitudinal ADNI Experiment - Delta Model (Change-Based)
==========================================================
Uses CHANGE between baseline and follow-up scans.
Input: (baseline_features, followup_features, delta)

This tests whether observing change helps prediction.

Output: results/delta_model/
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
OUTPUT_DIR = r"D:\discs\project_longitudinal\results\delta_model"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class DeltaClassifier(nn.Module):
    """Classifier that uses baseline, followup, and delta features."""
    def __init__(self, input_dim=512, hidden_dim=64, dropout=0.5):
        super().__init__()
        # Input: baseline (512) + followup (512) + delta (512) = 1536
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, baseline, followup, delta):
        x = torch.cat([baseline, followup, delta], dim=1)
        return self.net(x)

def load_paired_data():
    """Load features and create pairs (baseline, followup) per subject."""
    print("Loading features...")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    
    features = data['features']
    subject_ids = data['subject_ids']
    visit_nums = data['visit_nums']
    labels = data['labels']
    splits = data['splits']
    
    print(f"Total scans: {len(features)}")
    
    # Group by subject
    pairs = []
    
    unique_subjects = np.unique(subject_ids)
    for subj in unique_subjects:
        mask = subject_ids == subj
        subj_features = features[mask]
        subj_visits = visit_nums[mask]
        subj_labels = labels[mask]
        subj_splits = splits[mask]
        
        # Sort by visit number
        order = np.argsort(subj_visits)
        subj_features = subj_features[order]
        subj_visits = subj_visits[order]
        
        # Need at least 2 visits
        if len(subj_features) < 2:
            continue
        
        # Create pair: first visit vs last visit
        baseline = subj_features[0]
        followup = subj_features[-1]
        delta = followup - baseline
        
        pairs.append({
            'subject_id': subj,
            'baseline': baseline,
            'followup': followup,
            'delta': delta,
            'label': subj_labels[0],  # Label is same for all visits
            'split': subj_splits[0],
            'num_visits': len(subj_features)
        })
    
    pairs_df = pd.DataFrame(pairs)
    print(f"Subjects with paired data: {len(pairs_df)}")
    
    # Split train/test
    train_df = pairs_df[pairs_df['split'] == 'train']
    test_df = pairs_df[pairs_df['split'] == 'test']
    
    # Extract arrays
    X_train_bl = np.stack(train_df['baseline'].values)
    X_train_fu = np.stack(train_df['followup'].values)
    X_train_delta = np.stack(train_df['delta'].values)
    y_train = train_df['label'].values.astype(int)
    
    X_test_bl = np.stack(test_df['baseline'].values)
    X_test_fu = np.stack(test_df['followup'].values)
    X_test_delta = np.stack(test_df['delta'].values)
    y_test = test_df['label'].values.astype(int)
    
    # Normalize each component separately
    scaler_bl = StandardScaler()
    scaler_fu = StandardScaler()
    scaler_delta = StandardScaler()
    
    X_train_bl = scaler_bl.fit_transform(X_train_bl)
    X_train_fu = scaler_fu.fit_transform(X_train_fu)
    X_train_delta = scaler_delta.fit_transform(X_train_delta)
    
    X_test_bl = scaler_bl.transform(X_test_bl)
    X_test_fu = scaler_fu.transform(X_test_fu)
    X_test_delta = scaler_delta.transform(X_test_delta)
    
    print(f"Train: {len(X_train_bl)}, Test: {len(X_test_bl)}")
    print(f"Train label dist: {np.bincount(y_train)}")
    print(f"Test label dist:  {np.bincount(y_test)}")
    
    return (X_train_bl, X_train_fu, X_train_delta, y_train,
            X_test_bl, X_test_fu, X_test_delta, y_test)

def train_model(X_train_bl, X_train_fu, X_train_delta, y_train,
                X_test_bl, X_test_fu, X_test_delta, y_test,
                epochs=100, lr=1e-3):
    """Train the delta classifier."""
    
    # Convert to tensors
    train_bl_t = torch.FloatTensor(X_train_bl)
    train_fu_t = torch.FloatTensor(X_train_fu)
    train_delta_t = torch.FloatTensor(X_train_delta)
    train_y_t = torch.LongTensor(y_train)
    
    test_bl_t = torch.FloatTensor(X_test_bl).to(DEVICE)
    test_fu_t = torch.FloatTensor(X_test_fu).to(DEVICE)
    test_delta_t = torch.FloatTensor(X_test_delta).to(DEVICE)
    
    # Create dataloader
    train_dataset = TensorDataset(train_bl_t, train_fu_t, train_delta_t, train_y_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = DeltaClassifier(input_dim=X_train_bl.shape[1]).to(DEVICE)
    
    # Class weights
    class_counts = np.bincount(y_train)
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
        
        for bl, fu, delta, y in train_loader:
            bl, fu, delta, y = bl.to(DEVICE), fu.to(DEVICE), delta.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(bl, fu, delta)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_bl_t, test_fu_t, test_delta_t)
            test_probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
        
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
        test_outputs = model(test_bl_t, test_fu_t, test_delta_t)
        test_probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
        test_preds = test_outputs.argmax(dim=1).cpu().numpy()
    
    auc = roc_auc_score(y_test, test_probs)
    auprc = average_precision_score(y_test, test_probs)
    acc = accuracy_score(y_test, test_preds)
    
    return model, {
        'auc': auc,
        'auprc': auprc,
        'accuracy': acc
    }

def bootstrap_ci(y_true, y_probs, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence intervals."""
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
    print("DELTA MODEL (CHANGE-BASED)")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Load data
    data = load_paired_data()
    X_train_bl, X_train_fu, X_train_delta, y_train = data[:4]
    X_test_bl, X_test_fu, X_test_delta, y_test = data[4:]
    
    # Train model
    print("\nTraining model...")
    model, metrics = train_model(
        X_train_bl, X_train_fu, X_train_delta, y_train,
        X_test_bl, X_test_fu, X_test_delta, y_test
    )
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pt'))
    
    # Print summary
    print("\n" + "="*60)
    print("DELTA MODEL RESULTS")
    print("="*60)
    print(f"AUC:      {metrics['auc']:.4f}")
    print(f"AUPRC:    {metrics['auprc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("="*60)
    
    return metrics

if __name__ == "__main__":
    main()
