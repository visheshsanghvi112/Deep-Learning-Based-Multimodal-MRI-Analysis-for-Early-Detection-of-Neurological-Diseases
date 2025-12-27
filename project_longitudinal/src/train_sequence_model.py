"""
Longitudinal ADNI Experiment - Sequence Model (Full Temporal)
==============================================================
Uses ALL scans per subject as a temporal sequence.
LSTM/GRU to model disease progression trajectory.

Output: results/sequence_model/
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import json

# Configuration
FEATURES_PATH = r"D:\discs\project_longitudinal\data\features\longitudinal_features.npz"
OUTPUT_DIR = r"D:\discs\project_longitudinal\results\sequence_model"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SequenceDataset(Dataset):
    """Dataset for variable-length sequences."""
    def __init__(self, sequences, labels):
        self.sequences = sequences  # List of (seq_len, feature_dim) arrays
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), self.labels[idx]

def collate_fn(batch):
    """Custom collate to handle variable-length sequences."""
    sequences, labels = zip(*batch)
    lengths = torch.LongTensor([len(s) for s in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, lengths, torch.LongTensor(labels)

class LSTMClassifier(nn.Module):
    """LSTM-based sequence classifier."""
    def __init__(self, input_dim=512, hidden_dim=64, num_layers=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)  # * 2 for bidirectional
    
    def forward(self, x, lengths):
        # Pack sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(packed)
        
        # Use final hidden states from both directions
        # h_n shape: (num_layers * num_directions, batch, hidden_dim)
        h_forward = h_n[-2, :, :]  # Last layer, forward
        h_backward = h_n[-1, :, :]  # Last layer, backward
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        
        out = self.dropout(h_combined)
        out = self.fc(out)
        return out

def load_sequence_data():
    """Load features as sequences per subject."""
    print("Loading features...")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    
    features = data['features']
    subject_ids = data['subject_ids']
    visit_nums = data['visit_nums']
    labels = data['labels']
    splits = data['splits']
    
    print(f"Total scans: {len(features)}")
    
    # Fit scaler on all training data
    train_mask = splits == 'train'
    scaler = StandardScaler()
    scaler.fit(features[train_mask])
    features_scaled = scaler.transform(features)
    
    # Group by subject into sequences
    sequences_train = []
    labels_train = []
    sequences_test = []
    labels_test = []
    
    unique_subjects = np.unique(subject_ids)
    
    for subj in unique_subjects:
        mask = subject_ids == subj
        subj_features = features_scaled[mask]
        subj_visits = visit_nums[mask]
        subj_labels = labels[mask]
        subj_splits = splits[mask]
        
        # Sort by visit number
        order = np.argsort(subj_visits)
        subj_features = subj_features[order]
        
        # Get split (same for all visits of a subject)
        split = subj_splits[0]
        label = int(subj_labels[0])
        
        if split == 'train':
            sequences_train.append(subj_features)
            labels_train.append(label)
        else:
            sequences_test.append(subj_features)
            labels_test.append(label)
    
    print(f"Train sequences: {len(sequences_train)}")
    print(f"Test sequences:  {len(sequences_test)}")
    
    # Sequence length stats
    train_lens = [len(s) for s in sequences_train]
    test_lens = [len(s) for s in sequences_test]
    print(f"Train seq lengths - min: {min(train_lens)}, max: {max(train_lens)}, mean: {np.mean(train_lens):.1f}")
    
    return sequences_train, labels_train, sequences_test, labels_test

def train_model(sequences_train, labels_train, sequences_test, labels_test,
                epochs=100, lr=1e-3):
    """Train the sequence classifier."""
    
    # Create datasets
    train_dataset = SequenceDataset(sequences_train, labels_train)
    test_dataset = SequenceDataset(sequences_test, labels_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Get feature dimension
    input_dim = sequences_train[0].shape[1]
    
    # Initialize model
    model = LSTMClassifier(input_dim=input_dim).to(DEVICE)
    
    # Class weights
    class_counts = np.bincount(labels_train)
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
        
        for seqs, lengths, y in train_loader:
            seqs, y = seqs.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(seqs, lengths)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate
        model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for seqs, lengths, y in test_loader:
                seqs = seqs.to(DEVICE)
                outputs = model(seqs, lengths)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y.numpy())
        
        auc = roc_auc_score(all_labels, all_probs)
        
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
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for seqs, lengths, y in test_loader:
            seqs = seqs.to(DEVICE)
            outputs = model(seqs, lengths)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    auc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    
    return model, {
        'auc': auc,
        'auprc': auprc,
        'accuracy': acc
    }

def main():
    print("="*60)
    print("SEQUENCE MODEL (LSTM)")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Load data
    sequences_train, labels_train, sequences_test, labels_test = load_sequence_data()
    
    # Train model
    print("\nTraining model...")
    model, metrics = train_model(
        sequences_train, labels_train,
        sequences_test, labels_test
    )
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pt'))
    
    # Print summary
    print("\n" + "="*60)
    print("SEQUENCE MODEL RESULTS")
    print("="*60)
    print(f"AUC:      {metrics['auc']:.4f}")
    print(f"AUPRC:    {metrics['auprc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("="*60)
    
    return metrics

if __name__ == "__main__":
    main()
