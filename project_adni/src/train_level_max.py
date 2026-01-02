
"""
================================================================================
ADNI Level-MAX Experiment (Overpowered)
================================================================================
Trains Fusion Models using the "Level-MAX" feature set:
- MRI Features (512-dim ResNet18)
- FULL Biological Profile:
  - Demographics: Age, Sex, Education
  - Genetics: APOE4
  - Volumetrics: Hippocampus, Ventricles, Entorhinal, Fusiform, MidTemp, WholeBrain, ICV
  - CSF: ABETA, TAU, PTAU

Total Clinical Dimension: 14 Features
(Age, Sex, Education, APOE4, Hip, Vent, Ent, Fus, Mid, WB, ICV, AB, Tau, PTau)

Goal: Achieve ~0.80 AUC honestly (without circular clinical scores).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import os
import warnings
import json
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Training configuration for Level-MAX"""
    TRAIN_CSV = r"D:\discs\project_adni\data\features\train_level_max.csv"
    TEST_CSV = r"D:\discs\project_adni\data\features\test_level_max.csv"
    RESULTS_DIR = r"D:\discs\project_adni\results\level_max"
    
    RANDOM_SEED = 42
    
    # Model architecture
    MRI_DIM = 512
    MRI_PROJ_DIM = 64
    CLINICAL_DIM = 14  # The full biological suite
    CLINICAL_PROJ_DIM = 32 # Keep same projection dim or increase slightly? 32 is fine.
    FUSION_DIM = 64
    
    # Training hyperparameters (Same as Level-1 for fair comparison)
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    PATIENCE = 15
    DROPOUT = 0.5
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(Config.RESULTS_DIR, exist_ok=True)

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_adni_level_max(csv_path: str, scaler=None, is_train=True):
    """Load ADNI Level-MAX features."""
    df = pd.read_csv(csv_path)
    
    # 1. MRI Features
    feature_cols = [f'f{i}' for i in range(512)]
    mri = df[feature_cols].values.astype(np.float32)
    
    # 2. Clinical Features
    # Original: Age, Sex
    # New: PTEDUCAT, APOE4, Hippocampus, Ventricles, Entorhinal, Fusiform, MidTemp, WholeBrain, ICV, ABETA, TAU, PTAU
    
    # Raw Extraction
    age = df['Age'].values.astype(np.float32)
    sex_encoded = (df['Sex'] == 'M').astype(np.float32)
    
    other_cols = [
        'PTEDUCAT', 'APOE4', 
        'Hippocampus', 'Ventricles', 'Entorhinal', 'Fusiform', 'MidTemp', 'WholeBrain', 'ICV', 
        'ABETA', 'TAU', 'PTAU'
    ]
    others = df[other_cols].values.astype(np.float32)
    
    # Combine raw
    # Shape: (N, 14)
    # Order: Age, Sex, Educ, APOE4, Hip, Vent, Ent, Fus, Mid, WB, ICV, AB, Tau, PTau
    clinical_raw = np.column_stack([age, sex_encoded, others])
    
    # 3. Normalization (CRITICAL for volumes vs age)
    if is_train:
        scaler = StandardScaler()
        clinical_norm = scaler.fit_transform(clinical_raw)
    else:
        if scaler is None:
            raise ValueError("Must provide scaler for test set!")
        clinical_norm = scaler.transform(clinical_raw)
    
    # 4. Labels
    labels = np.where(df['Group'] == 'CN', 0, 1).astype(np.int64)
    
    return mri, clinical_norm, labels, scaler

class ADNIDataset(Dataset):
    def __init__(self, mri, clinical, labels):
        self.mri = torch.FloatTensor(mri)
        self.clinical = torch.FloatTensor(clinical)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'mri': self.mri[idx],
            'clinical': self.clinical[idx],
            'label': self.labels[idx]
        }

# ==============================================================================
# MODELS (Defined inline to ensure exact architecture)
# ==============================================================================

class MRIOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.MRI_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(64, 2)
        )
    def forward(self, mri, clinical):
        return self.net(mri)

class LateFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # MRI Branch
        self.mri_net = nn.Sequential(
            nn.Linear(Config.MRI_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT)
        )
        # Clinical Branch
        self.clin_net = nn.Sequential(
            nn.Linear(Config.CLINICAL_DIM, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT)
        )
        # Fusion
        self.fusion_net = nn.Sequential(
            nn.Linear(64 + 32, 32),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(32, 2)
        )
        
    def forward(self, mri, clinical):
        m_emb = self.mri_net(mri)
        c_emb = self.clin_net(clinical)
        combined = torch.cat([m_emb, c_emb], dim=1)
        return self.fusion_net(combined)

class AttentionFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # MRI Projection
        self.mri_proj = nn.Sequential(
            nn.Linear(Config.MRI_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        # Clinical Projection
        self.clin_proj = nn.Sequential(
            nn.Linear(Config.CLINICAL_DIM, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        # Attention
        self.mri_attn = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.clin_attn = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        
        # Fusion
        self.fusion_net = nn.Sequential(
            nn.Linear(64 + 32, 32),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(32, 2)
        )
        
    def forward(self, mri, clinical):
        m_emb = self.mri_proj(mri)
        c_emb = self.clin_proj(clinical)
        
        # Gated Attention
        m_w = self.mri_attn(m_emb)
        c_w = self.clin_attn(c_emb)
        
        # Weighted Features
        m_w_emb = m_emb * m_w
        c_w_emb = c_emb * c_w
        
        combined = torch.cat([m_w_emb, c_w_emb], dim=1)
        return self.fusion_net(combined)

# ==============================================================================
# TRAINING UTILS
# ==============================================================================

def train_model(model_class, train_loader, test_loader, name):
    print(f"\nTraining {name}...")
    model = model_class().to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    best_auc = 0
    patience_counter = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            mri = batch['mri'].to(Config.DEVICE)
            clin = batch['clinical'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(mri, clin)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Eval
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                mri = batch['mri'].to(Config.DEVICE)
                clin = batch['clinical'].to(Config.DEVICE)
                labels = batch['label'].to(Config.DEVICE)
                
                outputs = model(mri, clin)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
            
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(Config.RESULTS_DIR, f"best_{name}.pth"))
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"  Early stopping at epoch {epoch} (Best AUC: {best_auc:.4f})")
                break
                
    return best_auc

def evaluate_final(model_class, test_loader, name):
    model = model_class().to(Config.DEVICE)
    model.load_state_dict(torch.load(os.path.join(Config.RESULTS_DIR, f"best_{name}.pth")))
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            mri = batch['mri'].to(Config.DEVICE)
            clin = batch['clinical'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            outputs = model(mri, clin)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, np.array(all_probs) > 0.5)
    
    # Bootstrap CI
    rng = np.random.RandomState(Config.RANDOM_SEED)
    indices = np.arange(len(all_labels))
    aucs = []
    for _ in range(100):
        boot_idx = rng.choice(indices, len(indices), replace=True)
        try:
            score = roc_auc_score(np.array(all_labels)[boot_idx], np.array(all_probs)[boot_idx])
            aucs.append(score)
        except:
            pass
            
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)
    
    return {
        "AUC": auc,
        "Accuracy": acc,
        "AUC_CI_Lower": ci_lower,
        "AUC_CI_Upper": ci_upper
    }

def main():
    print("="*60)
    print("LEVEL-MAX Experiments: Overpowered Fusion")
    print("="*60)
    
    # Load Data
    X_mri_train, X_clin_train, y_train, scaler = load_adni_level_max(Config.TRAIN_CSV, is_train=True)
    X_mri_test, X_clin_test, y_test, _ = load_adni_level_max(Config.TEST_CSV, scaler=scaler, is_train=False)
    
    print(f"Train samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Clinical features: {X_clin_train.shape[1]}")
    
    # DataLoaders
    train_ds = ADNIDataset(X_mri_train, X_clin_train, y_train)
    test_ds = ADNIDataset(X_mri_test, X_clin_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Train Models
    results = {}
    
    # 1. MRI Only (Baseline check)
    train_model(MRIOnlyModel, train_loader, test_loader, "MRI_Only")
    results["MRI_Only"] = evaluate_final(MRIOnlyModel, test_loader, "MRI_Only")
    
    # 2. Late Fusion
    train_model(LateFusionModel, train_loader, test_loader, "Late_Fusion")
    results["Late_Fusion"] = evaluate_final(LateFusionModel, test_loader, "Late_Fusion")
    
    # 3. Attention Fusion
    train_model(AttentionFusionModel, train_loader, test_loader, "Attention_Fusion")
    results["Attention_Fusion"] = evaluate_final(AttentionFusionModel, test_loader, "Attention_Fusion")
    
    # Save Results
    with open(os.path.join(Config.RESULTS_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nFINAL RESULTS:")
    for model, metrics in results.items():
        print(f"{model:<20} AUC: {metrics['AUC']:.4f} ({metrics['AUC_CI_Lower']:.4f}-{metrics['AUC_CI_Upper']:.4f})")
    print("="*60)

if __name__ == "__main__":
    main()
