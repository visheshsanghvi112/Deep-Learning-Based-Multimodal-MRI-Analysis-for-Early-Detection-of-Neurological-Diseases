"""
================================================================================
ADNI Level-2: Clinical-Informed Models (Upper-Bound Reference)
================================================================================
Merges ADNIMERGE clinical features with MRI features for enhanced multimodal.

IMPORTANT DISCLAIMERS:
- MMSE and CDRSB are cognitively DOWNSTREAM measures
- These results serve as UPPER-BOUND REFERENCE, not primary early-detection claims
- Early detection should NOT rely on cognitive scores available only after diagnosis

This experiment answers: "What is the ceiling performance with full clinical data?"

Author: Research Pipeline
Date: December 2025
================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION (IDENTICAL TO LEVEL-1)
# ==============================================================================

class Config:
    """Training configuration - IDENTICAL to Level-1 (no tuning)"""
    ADNIMERGE_CSV = "D:/discs/ADNI/ADNIMERGE_23Dec2025.csv"
    BASELINE_CSV = "D:/discs/adni_baseline_selection.csv"
    TRAIN_CSV = "D:/discs/adni_train.csv"
    TEST_CSV = "D:/discs/adni_test.csv"
    
    RANDOM_SEED = 42
    
    # Model architecture (SAME AS LEVEL-1)
    MRI_DIM = 512
    MRI_PROJ_DIM = 64
    # Clinical features: MMSE, CDRSB, PTEDUCAT, AGE, APOE4 = 5 features (like OASIS)
    CLINICAL_DIM = 5
    CLINICAL_PROJ_DIM = 32
    
    # Training hyperparameters (SAME AS LEVEL-1)
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    PATIENCE = 15
    DROPOUT = 0.5
    
    N_BOOTSTRAP = 1000
    CI_LEVEL = 0.95
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# STEP 1: MERGE ADNIMERGE CLINICAL FEATURES
# ==============================================================================

def merge_clinical_features():
    """Merge ADNIMERGE clinical features with existing MRI features."""
    print("\n" + "="*70)
    print("STEP 1: MERGING ADNIMERGE CLINICAL FEATURES")
    print("="*70)
    
    # Load ADNIMERGE
    merge_df = pd.read_csv(Config.ADNIMERGE_CSV, low_memory=False)
    print(f"Loaded ADNIMERGE: {len(merge_df)} rows")
    
    # Load our baseline selection
    baseline_df = pd.read_csv(Config.BASELINE_CSV)
    our_subjects = set(baseline_df['Subject'].unique())
    print(f"Our subjects: {len(our_subjects)}")
    
    # Filter ADNIMERGE to baseline visits
    baseline_merge = merge_df[
        (merge_df['PTID'].isin(our_subjects)) & 
        (merge_df['VISCODE'].isin(['bl', 'sc', 'scmri', 'm00']))
    ].copy()
    print(f"Baseline rows: {len(baseline_merge)}")
    
    # Select clinical features
    clinical_cols = ['PTID', 'MMSE', 'CDRSB', 'PTEDUCAT', 'AGE', 'APOE4']
    clinical_df = baseline_merge[clinical_cols].copy()
    
    # Handle missing APOE4 (fill with 0 = no risk allele)
    clinical_df['APOE4'] = clinical_df['APOE4'].fillna(0)
    
    # Drop any remaining rows with missing values
    clinical_df = clinical_df.dropna()
    print(f"Complete clinical data: {len(clinical_df)} subjects")
    
    # Rename for merge
    clinical_df = clinical_df.rename(columns={'PTID': 'Subject'})
    
    # Load train/test splits
    train_df = pd.read_csv(Config.TRAIN_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)
    
    # Merge clinical features
    train_merged = train_df.merge(clinical_df, on='Subject', how='inner')
    test_merged = test_df.merge(clinical_df, on='Subject', how='inner')
    
    print(f"\nTrain set merged: {len(train_merged)}/{len(train_df)} subjects")
    print(f"Test set merged:  {len(test_merged)}/{len(test_df)} subjects")
    
    # Check clinical feature stats
    print("\nClinical Feature Statistics (Train):")
    for col in ['MMSE', 'CDRSB', 'PTEDUCAT', 'AGE', 'APOE4']:
        print(f"  {col}: mean={train_merged[col].mean():.2f}, std={train_merged[col].std():.2f}")
    
    return train_merged, test_merged


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data_level2(train_df, test_df):
    """Load Level-2 features (MRI + rich clinical)."""
    
    # MRI features
    feature_cols = [f'f{i}' for i in range(512)]
    train_mri = train_df[feature_cols].values.astype(np.float32)
    test_mri = test_df[feature_cols].values.astype(np.float32)
    
    # Clinical features: MMSE, CDRSB, PTEDUCAT, AGE, APOE4
    clinical_cols = ['MMSE', 'CDRSB', 'PTEDUCAT', 'AGE', 'APOE4']
    train_clinical = train_df[clinical_cols].values.astype(np.float32)
    test_clinical = test_df[clinical_cols].values.astype(np.float32)
    
    # Labels: Binary (CN=0 vs MCI+AD=1)
    train_labels = np.where(train_df['Group'] == 'CN', 0, 1).astype(np.int64)
    test_labels = np.where(test_df['Group'] == 'CN', 0, 1).astype(np.int64)
    
    return train_mri, train_clinical, train_labels, test_mri, test_clinical, test_labels


class MultimodalDataset(Dataset):
    def __init__(self, mri, clinical, labels):
        self.mri = torch.FloatTensor(mri)
        self.clinical = torch.FloatTensor(clinical)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {'mri': self.mri[idx], 'clinical': self.clinical[idx], 'label': self.labels[idx]}


# ==============================================================================
# MODEL ARCHITECTURES (IDENTICAL TO LEVEL-1)
# ==============================================================================

class MRIOnlyModel(nn.Module):
    def __init__(self, mri_dim=512, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(mri_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim // 2, 2)
    
    def forward(self, mri, clinical=None):
        features = self.encoder(mri)
        return {'logits': self.classifier(features), 'features': features}


class LateFusionModel(nn.Module):
    def __init__(self, mri_dim=512, clinical_dim=5, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.mri_encoder = nn.Sequential(
            nn.Linear(mri_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, mri, clinical):
        mri_feat = self.mri_encoder(mri)
        clin_feat = self.clinical_encoder(clinical)
        fused = self.fusion_layer(torch.cat([mri_feat, clin_feat], dim=-1))
        return {'logits': self.classifier(fused), 'features': fused}


class AttentionFusionModel(nn.Module):
    def __init__(self, mri_dim=512, clinical_dim=5, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.mri_encoder = nn.Sequential(
            nn.Linear(mri_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, mri, clinical):
        mri_feat = self.mri_encoder(mri)
        clin_feat = self.clinical_encoder(clinical)
        gate_values = self.gate(torch.cat([mri_feat, clin_feat], dim=-1))
        fused = gate_values * mri_feat + (1 - gate_values) * clin_feat
        return {'logits': self.classifier(fused), 'features': fused, 'gate_values': gate_values}


# ==============================================================================
# TRAINING & EVALUATION
# ==============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        mri, clinical, labels = batch['mri'].to(device), batch['clinical'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(mri, clinical)
        loss = criterion(outputs['logits'], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            mri, clinical, labels = batch['mri'].to(device), batch['clinical'].to(device), batch['label'].to(device)
            outputs = model(mri, clinical)
            probs = torch.softmax(outputs['logits'], dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, (all_probs >= 0.5).astype(int))
    return {'auc': auc, 'accuracy': acc, 'probs': all_probs, 'labels': all_labels}


def train_model(model, train_loader, device, config):
    model.to(device)
    train_labels = np.concatenate([batch['label'].numpy() for batch in train_loader])
    class_counts = np.bincount(train_labels)
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    class_weights = class_weights / class_weights.sum()
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    best_loss, patience_counter, best_state = float('inf'), 0, None
    for epoch in range(config.NUM_EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        if loss < best_loss:
            best_loss = loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.PATIENCE:
            break
    
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    return epoch + 1


def bootstrap_ci(y_true, y_prob, n=1000, ci=0.95):
    aucs = []
    for _ in range(n):
        idx = np.random.randint(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return np.percentile(aucs, (1-ci)/2*100), np.percentile(aucs, (1+ci)/2*100)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("ADNI LEVEL-2: CLINICAL-INFORMED MODELS (UPPER-BOUND REFERENCE)")
    print("="*70)
    print("\n[!] DISCLAIMER: MMSE/CDRSB are cognitively DOWNSTREAM measures")
    print("    These results are UPPER-BOUND REFERENCE, not early-detection claims")
    print("    Early detection should NOT rely on cognitive scores")
    
    np.random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)
    
    # Step 1: Merge clinical features
    train_df, test_df = merge_clinical_features()
    
    # Step 2: Load data
    print("\n" + "="*70)
    print("STEP 2: LOADING LEVEL-2 DATA")
    print("="*70)
    
    train_mri, train_clinical, train_labels, test_mri, test_clinical, test_labels = load_data_level2(train_df, test_df)
    
    print(f"\nTrain: {len(train_labels)} (CN={sum(train_labels==0)}, Impaired={sum(train_labels==1)})")
    print(f"Test:  {len(test_labels)} (CN={sum(test_labels==0)}, Impaired={sum(test_labels==1)})")
    print(f"Clinical features: MMSE, CDRSB, Education, Age, APOE4")
    
    # Normalize
    mri_scaler, clin_scaler = StandardScaler(), StandardScaler()
    train_mri_s = mri_scaler.fit_transform(train_mri)
    train_clin_s = clin_scaler.fit_transform(train_clinical)
    test_mri_s = mri_scaler.transform(test_mri)
    test_clin_s = clin_scaler.transform(test_clinical)
    
    train_dataset = MultimodalDataset(train_mri_s, train_clin_s, train_labels)
    test_dataset = MultimodalDataset(test_mri_s, test_clin_s, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Step 3: Train models
    print("\n" + "="*70)
    print("STEP 3: TRAINING LEVEL-2 MODELS")
    print("="*70)
    
    results = {}
    
    # Level-1 baseline (for comparison)
    level1_mri_auc = 0.583  # From previous run
    
    models = [
        ("MRI-Only (L2)", MRIOnlyModel(Config.MRI_DIM, Config.MRI_PROJ_DIM, Config.DROPOUT)),
        ("Late Fusion (L2)", LateFusionModel(Config.MRI_DIM, Config.CLINICAL_DIM, Config.MRI_PROJ_DIM, Config.DROPOUT)),
        ("Attention Fusion (L2)", AttentionFusionModel(Config.MRI_DIM, Config.CLINICAL_DIM, Config.MRI_PROJ_DIM, Config.DROPOUT)),
    ]
    
    for name, model in models:
        print(f"\n    Training {name}...")
        torch.manual_seed(Config.RANDOM_SEED)
        epochs = train_model(model, train_loader, Config.DEVICE, Config)
        result = evaluate(model, test_loader, Config.DEVICE)
        ci_low, ci_high = bootstrap_ci(result['labels'], result['probs'], Config.N_BOOTSTRAP, Config.CI_LEVEL)
        results[name] = {'auc': result['auc'], 'acc': result['accuracy'], 'ci': (ci_low, ci_high), 'epochs': epochs}
        print(f"        AUC: {result['auc']:.3f} ({ci_low:.3f}-{ci_high:.3f}), Acc: {result['accuracy']*100:.1f}%")
    
    # Step 4: Report
    print("\n" + "="*70)
    print("ADNI LEVEL-2 RESULTS (CLINICAL-INFORMED UPPER-BOUND)")
    print("="*70)
    print(f"\nTest set: {len(test_labels)} subjects")
    print(f"Clinical features: MMSE, CDRSB, Education, Age, APOE4 (5 features)")
    
    print("\n" + "-"*70)
    print(f"{'Model':<25} {'AUC':>7} {'95% CI':>18} {'Acc':>8} {'dAUC vs L1':>12}")
    print("-"*70)
    
    for name, res in results.items():
        ci_str = f"({res['ci'][0]:.3f}-{res['ci'][1]:.3f})"
        delta = res['auc'] - level1_mri_auc
        delta_str = f"+{delta*100:.1f}%" if delta >= 0 else f"{delta*100:.1f}%"
        print(f"{name:<25} {res['auc']:>7.3f} {ci_str:>18} {res['acc']*100:>7.1f}% {delta_str:>12}")
    
    print("-"*70)
    
    # Comparison notes
    print("\n" + "="*70)
    print("COMPARISON WITH OASIS TRENDS")
    print("="*70)
    
    l2_mri = results.get('MRI-Only (L2)', {}).get('auc', 0)
    l2_late = results.get('Late Fusion (L2)', {}).get('auc', 0)
    l2_attn = results.get('Attention Fusion (L2)', {}).get('auc', 0)
    
    print("\nADNI Level-2 Observations:")
    print(f"  1. Multimodal benefit (Late vs MRI): +{(l2_late-l2_mri)*100:.1f}% AUC")
    print(f"  2. Attention benefit (Attn vs Late): +{(l2_attn-l2_late)*100:.1f}% AUC")
    print(f"  3. Best model: {max(results, key=lambda x: results[x]['auc'])}")
    
    print("\nOASIS Comparison (from prior results):")
    print("  - OASIS MRI-Only:     ~0.72 AUC")
    print("  - OASIS Late Fusion:  ~0.80 AUC (+8%)")
    print("  - OASIS shows stronger multimodal benefit")
    
    print("\n[!] REMINDER: Level-2 uses downstream cognitive scores (MMSE, CDRSB)")
    print("    These results are UPPER-BOUND, not realistic early-detection performance")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()
