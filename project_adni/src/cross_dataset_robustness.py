"""
================================================================================
CROSS-DATASET ROBUSTNESS EVALUATION (STRICT)
================================================================================
Purpose: Test representation robustness under dataset shift (OASIS <-> ADNI).
Constraint: Use ONLY Intersection of Clinical Features.
            NO cognitive downstream features.
            NO retraining feature extractors.

Feature Intersection:
- OASIS: Age, Education (from .npz indices 0, 5)
- ADNI:  Age, Education (from CSVs, Educ merged from ADNIMERGE)
- MRI:   512-dim ResNet18 (Shared architecture)

Experiments:
A) OASIS (Source) -> ADNI (Target)
B) ADNI (Source) -> OASIS (Target)

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
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    # Data Paths
    OASIS_NPZ = "D:/discs/extracted_features/oasis_all_features.npz"
    
    ADNI_TRAIN_CSV = "D:/discs/project_adni/data/features/train_level1.csv"
    ADNI_TEST_CSV  = "D:/discs/project_adni/data/features/test_level1.csv"
    ADNI_MERGE_CSV = "D:/discs/project_adni/data/csv/ADNIMERGE.csv"
    
    RANDOM_SEED = 42
    
    # Model config (Shared)
    MRI_DIM = 512
    MRI_PROJ_DIM = 64
    CLINICAL_DIM = 2  # Age, Education
    CLINICAL_PROJ_DIM = 32
    FUSION_DIM = 64
    attn_heads = 4
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    PATIENCE = 15
    DROPOUT = 0.5
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# MODELS (Reused Architecture)
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
    def __init__(self, mri_dim=512, clinical_dim=2, hidden_dim=32, dropout=0.5):
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
    def __init__(self, mri_dim=512, clinical_dim=2, hidden_dim=32, dropout=0.5):
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
        return {'logits': self.classifier(fused), 'features': fused}

# ==============================================================================
# DATA LOADING
# ==============================================================================

class RobustnessDataset(Dataset):
    def __init__(self, mri, clinical, labels):
        self.mri = torch.FloatTensor(mri)
        self.clinical = torch.FloatTensor(clinical)
        self.labels = torch.LongTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {'mri': self.mri[idx], 'clinical': self.clinical[idx], 'label': self.labels[idx]}

def load_oasis_with_split():
    """Load OASIS from NPZ, filter features/labels, create seeded 80/20 split."""
    print("[Loader] Loading OASIS .npz...")
    data = np.load(Config.OASIS_NPZ, allow_pickle=True)
    mri_full = data['mri_features']
    clin_full = data['clinical_features'] # [AGE, MMSE, nWBV, eTIV, ASF, EDUC]
    labels_raw = data['labels']
    
    # Filter Valid Labels (CDR 0 vs 0.5)
    valid_mask = []
    y = []
    for l in labels_raw:
        s = str(l)
        if s in ['0', '0.0']:
            valid_mask.append(True); y.append(0)
        elif s == '0.5':
            valid_mask.append(True); y.append(1)
        else:
            valid_mask.append(False)
    
    valid_mask = np.array(valid_mask)
    mri = mri_full[valid_mask]
    clin = clin_full[valid_mask]
    y = np.array(y)
    
    # Select Features: Age (0) and Educ (5)
    # Shapes: (N, 2)
    clinical_selected = clin[:, [0, 5]]
    
    print(f"  OASIS Subset: {len(y)} subjects (Age, Educ only)")
    
    # Create Split
    # Since OASIS doesn't have a fixed test CSV, we create one deterministically
    mri_train, mri_test, clin_train, clin_test, y_train, y_test = train_test_split(
        mri, clinical_selected, y, test_size=0.2, stratify=y, random_state=Config.RANDOM_SEED
    )
    
    return (mri_train, clin_train, y_train), (mri_test, clin_test, y_test)

def load_adni_splits():
    """Load ADNI fixed L1 splits, merge Education from ADNIMERGE."""
    print("[Loader] Loading ADNI CSVs...")
    
    # Load Main Splits
    train_df = pd.read_csv(Config.ADNI_TRAIN_CSV)
    test_df = pd.read_csv(Config.ADNI_TEST_CSV)
    
    # Load ADNIMERGE for Education
    merge_df = pd.read_csv(Config.ADNI_MERGE_CSV, low_memory=False)
    # Filter to baseline visits
    bl_merge = merge_df[merge_df['VISCODE'].isin(['bl', 'sc', 'scmri', 'm00'])].copy()
    educ_map = bl_merge[['PTID', 'PTEDUCAT']].dropna().drop_duplicates(subset='PTID').set_index('PTID')
    
    def process_adni_df(df):
        # Merge Education
        df = df.merge(educ_map, left_on='Subject', right_index=True, how='left')
        
        # Fill missing education if any (should be none based on report)
        if df['PTEDUCAT'].isna().any():
            print(f"  Warning: {df['PTEDUCAT'].isna().sum()} missing Education values filled with mean.")
            df['PTEDUCAT'] = df['PTEDUCAT'].fillna(df['PTEDUCAT'].mean())
            
        # Extract MRI
        feat_cols = [f'f{i}' for i in range(512)]
        mri = df[feat_cols].values.astype(np.float32)
        
        # Extract Clinical: Age, Education
        # Align with OASIS order: [Age, Educ]
        clinical = df[['Age', 'PTEDUCAT']].values.astype(np.float32)
        
        # Labels
        labels = np.where(df['Group'] == 'CN', 0, 1).astype(np.int64)
        
        return mri, clinical, labels
    
    train_data = process_adni_df(train_df)
    test_data = process_adni_df(test_df)
    
    print(f"  ADNI Train: {len(train_data[2])}, Test: {len(test_data[2])} (Age, Educ only)")
    return train_data, test_data

# ==============================================================================
# PIPELINE
# ==============================================================================

def train_phase(model, train_loader, val_loader):
    model.to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # Class weights
    labels = []
    for batch in train_loader: labels.extend(batch['label'].numpy())
    counts = np.bincount(labels)
    weights = torch.FloatTensor(1.0/counts).to(Config.DEVICE)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    best_auc = 0
    patience = 0
    best_state = None
    
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch['mri'].to(Config.DEVICE), batch['clinical'].to(Config.DEVICE))
            loss = criterion(out['logits'], batch['label'].to(Config.DEVICE))
            loss.backward()
            optimizer.step()
            
        # Validation
        val_auc = evaluate_phase(model, val_loader)['auc']
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= Config.PATIENCE: break
            
    if best_state:
        model.load_state_dict(best_state)
    return best_auc

def evaluate_phase(model, loader):
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch['mri'].to(Config.DEVICE), batch['clinical'].to(Config.DEVICE))
            probs.extend(torch.softmax(out['logits'], dim=1)[:,1].cpu().numpy())
            trues.extend(batch['label'].numpy())
    
    auc = roc_auc_score(trues, probs)
    acc = accuracy_score(trues, np.array(probs)>=0.5)
    return {'auc': auc, 'acc': acc}

def run_experiment(exp_name, source_data, target_data):
    print(f"\n>>> Running Experiment {exp_name} <<<")
    
    src_mri, src_clin, src_y = source_data
    tgt_mri, tgt_clin, tgt_y = target_data
    
    # Scaler frozen on Source
    scaler_mri = StandardScaler().fit(src_mri)
    scaler_clin = StandardScaler().fit(src_clin)
    
    src_mri_s = scaler_mri.transform(src_mri)
    src_clin_s = scaler_clin.transform(src_clin)
    tgt_mri_s = scaler_mri.transform(tgt_mri)
    tgt_clin_s = scaler_clin.transform(tgt_clin)
    
    # Validation split from Source (internal 20%)
    m_tr, m_val, c_tr, c_val, y_tr, y_val = train_test_split(
        src_mri_s, src_clin_s, src_y, test_size=0.2, stratify=src_y, random_state=Config.RANDOM_SEED
    )
    
    train_loader = DataLoader(RobustnessDataset(m_tr, c_tr, y_tr), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(RobustnessDataset(m_val, c_val, y_val), batch_size=Config.BATCH_SIZE)
    test_loader  = DataLoader(RobustnessDataset(tgt_mri_s, tgt_clin_s, tgt_y), batch_size=Config.BATCH_SIZE)
    
    results = {}
    models = [
        ("MRI-Only", MRIOnlyModel(Config.MRI_DIM, Config.MRI_PROJ_DIM, Config.DROPOUT)),
        ("Late Fusion", LateFusionModel(Config.MRI_DIM, Config.CLINICAL_DIM, Config.MRI_PROJ_DIM, Config.DROPOUT)),
        ("Attention Fusion", AttentionFusionModel(Config.MRI_DIM, Config.CLINICAL_DIM, Config.MRI_PROJ_DIM, Config.DROPOUT))
    ]
    
    print(f"{'Model':<20} {'Src Val AUC':>12} {'TGT TEST AUC':>15} {'Acc':>8} {'Drop':>8}")
    print("-" * 70)
    
    for name, model in models:
        torch.manual_seed(Config.RANDOM_SEED)
        
        # Train
        train_phase(model, train_loader, val_loader)
        
        # Evaluate Source (proxy)
        src_res = evaluate_phase(model, val_loader)
        
        # Evaluate Target
        tgt_res = evaluate_phase(model, test_loader)
        
        drop = src_res['auc'] - tgt_res['auc']
        results[name] = {'src_auc': src_res['auc'], 'tgt_auc': tgt_res['auc'], 'tgt_acc': tgt_res['acc'], 'drop': drop}
        
        print(f"{name:<20} {src_res['auc']:>12.3f} {tgt_res['auc']:>15.3f} {tgt_res['acc']:>8.1%} {drop:>8.3f}")
        
    return results

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("="*70)
    print("CROSS-DATASET ROBUSTNESS EVALUATION")
    print("Features: MRI (512) + Clinical [Age, Education]")
    print("="*70)
    
    # Load Data
    oasis_train, oasis_test = load_oasis_with_split()
    adni_train, adni_test   = load_adni_splits()
    
    # ------------------------------------------------------------------
    # Experiment A: OASIS (Source) -> ADNI (Target)
    # Source: OASIS Full Train (combination of split)
    # Target: ADNI Full Test
    # ------------------------------------------------------------------
    # We combine oasis_train+test labels to maximize source data? 
    # Or just use oasis_train? "Train on OASIS training split"
    # I'll use oasis_train tuple defined above.
    
    exp_a = run_experiment("A: OASIS -> ADNI", oasis_train, adni_test) 
    # Note: Target is ADNI Test Set (126 subjects)
    
    # ------------------------------------------------------------------
    # Experiment B: ADNI (Source) -> OASIS (Target)
    # Source: ADNI Train CSV
    # Target: OASIS Test Split (created above)
    # ------------------------------------------------------------------
    exp_b = run_experiment("B: ADNI -> OASIS", adni_train, oasis_test)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Confirmed: NO MMSE/CDRSB used. NO dataset merging.")
    print("Confirmed: Only Age + Education used for clinical.")

if __name__ == "__main__":
    main()
