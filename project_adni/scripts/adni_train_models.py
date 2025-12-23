"""
================================================================================
ADNI In-Dataset Model Training (Part B)
================================================================================
Trains the SAME model architectures used for OASIS on ADNI data.
- NO ADNI-specific tuning
- IDENTICAL hyperparameters to OASIS
- Purpose: Establish independent ADNI baseline for comparison

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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION (IDENTICAL TO OASIS)
# ==============================================================================

class Config:
    """Training configuration - IDENTICAL to OASIS pipeline (no ADNI tuning)"""
    TRAIN_CSV = "D:/discs/adni_train.csv"
    TEST_CSV = "D:/discs/adni_test.csv"
    
    RANDOM_SEED = 42
    
    # Model architecture (SAME AS OASIS)
    MRI_DIM = 512
    MRI_PROJ_DIM = 64
    CLINICAL_DIM = 2  # Age, Sex only (what we have in ADNI)
    CLINICAL_PROJ_DIM = 32
    FUSION_DIM = 64
    NUM_ATTENTION_HEADS = 4
    
    # Training hyperparameters (SAME AS OASIS)
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    PATIENCE = 15
    DROPOUT = 0.5
    
    # Bootstrap for confidence intervals
    N_BOOTSTRAP = 1000
    CI_LEVEL = 0.95
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_adni_data(csv_path: str):
    """Load ADNI features from CSV."""
    df = pd.read_csv(csv_path)
    
    # Extract MRI features (f0 to f511)
    feature_cols = [f'f{i}' for i in range(512)]
    mri = df[feature_cols].values.astype(np.float32)
    
    # Clinical features: Age, Sex (encoded as 0/1)
    sex_encoded = (df['Sex'] == 'M').astype(np.float32).values
    age = df['Age'].values.astype(np.float32)
    clinical = np.column_stack([age, sex_encoded])
    
    # Labels: Binary classification (CN=0 vs MCI+AD=1)
    labels = np.where(df['Group'] == 'CN', 0, 1).astype(np.int64)
    
    return mri, clinical, labels, df


class ADNIDataset(Dataset):
    """Dataset for ADNI training"""
    
    def __init__(self, mri: np.ndarray, clinical: np.ndarray, labels: np.ndarray):
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
# MODEL ARCHITECTURES (IDENTICAL TO OASIS)
# ==============================================================================

class MRIOnlyModel(nn.Module):
    """Model 1: MRI-Only Baseline"""
    
    def __init__(self, mri_dim: int = 512, hidden_dim: int = 32, dropout: float = 0.5):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(mri_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, 2)
    
    def forward(self, mri: torch.Tensor, clinical: torch.Tensor = None):
        features = self.encoder(mri)
        logits = self.classifier(features)
        return {'logits': logits, 'features': features}


class LateFusionModel(nn.Module):
    """Model 2: Late Fusion"""
    
    def __init__(self, mri_dim: int = 512, clinical_dim: int = 2, 
                 hidden_dim: int = 32, dropout: float = 0.5):
        super().__init__()
        
        self.mri_encoder = nn.Sequential(
            nn.Linear(mri_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        fused_dim = hidden_dim * 2
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, mri: torch.Tensor, clinical: torch.Tensor):
        mri_feat = self.mri_encoder(mri)
        clin_feat = self.clinical_encoder(clinical)
        fused = torch.cat([mri_feat, clin_feat], dim=-1)
        fused = self.fusion_layer(fused)
        logits = self.classifier(fused)
        return {'logits': logits, 'features': fused}


class AttentionFusionModel(nn.Module):
    """Model 3: Gated Attention Fusion"""
    
    def __init__(self, mri_dim: int = 512, clinical_dim: int = 2,
                 hidden_dim: int = 32, dropout: float = 0.5):
        super().__init__()
        
        self.mri_encoder = nn.Sequential(
            nn.Linear(mri_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, mri: torch.Tensor, clinical: torch.Tensor):
        mri_feat = self.mri_encoder(mri)
        clin_feat = self.clinical_encoder(clinical)
        
        combined = torch.cat([mri_feat, clin_feat], dim=-1)
        gate_values = self.gate(combined)
        
        fused = gate_values * mri_feat + (1 - gate_values) * clin_feat
        logits = self.classifier(fused)
        
        return {'logits': logits, 'features': fused, 'gate_values': gate_values}


# ==============================================================================
# TRAINING & EVALUATION
# ==============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        mri = batch['mri'].to(device)
        clinical = batch['clinical'].to(device)
        labels = batch['label'].to(device)
        
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
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            mri = batch['mri'].to(device)
            clinical = batch['clinical'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(mri, clinical)
            probs = torch.softmax(outputs['logits'], dim=1)[:, 1]
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    auc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    
    return {'auc': auc, 'accuracy': acc, 'probs': all_probs, 'labels': all_labels}


def train_model(model, train_loader, device, config):
    """Train model with early stopping using training set only."""
    model.to(device)
    
    # Class weights
    train_labels = np.concatenate([batch['label'].numpy() for batch in train_loader])
    class_counts = np.bincount(train_labels)
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    class_weights = class_weights / class_weights.sum()
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                            weight_decay=config.WEIGHT_DECAY)
    
    best_train_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.PATIENCE:
            break
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(device)
    
    return epoch + 1


def bootstrap_ci(y_true, y_prob, n_bootstrap=1000, ci=0.95):
    """Calculate bootstrap confidence interval for AUC."""
    aucs = []
    n = len(y_true)
    
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    
    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)
    
    return lower, upper


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("ADNI IN-DATASET MODEL TRAINING (Part B)")
    print("="*70)
    print("\n[!] Using IDENTICAL hyperparameters to OASIS (no ADNI-specific tuning)")
    
    # Set seeds
    np.random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)
    
    # Load data
    print("\n[1] Loading ADNI data...")
    train_mri, train_clinical, train_labels, _ = load_adni_data(Config.TRAIN_CSV)
    test_mri, test_clinical, test_labels, _ = load_adni_data(Config.TEST_CSV)
    
    print(f"    Training: {len(train_labels)} subjects (CN={sum(train_labels==0)}, Impaired={sum(train_labels==1)})")
    print(f"    Test:     {len(test_labels)} subjects (CN={sum(test_labels==0)}, Impaired={sum(test_labels==1)})")
    
    # Normalize features
    mri_scaler = StandardScaler()
    clinical_scaler = StandardScaler()
    
    train_mri_scaled = mri_scaler.fit_transform(train_mri)
    train_clinical_scaled = clinical_scaler.fit_transform(train_clinical)
    
    test_mri_scaled = mri_scaler.transform(test_mri)
    test_clinical_scaled = clinical_scaler.transform(test_clinical)
    
    # Create datasets
    train_dataset = ADNIDataset(train_mri_scaled, train_clinical_scaled, train_labels)
    test_dataset = ADNIDataset(test_mri_scaled, test_clinical_scaled, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Results storage
    results = {}
    
    # Train each model
    models_to_train = [
        ("MRI-Only", MRIOnlyModel(mri_dim=Config.MRI_DIM, hidden_dim=Config.MRI_PROJ_DIM, dropout=Config.DROPOUT)),
        ("Late Fusion", LateFusionModel(mri_dim=Config.MRI_DIM, clinical_dim=Config.CLINICAL_DIM, 
                                         hidden_dim=Config.MRI_PROJ_DIM, dropout=Config.DROPOUT)),
        ("Attention Fusion", AttentionFusionModel(mri_dim=Config.MRI_DIM, clinical_dim=Config.CLINICAL_DIM,
                                                   hidden_dim=Config.MRI_PROJ_DIM, dropout=Config.DROPOUT)),
    ]
    
    print("\n[2] Training models...")
    
    for model_name, model in models_to_train:
        print(f"\n    Training {model_name}...")
        
        # Reset seeds for reproducibility
        torch.manual_seed(Config.RANDOM_SEED)
        
        # Train
        epochs = train_model(model, train_loader, Config.DEVICE, Config)
        
        # Evaluate on TEST set
        test_result = evaluate(model, test_loader, Config.DEVICE)
        
        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_ci(
            test_result['labels'], test_result['probs'], 
            n_bootstrap=Config.N_BOOTSTRAP, ci=Config.CI_LEVEL
        )
        
        results[model_name] = {
            'auc': test_result['auc'],
            'accuracy': test_result['accuracy'],
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'epochs': epochs
        }
        
        print(f"        AUC: {test_result['auc']:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
        print(f"        Accuracy: {test_result['accuracy']*100:.1f}%")
        print(f"        Epochs: {epochs}")
    
    # Print summary table
    print("\n" + "="*70)
    print("ADNI IN-DATASET RESULTS (TEST SET)")
    print("="*70)
    print(f"\nTest set: {len(test_labels)} subjects")
    print(f"Task: Binary classification (CN vs MCI+AD)")
    print(f"Hyperparameters: IDENTICAL to OASIS (no tuning)")
    
    print("\n" + "-"*70)
    print(f"{'Model':<20} {'AUC':>8} {'95% CI':>18} {'Accuracy':>10}")
    print("-"*70)
    
    for model_name, res in results.items():
        ci_str = f"({res['ci_lower']:.3f}-{res['ci_upper']:.3f})"
        print(f"{model_name:<20} {res['auc']:>8.3f} {ci_str:>18} {res['accuracy']*100:>9.1f}%")
    
    print("-"*70)
    
    # Comparison notes
    print("\n" + "="*70)
    print("COMPARISON NOTES (ADNI vs OASIS TRENDS)")
    print("="*70)
    
    mri_auc = results['MRI-Only']['auc']
    late_auc = results['Late Fusion']['auc']
    attn_auc = results['Attention Fusion']['auc']
    
    print(f"\n1. Multimodal benefit (Late Fusion vs MRI-Only):")
    diff = (late_auc - mri_auc) * 100
    print(f"   ADNI: {'+' if diff >= 0 else ''}{diff:.1f}% AUC")
    
    print(f"\n2. Attention benefit (Attention vs Late Fusion):")
    diff = (attn_auc - late_auc) * 100
    print(f"   ADNI: {'+' if diff >= 0 else ''}{diff:.1f}% AUC")
    
    print(f"\n3. Overall best model: {max(results, key=lambda x: results[x]['auc'])}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\n[!] Cross-dataset experiments NOT performed (awaiting explicit instruction)")
    
    return results


if __name__ == "__main__":
    results = main()
