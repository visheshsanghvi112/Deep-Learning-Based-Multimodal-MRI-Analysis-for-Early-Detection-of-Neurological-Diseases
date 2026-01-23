"""
Configuration for Longitudinal Multimodal Fusion Experiment
============================================================
All paths, hyperparameters, and settings in one place.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Raw data
ADNI_ROOT = r"C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T\ADNI"
ADNIMERGE_PATH = os.path.join(ADNI_ROOT, "ADNIMERGE_23Dec2025.csv")
ADNIMERGE_BACKUP = r"D:\discs\data\ADNI\ADNIMERGE_23Dec2025.csv"

# Existing preprocessed data (reuse from project_longitudinal)
LONGITUDINAL_FEATURES_PATH = r"D:\discs\project_longitudinal\data\features\longitudinal_features.npz"
LONGITUDINAL_SPLITS_PATH = r"D:\discs\project_longitudinal\data\processed\train_test_split.csv"

# This project
PROJECT_ROOT = Path(r"D:\discs\project_longitudinal_fusion")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create directories
for d in [DATA_DIR, CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Biomarkers to use (from ADNIMERGE)
VOLUMETRIC_BIOMARKERS = [
    'Hippocampus',    # Best single predictor (0.725 AUC)
    'Ventricles',     # Enlargement marker
    'Entorhinal',     # Early atrophy site (0.691 AUC)
    'MidTemp',        # Temporal lobe (0.678 AUC)
    'Fusiform',       # Face recognition area
    'WholeBrain'      # Total brain volume
]

DEMOGRAPHIC_FEATURES = ['AGE', 'PTGENDER', 'APOE4']

# Feature dimensions
RESNET_DIM = 512
BASELINE_BIO_DIM = 9   # 6 volumes + age + sex + APOE4
FOLLOWUP_BIO_DIM = 6   # 6 volumes only
DELTA_BIO_DIM = 6      # 6 delta values

TOTAL_BIO_DIM = BASELINE_BIO_DIM + FOLLOWUP_BIO_DIM + DELTA_BIO_DIM  # 21
TOTAL_RESNET_DIM = RESNET_DIM * 3  # 1536

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Architecture
MODEL_CONFIG = {
    'resnet_dim': RESNET_DIM,
    'bio_dim': TOTAL_BIO_DIM,
    'hidden_dim': 256,
    'num_heads': 4,           # Multi-head attention
    'num_layers': 2,          # Transformer layers
    'dropout': 0.4,
    'use_batch_norm': True,
    'use_layer_norm': True
}

# Training
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,     # L2 regularization
    'epochs': 150,
    'patience': 20,           # Early stopping
    'lr_patience': 10,        # LR scheduler patience
    'lr_factor': 0.5,         # LR reduction factor
    'min_lr': 1e-6,
    'gradient_clip': 1.0      # Gradient clipping
}

# Cross-validation
CV_CONFIG = {
    'n_folds': 5,
    'stratified': True,
    'shuffle': True
}

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

RANDOM_SEED = 42
DEVICE = 'cuda'  # Will fallback to CPU if not available

# Experiment name for logging
EXPERIMENT_NAME = "longitudinal_fusion_v1"

# =============================================================================
# BASELINES TO COMPARE
# =============================================================================

BASELINE_MODELS = [
    'logistic_regression',
    'random_forest',
    'xgboost',
    'svm',
    'mlp'
]

# =============================================================================
# VISUALIZATION
# =============================================================================

FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Color scheme (publication-ready)
COLORS = {
    'primary': '#2563EB',      # Blue
    'secondary': '#7C3AED',    # Purple
    'success': '#059669',      # Green
    'warning': '#D97706',      # Orange
    'danger': '#DC2626',       # Red
    'neutral': '#6B7280',      # Gray
    
    # Model-specific colors
    'resnet': '#6B7280',
    'biomarker': '#2563EB',
    'fusion': '#059669'
}
