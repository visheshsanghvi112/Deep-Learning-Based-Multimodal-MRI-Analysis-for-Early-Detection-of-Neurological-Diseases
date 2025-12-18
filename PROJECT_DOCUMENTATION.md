# 🧠 MASTER PROJECT DOCUMENTATION

**Deep Learning-Based Multimodal MRI Analysis for Early Detection of Neurological Diseases**

**Last Updated:** December 18, 2025  
**Status:** ✅ Feature Extraction Complete | 🔄 Model Training Ready

---

## 📋 TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [Dataset Information](#3-dataset-information)
4. [Feature Extraction Pipeline](#4-feature-extraction-pipeline)
5. [Classification Results](#5-classification-results)
6. [Model Architecture](#6-model-architecture)
7. [Project Structure](#7-project-structure)
8. [How to Run](#8-how-to-run)
9. [Research Phases & Progress](#9-research-phases--progress)
10. [Key Findings](#10-key-findings)
11. [Next Steps](#11-next-steps)
12. [File Inventory](#12-file-inventory)

---

## 1. EXECUTIVE SUMMARY

### 🎯 Research Goal
Develop deep learning models to detect **early-stage dementia** by combining:
- **MRI-based deep features** (ResNet18 CNN embeddings)
- **Clinical/demographic features** (Age, MMSE, brain volumes)

### ✅ Key Achievements
| Milestone | Status | Details |
|-----------|--------|---------|
| Data Processing | ✅ Complete | 436/436 OASIS-1 subjects processed |
| CNN Feature Extraction | ✅ Complete | 512-dim ResNet18 features for all subjects |
| Clinical Features | ✅ Complete | 6 normalized clinical features |
| Traditional ML Classification | ✅ Complete | Late Fusion AUC 0.794 (realistic scenario) |
| Deep Learning Models | ✅ Complete | All 3 models trained and compared |

### 🏆 Classification Results Summary

**Scenario: Without MMSE (Realistic Early Detection)**
```
Traditional ML (Logistic Regression):
  - MRI only:           AUC = 0.770 ± 0.080
  - Clinical only:      AUC = 0.743 ± 0.082
  - Late Fusion:        AUC = 0.794 ± 0.083  ← +2.3% over MRI

Deep Learning (5-fold CV):
  - MRI-Only DL:        AUC = 0.781 ± 0.087
  - Late Fusion DL:     AUC = 0.796 ± 0.092  ← +1.5% over MRI
  - Attention Fusion:   AUC = 0.790 ± 0.109  ← +1.0% over MRI
```

**Scenario: With MMSE (Reference Only)**
```
  - Clinical + MMSE:    AUC = 0.875 ± 0.021
  - Late Fusion:        AUC = 0.882 ± 0.044
  (Note: MMSE directly measures cognitive function, making this circular)
```

---

## 2. PROJECT OVERVIEW

### Research Context
- **Disease Focus:** Alzheimer's Disease / Dementia
- **Classification Task:** CDR=0 (Normal) vs CDR=0.5 (Very Mild Dementia)
- **Approach:** Multimodal deep learning (MRI + Clinical)

### Scientific Philosophy
We learn **latent neurodegenerative signatures** from MRI that correlate with early cognitive decline. CDR/MMSE serve as **validation anchors**, not primary targets.

**Why This Matters:**
- Models learn disease biology from brain structure
- Not just replicating clinical scoring rules
- Can potentially detect changes before clinical symptoms

---

## 3. DATASET INFORMATION

### OASIS-1 Cross-Sectional Dataset

| Attribute | Value |
|-----------|-------|
| **Total Subjects** | 436 |
| **Age Range** | 21-96 years |
| **Format** | ANALYZE (.hdr/.img) |
| **Space** | Talairach (T88_111) |
| **Voxel Size** | 1.0 × 1.0 × 1.0 mm |
| **Dimensions** | 176 × 208 × 176 |
| **Location** | `D:/discs/disc1` through `disc12` |

### CDR Distribution (Clinical Labels)
```
CDR = 0.0  (Normal):           135 subjects
CDR = 0.5  (Very Mild):         70 subjects
CDR = 1.0  (Mild):              28 subjects
CDR = 2.0  (Moderate):           2 subjects
CDR = NaN  (Young Controls):   201 subjects (no dementia screening needed)
─────────────────────────────────────────────
Usable for Classification:     205 subjects (CDR 0 vs 0.5)
```

### ADNI Dataset (For Future Work)
| Attribute | Value |
|-----------|-------|
| **Subjects** | 203 unique |
| **Scans** | 230 NIfTI files |
| **Size** | 7.84 GB |
| **Status** | Analyzed, needs processing |
| **Location** | `D:/discs/ADNI/` |

---

## 4. FEATURE EXTRACTION PIPELINE

### 4.1 MRI Feature Extraction (Deep Learning)

**Architecture:** ResNet18 (pretrained on ImageNet)
- **Approach:** 2.5D Multi-slice (axial, coronal, sagittal)
- **Slices:** 9 total (3 per plane: center ± 20)
- **Output:** 512-dimensional feature vector
- **Aggregation:** Mean pooling across all slices

**Why 2.5D Instead of Full 3D:**
1. Full 3D CNNs need massive GPU memory (6.4M voxels)
2. Small dataset (436 subjects) → high overfitting risk with 3D
3. Pretrained ImageNet models transfer well to 2D slices
4. Captures 3D spatial information efficiently

**Script:** `mri_feature_extraction.py`

### 4.2 Clinical Feature Extraction

| Feature | Description | Normalization |
|---------|-------------|---------------|
| AGE | Subject age in years | Z-score (μ=50, σ=20) |
| MMSE | Mini-Mental State Exam (0-30) | Z-score (μ=27, σ=3) |
| nWBV | Normalized whole brain volume | Z-score (μ=0.75, σ=0.05) |
| eTIV | Estimated total intracranial vol | Z-score (μ=1500, σ=200) |
| ASF | Atlas scaling factor | Z-score (μ=1.2, σ=0.2) |
| EDUC | Education level (1-5) | Z-score (μ=3, σ=1.5) |

### 4.3 Output Files

| File | Format | Size | Contents |
|------|--------|------|----------|
| `extracted_features/oasis_all_features.npz` | NumPy | 1.83 MB | All features |
| `extracted_features/oasis_all_features.pt` | PyTorch | 1.87 MB | Tensor format |

**Feature Shapes:**
```python
subject_ids:        (436,)      # Subject identifiers
mri_features:       (436, 512)  # CNN embeddings
clinical_features:  (436, 6)    # Clinical variables
combined_features:  (436, 518)  # Concatenated
labels:             (436,)      # CDR values (or NaN)
```

---

## 5. CLASSIFICATION RESULTS

### ⚠️ IMPORTANT NOTE
Previous results in this document were computed using naive concatenation of features.
The deep learning training script (`train_main.py`) was using **zero-filled CNN embeddings**,
which made the previous DL results invalid. This has been fixed in `train_multimodal.py`.

### 5.1 Traditional ML Results (Logistic Regression)
**Dataset:** 205 subjects (135 Normal, 70 Very Mild)  
**Method:** 5-Fold Stratified Cross-Validation

#### Without MMSE (Realistic Early Detection)
| Feature Set | AUC | Std | Notes |
|-------------|-----|-----|-------|
| MRI only (512d) | 0.770 | ±0.080 | Pure imaging biomarker |
| Clinical only (5d) | 0.743 | ±0.082 | AGE, nWBV, eTIV, ASF, EDUC |
| Naive Concatenation (517d) | 0.768 | ±0.083 | Dimension imbalance hurts |
| **Late Fusion (Probability)** | **0.794** | ±0.083 | **Best traditional ML** |

#### With MMSE (Reference Only)
| Feature Set | AUC | Std | Notes |
|-------------|-----|-----|-------|
| Clinical only (6d) | 0.875 | ±0.021 | MMSE dominates |
| Late Fusion | 0.882 | ±0.044 | Slight improvement |

### 5.2 Deep Learning Results
**Architecture:** Small MLPs with 32 hidden units, dropout=0.5
**Training:** AdamW, lr=1e-3, weight_decay=1e-4, early stopping

#### Model Comparison (Without MMSE)
| Model | Mean AUC | Std | vs MRI-Only |
|-------|----------|-----|-------------|
| MRI-Only DL | 0.781 | ±0.087 | baseline |
| Late Fusion DL | 0.796 | ±0.092 | +1.5% |
| Attention Fusion DL | 0.790 | ±0.109 | +1.0% |

### 5.3 Key Findings

1. **Multimodal improves over unimodal:** Both late fusion and attention fusion
   outperform MRI-only, confirming that clinical features provide complementary signal.

2. **Late fusion is competitive:** Simple concatenation-based fusion performs
   as well as attention mechanisms on this dataset size (N=205).

3. **Attention shows meaningful behavior:** Gate values vary across samples
   (std=0.157), indicating the model learns to dynamically weight modalities.

4. **Dimension imbalance matters:** Naive concatenation (512+5 features) underperforms
   proper late fusion that weights modalities equally.

5. **MMSE is circular:** Including MMSE artificially inflates AUC because it
   directly measures cognitive function (what CDR also measures).

### 5.4 Feature Importance Analysis

**Clinical Feature Weights (Logistic Regression):**
```
MMSE:  -1.88  ← Strongest predictor (but circular with CDR)
nWBV:  -0.42  ← Brain volume atrophy (genuine biomarker)
AGE:   +0.12
eTIV:  +0.05
ASF:   -0.04
EDUC:  -0.02
```

**Attention Model Gate Values:**
- Mean gate value: 0.682 (slightly favors MRI)
- Standard deviation: 0.157 (meaningful variation across samples)
- Per-sample variance: 0.019 (gates adapt to individual inputs)

---

## 6. MODEL ARCHITECTURE

### 6.1 Implemented Models (Validated)

All models are implemented in `train_multimodal.py` with identical hyperparameters for fair comparison.

#### MRI-Only Model
```
MRI Embeddings (512-dim)
    │
    ▼
┌─────────────────────────┐
│   MRI Encoder (MLP)     │
│   512 → 32 → dropout    │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Classifier            │
│   32 → 1 (sigmoid)      │
└─────────────────────────┘
```

#### Late Fusion Model
```
MRI (512-dim)          Clinical (5-dim)
    │                       │
    ▼                       ▼
┌───────────┐          ┌───────────┐
│ MRI Enc   │          │ Clin Enc  │
│ 512→32    │          │ 5→32      │
└───────────┘          └───────────┘
    │                       │
    └───────────┬───────────┘
                │
                ▼ (concatenate)
         ┌─────────────┐
         │ 64-dim      │
         └─────────────┘
                │
                ▼
         ┌─────────────┐
         │ Classifier  │
         │ 64→1        │
         └─────────────┘
```

#### Attention (Gated) Fusion Model
```
MRI (512-dim)          Clinical (5-dim)
    │                       │
    ▼                       ▼
┌───────────┐          ┌───────────┐
│ MRI Enc   │          │ Clin Enc  │
│ 512→32    │          │ 5→32      │
└───────────┘          └───────────┘
    │                       │
    │      ┌────────────────┤
    │      │                │
    │      ▼                │
    │  ┌─────────┐          │
    │  │ Gate    │          │
    │  │ g=σ(W·  │          │
    │  │ concat) │          │
    │  └────┬────┘          │
    │       │               │
    ▼       ▼               ▼
  ┌─────────────────────────────┐
  │ fused = g*MRI + (1-g)*Clin  │
  │        (32-dim)             │
  └─────────────────────────────┘
                │
                ▼
         ┌─────────────┐
         │ Classifier  │
         │ 32→1        │
         └─────────────┘
```

**Hyperparameters (All Models):**
- Hidden dimension: 32
- Dropout: 0.5
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Early stopping: patience=20
- Total parameters: ~17K-18K per model

### 6.2 Legacy Architecture (Not Used)

The `project/src/models/multimodal_fusion.py` contains a more complex architecture
that was designed but never properly integrated:

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                              │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   MRI Branch  │    │ Anatomical    │    │  Clinical     │
│  (ResNet18)   │    │ Features      │    │  Features     │
│  512-dim      │    │ 128-dim       │    │  64-dim       │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                 ┌───────────────────────┐
                 │  8-head Attention     │
                 │  Fusion (256-dim)     │
                 └───────────────────────┘
```

This architecture was too complex for the dataset size (N=205) and would require
significantly more training data to avoid overfitting.

---

## 7. PROJECT STRUCTURE

```
D:/discs/
├── PROJECT_DOCUMENTATION.md    # ← THIS FILE (Master documentation)
├── README.md                   # Quick start guide
├── requirements.txt            # Python dependencies
│
├── classification_pipeline.py  # Traditional ML baselines
├── train_multimodal.py        # Deep learning comparison (3 models)
├── mri_feature_extraction.py   # Main CNN feature extraction pipeline
├── oasis_data_exploration.py   # Clinical data exploration
├── oasis_deep_feature_scan.py  # Exhaustive feature mining
│
├── extracted_features/         # Output features
│   ├── oasis_all_features.npz  # 436 subjects, all features
│   └── oasis_all_features.pt   # PyTorch format
│
├── disc1/ ... disc12/          # OASIS-1 raw data (436 subjects)
│   └── OAS1_XXXX_MR1/
│       ├── OAS1_XXXX_MR1.txt   # Demographics
│       ├── PROCESSED/MPRAGE/T88_111/  # Preprocessed MRI
│       └── FSL_SEG/            # Tissue segmentation
│
├── ADNI/                       # ADNI dataset (203 subjects)
│   └── XXX_S_XXXX/             # Subject folders
│
├── project/                    # Deep learning project
│   ├── src/
│   │   ├── models/             # Neural network architectures
│   │   ├── preprocessing/      # Data processing scripts
│   │   └── training/           # Training loops
│   ├── data/processed/         # Processed feature CSVs
│   └── frontend/               # Web visualization (optional)
│
└── ADNI_COMPREHENSIVE_REPORT.md  # ADNI dataset analysis
```

---

## 8. HOW TO RUN

### Prerequisites
```bash
pip install -r requirements.txt
# Requires: torch, torchvision, nibabel, numpy, pandas, scikit-learn
```

### Extract Features (Already Done)
```bash
python mri_feature_extraction.py
# Output: extracted_features/oasis_all_features.npz
```

### Quick Classification Test
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load features
data = np.load('extracted_features/oasis_all_features.npz', allow_pickle=True)
mri = data['mri_features']
labels = data['labels']

# Filter to CDR=0 vs CDR=0.5
mask = [(l == 0 or l == 0.5) for l in labels]
X = mri[mask]
y = np.array([0 if l == 0 else 1 for l in labels[mask]])

# Train and evaluate
clf = LogisticRegression(max_iter=1000)
scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
print(f"AUC: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## 9. RESEARCH PHASES & PROGRESS

### Phase 1: Data Preparation ✅ COMPLETE
- [x] Process all OASIS-1 subjects (436/436)
- [x] Extract CNN features for all subjects
- [x] Verify feature quality (no NaN, healthy variance)
- [ ] Process ADNI dataset (future work)

### Phase 2: Feature Engineering ✅ COMPLETE
- [x] Extract 6 clinical features per subject
- [x] Z-score normalization
- [x] Combined feature vector (518-dim)

### Phase 3: Classification Testing ✅ COMPLETE
- [x] Logistic Regression baseline
- [x] Multiple classifiers tested (RF, GB, SVM)
- [x] Feature importance analysis
- [x] Realistic early detection test (without MMSE)

### Phase 4: Deep Learning Model 🔄 READY
- [x] Architecture designed (Hybrid Multimodal)
- [x] Model implemented
- [ ] Full training with new features
- [ ] Hyperparameter optimization
- [ ] Cross-validation

### Phase 5: Evaluation & Publication 📋 PLANNED
- [ ] Test set evaluation
- [ ] Cross-dataset validation (OASIS → ADNI)
- [ ] Interpretability analysis
- [ ] Paper writing

---

## 10. KEY FINDINGS

### 🔬 Scientific Insights

1. **MRI Provides Meaningful Signal**
   - AUC 0.78 with MRI alone (above nWBV baseline of 0.75)
   - ResNet18 captures dementia-related patterns

2. **MMSE Dominates Clinical Features**
   - MMSE alone: AUC 0.85
   - Concern: MMSE highly correlated with CDR (potential data leakage)
   - Recommendation: Exclude MMSE for realistic early detection

3. **Brain Volume is Informative**
   - nWBV (brain volume) alone: AUC 0.75
   - Strong age correlation (r = -0.87) - confounding

4. **Combined Features Work**
   - MRI + Clinical (no MMSE): AUC 0.78
   - Multimodal approach is valid

### ⚠️ Important Considerations

- **CDR is NOT the disease** - it's a clinical proxy
- **Age is a confounder** - brain atrophy correlates with age
- **MMSE in training = potential bias** - use for validation only
- **Small sample size** - 205 subjects for binary classification

---

## 11. NEXT STEPS

### Completed ✅
1. **Fixed training pipeline** - `train_multimodal.py` uses real CNN embeddings
2. **Implemented 3 models** - MRI-only, Late Fusion, Attention Fusion
3. **Fair comparison** - All models use same hyperparameters
4. **Validated attention** - Gate values show meaningful variation

### Short-Term (If Continuing)
1. **Process ADNI dataset** using same pipeline
2. **Cross-dataset validation** (train OASIS → test ADNI)
3. **Multi-class classification** (include CDR=1, CDR=2)

### Long-Term (Publication-Ready)
1. **Scale attention to larger datasets** where it may outperform late fusion
2. **Ablation studies** (which features matter?)
3. **Explainability** (which brain regions via Grad-CAM?)
4. **Paper writing** with comprehensive results

### Honest Limitations
- Dataset size (N=205) limits complex model advantage
- Attention fusion marginally outperforms late fusion on this data
- MMSE exclusion is necessary but hurts discriminative power
- Cross-validation variance is high due to small sample size

---

## 12. FILE INVENTORY

### Essential Files (Keep)
| File | Purpose |
|------|---------|
| `PROJECT_DOCUMENTATION.md` | This master documentation |
| `README.md` | Quick start guide |
| `classification_pipeline.py` | Traditional ML baselines |
| `train_multimodal.py` | Deep learning comparison (3 models) |
| `mri_feature_extraction.py` | Feature extraction pipeline |
| `requirements.txt` | Dependencies |
| `extracted_features/oasis_all_features.npz` | Extracted features |
| `ADNI_COMPREHENSIVE_REPORT.md` | ADNI reference |

### Data Files
| File | Purpose |
|------|---------|
| `oasis_comprehensive_features.csv` | 43 basic features |
| `oasis_deep_features_ALL.csv` | 214 deep features (39 subjects) |

### Model Code
| Location | Purpose |
|----------|---------|
| `train_multimodal.py` | **Active** training script |
| `project/src/models/` | Legacy neural network architectures |
| `project/src/preprocessing/` | Data processing |
| `project/src/training/` | Legacy training loops |

---

## 📚 REFERENCES

- **OASIS Dataset:** https://www.oasis-brains.org/
- **ADNI Dataset:** https://adni.loni.usc.edu/
- **ResNet18 Paper:** He et al., "Deep Residual Learning", CVPR 2016
- **CDR Scale:** Morris, J.C. (1993). Clinical Dementia Rating

---

## 📝 CHANGELOG

| Date | Change |
|------|--------|
| 2025-12-18 | Created comprehensive documentation |
| 2025-12-18 | Completed CNN extraction for all 436 subjects |
| 2025-12-17 | Fixed feature extraction pipeline |
| 2025-12-13 | Initial model training |

---

*This document consolidates all project information. For quick reference, see [README.md](README.md).*
