# 🧠 MASTER PROJECT DOCUMENTATION

**Deep Learning-Based Multimodal MRI Analysis for Early Detection of Neurological Diseases**

**Last Updated:** December 27, 2025  
**Status:** ✅ Cross-Sectional Complete | ✅ Longitudinal Experiment Complete | 📊 Results Analyzed

---

## 📋 TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [Dataset Information](#3-dataset-information)
4. [Feature Extraction Pipeline](#4-feature-extraction-pipeline)
5. [Classification Results](#5-classification-results)
6. [Model Architecture](#6-model-architecture)
7. [Project Structure](#7-project-structure)
8. [Longitudinal Progression Experiment](#8-longitudinal-progression-experiment)
9. [How to Run](#9-how-to-run)
10. [Research Phases & Progress](#10-research-phases--progress)
11. [Key Findings](#11-key-findings)
12. [Next Steps](#12-next-steps)
13. [File Inventory](#13-file-inventory)

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
| ADNI Integration | ✅ Complete | 629 subjects, cross-dataset transfer |
| **Longitudinal Experiment** | ✅ NEW | 2,262 scans, progression prediction |

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

### ADNI Dataset (Integration In-Progress)
| Attribute | Value |
|-----------|-------|
| **Total Unique Subjects** | **625** (Combined from all folders) |
| **From Project Folder** | 404 subjects (`D:/discs/ADNI`) |
| **From Downloads** | 605 subjects (`Downloads` folder) |
| **Labels** | ✅ Found (CN, MCI, AD) in `ADNI1_Complete_1Yr_1.5T_12_19_2025.csv` |
| **Status** | 🔄 Feature Extraction Running |
| **Output** | `extracted_features/adni_features.csv` (MRI Feature + Group Label) |

**Integration Strategy:**
Data is physically split between the workspace and the user's Downloads folder. A custom inventory script (`adni_inventory_check.py`) maps these files, and a batch extractor (`adni_batch_feature_extraction.py`) consolidates them into a single labeled dataset.

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
│
├── 📄 DOCUMENTATION FILES
│   ├── README.md                              # Main project README with research summary
│   ├── PROJECT_DOCUMENTATION.md               # ← THIS FILE (Master documentation)
│   ├── DATA_CLEANING_AND_PREPROCESSING.md     # 20+ pages thesis-ready data cleaning docs
│   ├── PROJECT_ASSESSMENT_HONEST_TAKE.md      # 15+ pages honest analysis of fusion results
│   ├── REALISTIC_PATH_TO_PUBLICATION.md       # 12+ pages roadmap to publication
│   ├── RESEARCH_PAPER_FULL.md                 # Complete research paper draft
│   ├── RESEARCH_PAPER_IEEE_FORMAT.md          # IEEE formatted paper version
│   ├── ADNIMERGE_USAGE_SUMMARY.md             # Analysis of ADNIMERGE data utilization
│   ├── DEPLOYMENT_GUIDE.md                    # Frontend deployment guide (Vercel)
│   └── README_FIGURES.md                      # Figure descriptions for paper
│
├── 📊 EXTRACTED FEATURES (Output)
│   └── extracted_features/
│       ├── oasis_all_features.npz             # OASIS: 436 subjects (1.83 MB)
│       ├── oasis_all_features.pt              # PyTorch tensor format (1.87 MB)
│       ├── adni_features.csv                  # ADNI: 1,325 feature vectors (7.1 MB)
│       └── checkpoint.json                    # Feature extraction progress
│
├── 📈 FIGURES (Research Visualizations)
│   └── figures/
│       ├── A1_oasis_model_comparison.png      # OASIS model comparison chart
│       ├── A2_oasis_class_distribution.png    # OASIS class distribution
│       ├── B1_adni_level1_honest.png          # ADNI Level-1 honest results
│       ├── B2_level1_vs_level2_circularity.png # Circularity comparison
│       ├── B3_adni_class_distribution.png     # ADNI class distribution
│       ├── C1_in_vs_cross_dataset_collapse.png # Cross-dataset performance
│       ├── C2_transfer_robustness_heatmap.png # Transfer learning heatmap
│       ├── C3_auc_drop_robustness.png         # AUC drop analysis
│       ├── D1_preprocessing_pipeline.png      # Data preprocessing pipeline
│       ├── D2_sample_size_reduction.png       # Sample filtering visualization
│       ├── D3_age_distribution.png            # Age distribution charts
│       ├── D4_sex_distribution.png            # Sex distribution charts
│       └── D5_feature_dimensions.png          # Feature dimension analysis
│
├── 🧠 OASIS-1 RAW DATA
│   └── disc1/ ... disc12/                     # 12 discs containing 436 subjects
│       └── OAS1_XXXX_MR1/                     # Subject folder
│           ├── OAS1_XXXX_MR1.txt              # Demographics & clinical data
│           ├── PROCESSED/MPRAGE/T88_111/      # Preprocessed MRI (Talairach space)
│           │   └── OAS1_XXXX_MR1_mpr_n4_anon_sbj_111.*  # .hdr/.img files
│           └── FSL_SEG/                       # Tissue segmentation masks
│
├── 🧬 ADNI RAW DATA
│   └── ADNI/
│       ├── XXX_S_XXXX/                        # 404 subject folders with NIfTI scans
│       │   └── *.nii                          # Structural MRI (NIfTI format)
│       ├── ADNIMERGE_23Dec2025.csv            # Complete ADNI clinical data (13.26 MB)
│       ├── ADNI1_Complete_1Yr_1.5T_*.csv      # ADNI-1 metadata files
│       └── (230 total NIfTI scans from 203 unique subjects)
│
├── 🚀 PROJECT (Main Deep Learning Codebase)
│   └── project/
│       ├── .gitignore
│       │
│       ├── 📁 frontend/                       # Next.js 16 Web Application
│       │   ├── package.json                   # Dependencies
│       │   ├── next.config.ts                 # Next.js configuration
│       │   ├── vercel.json                    # Vercel deployment config
│       │   ├── public/                        # Static assets (14 files)
│       │   │   └── *.md                       # Downloadable documentation
│       │   └── src/
│       │       ├── app/                       # Next.js App Router pages
│       │       │   ├── page.tsx               # Homepage with 3D brain viz
│       │       │   ├── documentation/         # Documentation hub page
│       │       │   ├── dataset/               # OASIS dataset page
│       │       │   ├── adni/                  # ADNI dataset page
│       │       │   └── results/               # Results visualization page
│       │       ├── components/                # 42 React components
│       │       │   ├── hero-3d.tsx            # 3D brain visualization
│       │       │   └── ui/                    # shadcn/ui components
│       │       ├── hooks/                     # Custom React hooks
│       │       ├── lib/                       # Utility functions
│       │       └── styles/                    # CSS stylesheets
│       │
│       ├── 📁 scripts/                        # Python training & extraction scripts
│       │   ├── classification_pipeline.py    # Traditional ML baselines (13 KB)
│       │   ├── train_multimodal.py           # DL model comparison - 3 models (23 KB)
│       │   ├── mri_feature_extraction.py     # ResNet18 CNN extraction (47 KB)
│       │   ├── oasis_data_exploration.py     # Clinical data exploration (31 KB)
│       │   ├── oasis_deep_feature_scan.py    # Deep feature mining (38 KB)
│       │   └── deep_analysis.py              # Analysis utilities (26 KB)
│       │
│       ├── 📁 src/                            # Source modules
│       │   ├── models/                        # Neural network architectures
│       │   │   └── multimodal_fusion.py      # Fusion model definitions
│       │   ├── preprocessing/                 # Data processing (7 scripts)
│       │   ├── training/                      # Training loops (2 scripts)
│       │   ├── evaluation/                    # Evaluation metrics (3 scripts)
│       │   └── utils/                         # Helper utilities
│       │
│       ├── 📁 data/                           # Processed data CSVs
│       └── 📁 results/                        # Training results & metrics
│
├── 🔬 PROJECT_ADNI (ADNI-Specific Pipeline)
│   └── project_adni/
│       ├── README.md                          # ADNI pipeline documentation
│       ├── ADNI_COMPREHENSIVE_REPORT.md       # Detailed ADNI data analysis (12 KB)
│       ├── ADNI_INTEGRATION_GUIDE.md          # Integration instructions (14 KB)
│       │
│       ├── 📁 src/                            # ADNI training scripts
│       │   ├── train_level1.py               # Honest model - NO MMSE (16 KB)
│       │   ├── train_level2.py               # Circular model - WITH MMSE (17 KB)
│       │   ├── cross_dataset_robustness.py   # Transfer experiments (15 KB)
│       │   ├── baseline_selection.py         # Baseline scan selection (4 KB)
│       │   ├── data_split.py                 # Train/test splitting (3 KB)
│       │   ├── feature_extraction.py         # ADNI feature extraction (6 KB)
│       │   ├── file_matcher.py               # MRI-to-clinical matching (4 KB)
│       │   └── adnimerge_utils.py            # ADNIMERGE utilities (6 KB)
│       │
│       ├── 📁 scripts/                        # 28 utility scripts
│       │
│       ├── 📁 data/                           # ADNI processed data
│       │   ├── csv/                           # Train/test split CSVs
│       │   └── features/                      # Extracted feature files
│       │
│       └── 📁 results/                        # ADNI experiment results
│           ├── level1/                        # Honest baseline results
│           │   └── metrics.json               # Level-1 performance metrics
│           ├── level2/                        # Circular (MMSE) results
│           │   └── metrics.json               # Level-2 performance metrics
│           └── reports/                       # Cross-dataset reports
│
├── � PROJECT_LONGITUDINAL (NEW - Progression Experiment)
│   └── project_longitudinal/
│       ├── README.md                          # Longitudinal experiment overview
│       ├── src/
│       │   ├── data_inventory.py              # Scan all 2,294 NIfTI files
│       │   ├── data_preparation.py            # Create progression labels
│       │   ├── feature_extraction.py          # Extract per-scan features
│       │   ├── train_single_scan.py           # Single-scan baseline model
│       │   ├── train_delta_model.py           # Change-based delta model
│       │   ├── train_sequence_model.py        # LSTM sequence model
│       │   └── evaluate.py                    # Generate comparison report
│       ├── data/
│       │   ├── processed/                     # subject_inventory.csv, splits
│       │   └── features/                      # longitudinal_features.npz (4.65 MB)
│       ├── results/                           # Model metrics (JSON)
│       │   ├── single_scan/metrics.json
│       │   ├── delta_model/metrics.json
│       │   ├── sequence_model/metrics.json
│       │   └── comparison_report.md
│       └── docs/                              # Documentation
│           ├── TASK_DEFINITION.md
│           ├── LEAKAGE_PREVENTION.md
│           └── RESULTS_SUMMARY.md
│
├── �🐍 ROOT PYTHON SCRIPTS
│   ├── check_adnimerge_usage.py              # Analyze ADNIMERGE utilization
│   ├── visualize_adnimerge_usage.py          # Generate usage visualizations
│   ├── generate_adni_json.py                 # Generate ADNI metadata JSON
│   ├── extract_adni_samples.py               # Sample extraction utilities
│   ├── quick_adni_check.py                   # Quick data verification
│   ├── generate_data_figures.py              # Generate paper figures (20 KB)
│   ├── generate_visualizations.py            # Visualization utilities (19 KB)
│   └── generate_interpretability_images.py   # Interpretability visualizations
│
├── ⚙️ CONFIGURATION FILES
│   ├── requirements.txt                       # Python dependencies
│   └── .gitignore                            # Git ignore patterns
│
└── 📦 OTHER FILES
    ├── home-image.png                         # Homepage hero image (1.3 MB)
    ├── ADNIMERGE_usage_visualization.png      # Data usage chart
    ├── adnimerge_usage_report.txt             # Usage report text
    ├── robustness_results.txt                 # Cross-dataset robustness results
    └── plan.txt                               # Project planning notes
```

### Key Directory Purposes:

| Directory | Purpose |
|-----------|---------|
| `extracted_features/` | Pre-computed CNN features for fast training |
| `figures/` | Publication-ready visualizations |
| `project/frontend/` | Live website at neuroscope-mri.vercel.app (static) |
| `project/scripts/` | Main training & extraction scripts |
| `project_adni/` | ADNI-specific experiments (Level-1, Level-2, cross-dataset) |
| **`project_longitudinal/`** | **NEW: Longitudinal progression experiment (2,262 scans)** |
| `disc1-12/` | OASIS-1 raw MRI data (436 subjects) |
| `ADNI/` | ADNI raw data (404 folders, 203 unique subjects, 230 scans) |

### Data Scale Summary:

| Dataset | Raw Scans | Unique Subjects | Features Extracted |
|---------|-----------|-----------------|-------------------|
| **OASIS-1** | 436 | 436 | 436 × 518 dims |
| **ADNI-1 (Cross-Sectional)** | 1,825 | 629 → 203 (available) | 1,325 vectors |
| **ADNI-1 (Longitudinal)** | **2,262** | **629** | **2,262 × 512 dims** |
| **Total** | **4,523** | **1,065** | **~4,000+** |

---

## 8. LONGITUDINAL PROGRESSION EXPERIMENT

### 🎯 Research Question
> **Does observing CHANGE over time (multiple MRIs of the same person) help detect or predict dementia progression more reliably?**

### 8.1 Experiment Overview

| Aspect | Cross-Sectional (Baseline) | Longitudinal (This Section) |
|--------|---------------------------|----------------------------|
| Scans per subject | 1 (baseline only) | ALL available (avg 3.6) |
| Total scans | 629 | 2,262 |
| Task | Detection (snapshot) | Progression prediction |
| Question | "Detect from single scan" | "Predict from change over time" |

### 8.2 Data Preparation

**Source:** `C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T\ADNI`

| Metric | Value |
|--------|-------|
| Total NIfTI Scans | 2,294 |
| Unique Subjects | 639 |
| Scans After Filtering | 2,262 |
| Train Subjects | 503 |
| Test Subjects | 126 |
| Stable Subjects | 403 (64%) |
| Converter Subjects | 226 (36%) |

**Progression Labels:**
- **Stable (Label=0):** Diagnosis unchanged from baseline to last visit
- **Converter (Label=1):** Diagnosis worsened (CN→MCI, CN→AD, MCI→AD)

### 8.4 Phase 1 Results: Initial ResNet Experiment

| Model | AUC | AUPRC | Accuracy | Description |
|-------|-----|-------|----------|-------------|
| Single-Scan (Baseline) | 0.510 | 0.370 | 57.9% | Uses only first visit |
| Delta Model | 0.517 | 0.434 | 54.0% | Baseline + follow-up + change |
| Sequence Model (LSTM) | 0.441 | 0.366 | 47.6% | All visits as sequence |

**Initial Observation:** All models achieved near-chance performance. This prompted deep investigation.

### 8.5 Phase 2: Deep Investigation

**Issues Discovered:**

1. **Label Contamination**
   - 136 Dementia patients labeled "Stable" (they can't progress further!)
   - Both healthy (CN) and severely impaired (Dementia) labeled the same way
   - Model sees contradictory signal

2. **Wrong Feature Type**
   - ResNet18 trained on ImageNet (cats, dogs, cars)
   - Features are scale-invariant by design
   - Cannot capture absolute volume changes (atrophy)

3. **Feature Analysis**
   - Within-subject feature change: 0.129 (small)
   - Between-subject difference: 0.205 (larger)
   - ResNet features are STABLE over time - not capturing progression

### 8.6 Phase 3: Corrected Experiment with Actual Biomarkers

We re-ran using ADNIMERGE structural biomarkers instead of ResNet features:

**Individual Biomarker Power (MCI cohort, N=737):**

| Biomarker | AUC | Type |
|-----------|-----|------|
| **Hippocampus** | **0.725** | Structural (BEST!) |
| Entorhinal | 0.691 | Structural |
| MidTemp | 0.678 | Structural |
| ADAS13 | 0.767 | Cognitive (semi-circular) |
| APOE4 | 0.624 | Genetic |

**Feature Combination Results:**

| Approach | AUC | vs ResNet |
|----------|-----|-----------|
| ResNet features | 0.52 | baseline |
| Biomarkers (baseline only) | 0.74 | +22 points |
| **Biomarkers + Longitudinal Change** | **0.83** | **+31 points** |
| + Age + APOE4 | 0.81 | +29 points |
| + ADAS13 (cognitive) | 0.84 | +32 points |

### 8.7 Key Discoveries

1. **Longitudinal Data DOES Help**
   - Adding temporal change improves AUC by +9.5 points (0.74 → 0.83)
   - Hippocampal atrophy RATE is a powerful predictor

2. **APOE4 Genetic Risk**
   - 0 alleles: 23.5% conversion rate
   - 1 allele: 44.2% conversion rate
   - 2 alleles: 49.1% conversion rate
   - **Carriers have DOUBLE the risk!**

3. **Right Features > Complex Models**
   - Logistic regression with biomarkers: 0.83 AUC
   - LSTM with ResNet features: 0.44 AUC
   - **Simple models win with proper features**

4. **Education Doesn't Predict Progression**
   - ~32% conversion rate across all education levels
   - Not a protective factor in MCI population

### 8.8 Leakage Prevention

| Measure | Implementation |
|---------|----------------|
| Subject-Level Split | No subject in both train/test |
| Future Labels Only | Labels from FINAL diagnosis, not baseline |
| Separate Normalization | Scaler fit on train, applied to test |
| Isolated Experiment | Completely separate from cross-sectional work |

### 8.9 Conclusions

> **Phase 1 Conclusion (ResNet Features):**
> ResNet features provide only marginal improvement (+1.3%) for progression prediction. This is an honest negative result.

> **Phase 3 Conclusion (Biomarkers):**
> Proper structural biomarkers (hippocampus, ventricles, entorhinal) achieve **0.83 AUC** for MCI→Dementia prediction, with longitudinal change adding **+9.5 percentage points** over baseline-only models.

**The key insight:** Longitudinal data **DOES help**, but requires disease-specific biomarkers, not generic CNN features.

### 8.10 Project Location

All longitudinal code and results in: `project_longitudinal/`
```
project_longitudinal/
├── src/                      # Python scripts
│   ├── data_inventory.py     # Scan all 2,294 NIfTI files
│   ├── data_preparation.py   # Create progression labels
│   ├── feature_extraction.py # Extract per-scan features
│   ├── train_single_scan.py  # Baseline model
│   ├── train_delta_model.py  # Change-based model
│   ├── train_sequence_model.py # LSTM sequence model
│   └── evaluate.py           # Generate comparison report
├── data/features/            # longitudinal_features.npz (4.65 MB)
├── results/                  # Model metrics JSON files
│   └── biomarker_analysis/   # NEW: Biomarker experiment results
└── docs/
    ├── TASK_DEFINITION.md
    ├── LEAKAGE_PREVENTION.md
    ├── RESULTS_SUMMARY.md
    └── INVESTIGATION_REPORT.md  # Complete analysis (15+ findings)
```

---

## 9. HOW TO RUN

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
