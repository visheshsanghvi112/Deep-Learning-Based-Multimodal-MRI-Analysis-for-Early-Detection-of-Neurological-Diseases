# 🧠 MASTER PROJECT DOCUMENTATION

**Deep Learning-Based Multimodal MRI Analysis for Early Detection of Neurological Diseases**

**Last Updated:** January 2, 2026  
**Status:** ✅ Cross-Sectional Complete | ✅ Longitudinal Experiment Complete | ✅ Level-MAX Biomarker Fusion Complete | 📊 Results Analyzed

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
9. [Level-MAX Experiment (New)](#9-level-max-experiment-biomarker-fusion)
10. [How to Run](#10-how-to-run)
11. [Research Phases & Progress](#11-research-phases--progress)
12. [Key Findings](#12-key-findings)
13. [Next Steps](#13-next-steps)
14. [File Inventory](#14-file-inventory)

---

## 1. EXECUTIVE SUMMARY

### 🎯 Research Goal
Develop deep learning models to detect **early-stage dementia** by combining:
- **MRI-based deep features** (ResNet18 CNN embeddings)
- **Clinical/demographic features** (Age, MMSE, brain volumes, genetics, CSF)

### ✅ Key Achievements
| Milestone | Status | Details |
|-----------|--------|---------|
| Data Processing | ✅ Complete | 436/436 OASIS-1 subjects processed |
| CNN Feature Extraction | ✅ Complete | 512-dim ResNet18 features for all subjects |
| Clinical Features | ✅ Complete | Normalized clinical features |
| Traditional ML Classification | ✅ Complete | Late Fusion AUC 0.794 (realistic scenario) |
| Deep Learning Models | ✅ Complete | All 3 models trained and compared |
| ADNI Integration | ✅ Complete | 629 subjects, cross-dataset transfer |
| **Level-MAX Biomarker Fusion** | ✅ NEW | **14-feature biological profile, 0.81 AUC** |
| **Longitudinal Experiment** | ✅ NEW | 2,262 scans, progression prediction (0.83 AUC) |

### 🏆 Classification Results Summary

**A. OASIS-1 (Standard Benchmark - N=436)**
*Scenario: Without MMSE (Realistic Early Detection)*
```
Traditional ML (Logistic Regression):
  - MRI only:           AUC = 0.770 ± 0.080
  - Late Fusion:        AUC = 0.794 ± 0.083  ← +2.3% over MRI

Deep Learning (5-fold CV):
  - MRI-Only DL:        AUC = 0.781 ± 0.087
  - Late Fusion DL:     AUC = 0.796 ± 0.092  ← +1.5% over MRI
  - Attention Fusion:   AUC = 0.790 ± 0.109  ← +1.0% over MRI
```

**B. ADNI Cross-Sectional (N=629 - Three-Tiered Performance)**
We stratified performance based on "Honesty" vs "Feature Quality":

| Tier | Experiment Level | Features | AUC | Status |
|------|------------------|----------|-----|--------|
| **1** | **Level-1 (Baseline)** | MRI + Age/Sex | **0.60** | **Fail.** Demographics too weak. |
| **2** | **Level-MAX (Optimal)** | MRI + Bio-Profile* | **0.81** | **Success.** Detects true pathology. |
| **3** | **Level-2 (Circular)** | MRI + MMSE/CDR | **0.99** | **Cheating.** Uses diagnostic scores. |

*Bio-Profile*: Hippocampus, Ventricles, Entorhinal, Fusiform, MidTemp, WholeBrain, ICV, APOE4, Aβ42, Tau, pTau, Age, Sex, Education.

**Key Finding:** The fusion architecture was never broken—it was starved of quality information. Providing biological markers boosted performance by **+16.5%**.

---

## 2. PROJECT OVERVIEW

### Research Context
- **Disease Focus:** Alzheimer's Disease / Dementia
- **Classification Task:** CDR=0 (Normal) vs CDR=0.5+ (Very Mild) / CN vs MCI+AD
- **Approach:** Multimodal deep learning (MRI + Clinical) using Late Fusion and Attention Mechanisms.

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

### ADNI Dataset (Project Bench)
| Attribute | Value |
|-----------|-------|
| **Total Unique Subjects** | **629** (Baseline) |
| **Longitudinal Scans** | **2,262** total visits |
| **Biomarker Depth** | **High.** Includes Genetics (APOE4), CSF (Amyloid/Tau), Volumetrics. |
| **Labels** | CN (Cognitively Normal), MCI (Mild Cognitive Impairment), AD (Alzheimer's) |
| **Location** | `D:/discs/ADNI` + `ADNIMERGE.csv` |

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

**Script:** `mri_feature_extraction.py` / `feature_extraction.py`

### 4.2 Clinical Feature Extraction

We utilized two distinct clinical profiles depending on the experiment level:

**A. Basic Profile (OASIS / ADNI Level-1)**
Used for baseline establishment.
| Feature | Normalization |
|---------|---------------|
| AGE | Z-score (μ=50, σ=20) |
| Sex | 0/1 Encoding |
| nWBV | Z-score (OASIS only) |
| eTIV | Z-score (OASIS only) |
| ASF | Z-score (OASIS only) |
| EDUC | Z-score (OASIS only) |

**B. Extended Bio-Profile (ADNI Level-MAX only - 14 Dimensions)**
Used for the "Overpowered" experiment.
| Category | Features |
|----------|----------|
| **Genetics** | APOE4 (0, 1, 2 alleles) |
| **CSF Biomarkers**| ABETA, TAU, PTAU (mg/mL) |
| **Volumetrics** | Hippocampus, Entorhinal, Ventricles, Fusiform, MidTemp, ICV, WholeBrain |
| **Demographics** | Age, Sex, Education (PTEDUCAT) |

### 4.3 Output Files
| File | Format | Contents |
|------|--------|----------|
| `extracted_features/oasis_all_features.npz` | NumPy | All OASIS 518-dim features |
| `project_adni/data/features/train_level1.csv` | CSV | ADNI Level-1 (MRI+Age/Sex) |
| `project_adni/data/features/train_level_max.csv` | CSV | ADNI Level-MAX (MRI+Bio) |
| `longitudinal_features.npz` | NumPy | 2,262 scans MRI features |

---

## 5. CLASSIFICATION RESULTS (DETAILED)

### 5.1 OASIS-1 Results (N=436)

**Scenario: Without MMSE (Honest)**
| Model | Mean AUC | Std | Accuracy |
|-------|----------|-----|----------|
| MRI-Only DL | 0.781 | ±0.087 | 74.2% |
| **Late Fusion DL** | **0.796** | **±0.092** | **76.1%** |
| Attention Fusion | 0.790 | ±0.109 | 75.0% |

**Key Findings:**
1. **Multimodal improves over unimodal:** +1.5% gain confirms clinical features provide complementary signal.
2. **Late fusion is competitive:** Simple concatenation performs as well as attention on this small dataset.
3. **Attention shows meaningful behavior:** Gate values vary (std=0.157), indicating dynamic weighting.

### 5.2 ADNI Results (N=629)

**Level-1: Honest Baseline (MRI + Age/Sex)**
| Model | AUC | 95% CI |
|-------|-----|--------|
| MRI-Only | 0.583 | 0.47-0.68 |
| Late Fusion | 0.598 | 0.49-0.70 |
*Interpretation: Fusion fails because Age/Sex are weak predictors.*

**Level-MAX: Biomarker Fusion (MRI + 14 Bio-Features)**
| Model | AUC | Accuracy | Gain |
|-------|-----|----------|------|
| MRI-Only | 0.643 | 62.7% | Baseline |
| **Late Fusion** | **0.808** | **76.2%** | **+16.5%** |
| **Attention Fusion** | **0.808** | **75.4%** | **+16.5%** |
*Interpretation: Fusion succeeds brilliantly. The biology (Hippocampus, CSF) fills the gaps in the MRI model.*

---

## 6. MODEL ARCHITECTURE

### 6.1 Implemented Models (Validated)

**MRI-Only Model**
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

**Late Fusion Model**
```
MRI (512-dim)          Clinical (N-dim)
    │                       │
    ▼                       ▼
┌───────────┐          ┌───────────┐
│ MRI Enc   │          │ Clin Enc  │
│ 512→32    │          │ N→32      │
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

**Attention (Gated) Fusion Model**
```
MRI (512-dim)          Clinical (N-dim)
    │                       │
    ▼                       ▼
┌───────────┐          ┌───────────┐
│ MRI Enc   │          │ Clin Enc  │
│ 512→32    │          │ N→32      │
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

---

## 7. PROJECT STRUCTURE

```
```
D:/discs/
│
├── 📄 DOCUMENTATION FILES
│   ├── README.md                              # Main project README with research summary
│   ├── PROJECT_DOCUMENTATION.md               # 👈 THIS FILE (Master documentation)
│   ├── DATA_CLEANING_AND_PREPROCESSING.md     # 20+ pages thesis-ready data cleaning docs
│   ├── PROJECT_ASSESSMENT_HONEST_TAKE.md      # 15+ pages honest analysis of fusion results
│   ├── REALISTIC_PATH_TO_PUBLICATION.md       # 12+ pages roadmap to publication
│   ├── RESEARCH_PAPER_FULL.md                 # Complete research paper draft
│   ├── RESEARCH_PAPER_IEEE_FORMAT.md          # IEEE formatted paper version
│   ├── ADNIMERGE_USAGE_SUMMARY.md             # Analysis of ADNIMERGE data utilization
│   ├── DEPLOYMENT_GUIDE.md                    # Frontend deployment guide (Vercel)
│   ├── LEVEL_MAX_RESULTS.md                   # 🆕 Detailed Level-MAX Findings
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
│       ├── � data/                           # Processed data CSVs
│       └── 📁 results/                        # Training results & metrics
│
├── �🔬 PROJECT_ADNI (ADNI-Specific Pipeline)
│   └── project_adni/
│       ├── README.md                          # ADNI pipeline documentation
│       ├── ADNI_COMPREHENSIVE_REPORT.md       # Detailed ADNI data analysis (12 KB)
│       ├── ADNI_INTEGRATION_GUIDE.md          # Integration instructions (14 KB)
│       │
│       ├── 📁 src/                            # ADNI training scripts
│       │   ├── train_level1.py               # Honest model - NO MMSE (16 KB)
│       │   ├── train_level2.py               # Circular model - WITH MMSE (17 KB)
│       │   ├── train_level_max.py            # 🆕 Level-MAX (MRI + Biomarkers)
│       │   ├── create_level_max_dataset.py   # 🆕 Dataset Builder
│       │   ├── visualize_level_max.py        # 🆕 Level-MAX Visualizations
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
│       │       ├── train_level1.csv           # Baseline features
│       │       └── train_level_max.csv        # 🆕 Biomarker features
│       │
│       └── 📁 results/                        # ADNI experiment results
│           ├── level1/                        # Honest baseline results
│           │   └── metrics.json               # Level-1 performance metrics
│           ├── level2/                        # Circular (MMSE) results
│           │   └── metrics.json               # Level-2 performance metrics
│           ├── level_max/                     # 🆕 Level-MAX results
│           │   ├── results.json               # Metric Breakdown
│           │   ├── roc_comparison.png         # ROC Curves
│           │   └── level_comparison.png       # AUC Comparison Bar Chart
│           └── reports/                       # Cross-dataset reports
│
├── ⏳ PROJECT_LONGITUDINAL (NEW - Progression Experiment)
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
├── 🐍 ROOT PYTHON SCRIPTS
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

### 8.2 Key Discoveries

**1. ResNet Features Fail for Progression (0.52 AUC)**
- Raw CNN features trained on ImageNet are robust to scale/shape.
- They fail to capture the subtle *volume loss* (atrophy) that defines progression.
- LSTM/Delta models trained on these features performed near chance.

**2. Biomarker Rates Succeed (0.83 AUC)**
- Tracking the **rate of change** of specific structures (Hippocampus, Ventricles, Entorhinal) is highly predictive.
- **Hippocampus Atrophy Rate** alone is a powerful predictor.

**3. Genetic Risk (APOE4)**
- Carriers of the APOE4 allele have significantly higher progression rates (49% vs 23%).

**4. Conclusion**
Longitudinal data **DOES help**, but it requires disease-specific biomarkers (volumetrics), not generic deep learning features.

---

## 9. LEVEL-MAX EXPERIMENT (BIOMARKER FUSION)

### 9.1 Motivation
The Level-1 fusion results on ADNI (0.60 AUC) revealed a critical insight: **the fusion architecture wasn't broken—it was starved of information.** With only Age and Sex as clinical features, there was insufficient complementary signal.

### 9.2 Implementation facts
- **Source:** ADNIMERGE baseline dataset (`VISCODE='bl'`)
- **Imputation:** Median imputation (fit on train) used for missing CSF/Volumetrics.
- **Scaling:** StandardScaler applied to all 14 clinical dimensions.
- **Models:** Re-used Late/Attention Fusion architectures.

### 9.3 Results & Insights
**AUC: 0.81 (+16.5% over MRI-Only)**

This experiment proved that:
1.  **Feature Quality is King:** Deep learning cannot conjure signal from noise. It needs quality inputs (Hippocampus > Age).
2.  **Fusion Works:** The architecture correctly integrated the MRI embeddings with the biological signals to boost performance.
3.  **Honest Performance:** We achieved this without using MMSE/CDR, meaning the model is detecting valid biological pathology, not just clinical symptoms.

---

## 10. HOW TO RUN

### Prerequisites
- Python 3.9+
- PyTorch with CUDA

### 1. Run Level-MAX (Recommended High-Performance Model)
```bash
# 1. Generate Dataset (Merge MRI + Biomarkers)
python project_adni/src/create_level_max_dataset.py

# 2. Train Models
python project_adni/src/train_level_max.py

# 3. Visualize Results (ROC Curves & Bar Charts)
python project_adni/src/visualize_level_max.py
```

### 2. Run OASIS Experiment
```bash
python project/scripts/train_multimodal.py
```

### 3. Run Longitudinal Analysis
```bash
python project_longitudinal/src/evaluate.py
```

---

## 11. RESEARCH PHASES & PROGRESS

| Phase | Description | Status | Outcome |
|-------|-------------|--------|---------|
| **Phase I** | OASIS-1 Proof of Concept | ✅ Done | 0.79 AUC (Validated Fusion) |
| **Phase II** | ADNI Level-1 (Baseline) | ✅ Done | 0.60 AUC (Identified Data Gap) |
| **Phase III** | Longitudinal Progression | ✅ Done | 0.83 AUC (Validated Atrophy Rates) |
| **Phase IV** | **ADNI Level-MAX** | ✅ **Done** | **0.81 AUC (Solved Fusion Paradox)** |
| **Phase V** | Cross-Dataset Transfer | ✅ Done | Confirmed generalization challenges |
| **Phase VI** | Final Publication | 🔄 In Progress | Documentation & Figures |

---

## 12. KEY FINDINGS

1.  **Multimodal Synergy is Real:** Fusion works, but only if the modalities have *high quality*. Fusing MRI with weak demographics (Level-1) does nothing. Fusing MRI with Biology (Level-MAX) adds +16%.
2.  **Avoid Circularity:** Achieving 0.99 AUC with cognitive scores (Level-2) is easy but clinically useless. The real challenge is achieving high performance *honestly* (Level-MAX).
3.  **Longitudinal Power:** Tracking disease *trajectory* (atrophy rates) is superior to snapshot analysis, provided you track the right structures.
4.  **Architecture Robustness:** Simple "Late Fusion" is often as effective as complex "Attention Fusion" for these dataset sizes (~600 subjects).

---

## 13. NEXT STEPS

1.  **Publication:** Compile Level-MAX and Longitudinal results into the final paper.
2.  **Robustness:** Run 5-seed average for Level-MAX to report variance (±0.01).
3.  **Explainability:** Generate SHAP plots for the Level-MAX clinical branch to quantify `Hippocampus` vs `APOE4` contribution.
4.  **Integration:** Potentially train a "Super Model" that uses Longitudinal data *plus* the Level-MAX biological profile.

---

## 14. FILE INVENTORY (Primary)

**OASIS:**
- `extracted_features/oasis_all_features.npz`
- `project/scripts/train_multimodal.py`

**ADNI:**
- `project_adni/src/train_level1.py`
- `project_adni/src/train_level_max.py`
- `project_adni/data/features/train_level_max.csv`

**Longitudinal:**
- `project_longitudinal/src/train_delta_model.py`
- `project_longitudinal/data/features/longitudinal_features.npz`

**Documentation:**
- `docs/PROJECT_DOCUMENTATION.md` (This File)
- `docs/LEVEL_MAX_RESULTS.md`
- `docs/PROJECT_ASSESSMENT_HONEST_TAKE.md`
