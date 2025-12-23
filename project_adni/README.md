# ADNI Multimodal Analysis Project

## Project Overview

This folder contains the **ADNI (Alzheimer's Disease Neuroimaging Initiative)** analysis pipeline, kept **completely separate** from the OASIS workflow to avoid any file conflicts.

## Dataset Information

- **Source**: ADNI1 Complete 1Yr 1.5T MRI scans
- **Subjects**: 629 unique subjects
- **Classes**: CN (194), MCI (302), AD (133)
- **Task**: Binary classification (CN vs MCI+AD)

## Directory Structure

```
project_adni/
├── src/                      # Source code
│   ├── baseline_selection.py # Select one scan per subject
│   ├── feature_extraction.py # ResNet18 MRI feature extraction
│   ├── file_matcher.py       # Match CSV to NIfTI files
│   ├── data_split.py         # Train/test split
│   ├── train_level1.py       # Level-1: Age+Sex only
│   ├── train_level2.py       # Level-2: MMSE+CDRSB (upper-bound)
│   └── adnimerge_utils.py    # ADNIMERGE parsing
├── data/
│   ├── csv/                  # Metadata & registry files
│   │   ├── ADNIMERGE.csv     # Full ADNIMERGE clinical data
│   │   ├── ADNI1_registry.csv # Image registry
│   │   ├── baseline_selection.csv
│   │   └── matched_files.csv
│   └── features/             # Extracted features & splits
│       ├── subject_features.csv  # 629 x 512 MRI features
│       ├── train_level1.csv      # Train split (Age+Sex)
│       ├── test_level1.csv       # Test split (Age+Sex)
│       ├── train_level2.csv      # Train split (+MMSE,CDRSB)
│       └── test_level2.csv       # Test split (+MMSE,CDRSB)
├── results/
│   ├── level1/               # Level-1 experiment results
│   ├── level2/               # Level-2 experiment results
│   └── reports/              # Analysis reports
└── scripts/                  # One-off utility scripts
```

## Experiment Levels

### Level-1: Basic Clinical Features (Primary)
- **Clinical features**: Age, Sex (2 features)
- **Purpose**: Realistic early-detection scenario
- **Results**: MRI-Only AUC=0.583, Late Fusion AUC=0.598

### Level-2: Full Clinical Features (Upper-Bound Reference)
- **Clinical features**: MMSE, CDRSB, Education, Age, APOE4 (5 features)
- **Purpose**: Upper-bound reference (NOT for early-detection claims)
- **WARNING**: MMSE/CDRSB are downstream cognitive measures
- **Results**: Late Fusion AUC=0.988 (but circular)

## Key Findings

1. **MRI-only ADNI performance lower than OASIS** (0.58 vs 0.72)
   - Different cohort, acquisition, preprocessing

2. **Minimal multimodal benefit with basic features** (+1.5% AUC)
   - Age+Sex provide limited diagnostic signal

3. **Massive benefit with cognitive scores** (+40% AUC)
   - Confirms clinical features matter
   - But MMSE/CDRSB are not usable for early detection

## Data Processing Pipeline

1. **baseline_selection.py** - Select one scan per subject (prefer Visit='sc')
2. **file_matcher.py** - Link Image IDs to NIfTI file paths
3. **feature_extraction.py** - Extract ResNet18 embeddings (512-dim)
4. **data_split.py** - Subject-wise stratified 80/20 split
5. **train_level1.py** / **train_level2.py** - Model training

## Important Notes

- **No data leakage**: Subject-wise splitting, one scan per subject
- **Same architectures as OASIS**: MRIOnly, LateFusion, AttentionFusion
- **Same hyperparameters**: No ADNI-specific tuning
- **Cross-dataset experiments**: Pending (separate phase)

---
Author: Research Pipeline
Last Updated: 2025-12-23
