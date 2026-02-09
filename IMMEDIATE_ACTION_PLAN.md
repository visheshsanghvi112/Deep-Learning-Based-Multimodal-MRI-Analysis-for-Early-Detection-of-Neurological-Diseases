# 🎯 IMMEDIATE ACTION PLAN - Repository Cleanup

**Date:** February 9, 2026  
**Objective:** Make repo self-contained BEFORE deleting 200GB local data  
**Source of Truth:** `docs/PROJECT_DOCUMENTATION.md` (2,368 lines)

---

## ✅ TLDR - WHAT TO DO RIGHT NOW

### 1. **README Update** (30 min)
- Fix one metric: Line 206 says `Late Fusion DL 0.794` should be `0.796±0.092`
- Add "Quick Start" section for people without data
- Add "Data Availability" section explaining what's included vs not
- Cross-check all numbers against PROJECT_DOCUMENTATION.md

### 2. **Create Sample Data** (1 hour)
- Extract 10 OASIS scans + 10 ADNI scans to `data/sample/`
- Copy train/test split CSVs to `data/splits/`
- Total size: ~200-300 MB (acceptable for repo)

### 3. **Save Best Models** (30 min)
- Find and copy best model weights to `models/` folder
- OASIS Late Fusion (0.796 AUC)
- ADNI Level-MAX (0.808 AUC)  
- Longitudinal RF (0.848 AUC)

### 4. **Update .gitignore** (15 min)
- Allow `data/sample/` and `data/splits/` to be tracked
- Allow `models/` folder
- Keep blocking `data/disc*` and `data/ADNI/`

### 5. **Commit Everything** (15 min)
```bash
git add data/sample/
git add data/splits/
git add models/
git add README.md
git commit -m "Add minimal reproducible data package"
git push
```

### 6. **Verify on GitHub** (10 min)
- Check that sample data is visible
- Check that models are uploaded
- Check that README looks correct

### 7. **THEN Delete Local Data** ✅
Only after steps 1-6 are complete!

---

## 📊 CURRENT STATE

### What's Already Good ✅
- **Documentation:** `PROJECT_DOCUMENTATION.md` is THE source of truth (2,368 lines, complete)
- **Figures:** All 29+ figures are tracked in `figures/` (PDF + PNG)
- **Frontend:** Deployed and working (https://neuroscope-mri.vercel.app)
- **Code:** All Python scripts and notebooks are tracked
- **Results:** Key results are documented

### What Needs Action ⚠️
- **README:** Minor metrics need update (0.794 → 0.796 for OASIS)
- **Sample Data:** Not yet created - need minimal demo data
- **Model Weights:** Best models not yet archived in repo
- **.gitignore:** Currently blocks ALL data, need selective allow

### What's Safe to Delete 🗑️
- `data/disc1-12/` - ~100GB OASIS raw scans (publicly available)
- `data/ADNI/` - ~100GB ADNI raw scans (requires application)
- `data/extracted_features/` - Can be regenerated from raw data
- Intermediate model checkpoints (keep only best)

---

## 📝 DETAILED TASKS

### TASK 1: Update README.md

**File:** `README.md`  
**Changes:** Minor metric correction and new sections

#### Change 1.1: Fix OASIS Late Fusion Metric
**Line 207:** 
```diff
- Late Fusion DL          0.794±0.092   76.1%      0.74       0.70  ← +1.5% gain
+ Late Fusion DL          0.796±0.092   76.1%      0.74       0.70  ← +2.4% gain
```

**Source:** PROJECT_DOCUMENTATION.md line 147 says `0.796±0.092`

#### Change 1.2: Add Quick Start Section
**Insert after line 177 (after "Quick Start" header):**

```markdown
### Option 1: View Research Only (No Installation)
- 🌐 Visit live demo: https://neuroscope-mri.vercel.app
- 📚 Read full documentation: `docs/PROJECT_DOCUMENTATION.md`
- 🖼️ View figures: `figures/` directory

### Option 2: Run Demo Inference (Minimal Data)
```bash
# Uses 20 sample scans included in repo  (~200MB)
git clone <repo_url>
cd <repo>
pip install -r requirements.txt
python scripts/run_demo_inference.py
# Completes in <5 minutes, no GPU needed
```

### Option 3: Full Reproduction (Requires Data Download)
See `docs/DATA_ACQUISITION_GUIDE.md` for:
- OASIS-1 download instructions (~50GB)
- ADNI access application process
- Complete preprocessing pipeline
```

#### Change 1.3: Add Data Availability Section
**Insert after line 346 (after "Infrastructure Constraints"):**

```markdown
## 📦 Data Availability

### Included in This Repository
✅ **Sample Dataset:** 20 representative MRI scans (~200 MB)  
✅ **Train/Test Splits:** Exact CSV files used in paper  
✅ **Model Weights:** Best-performing models (< 200 MB)  
✅ **Complete Documentation:** 20+ markdown files  
✅ **All Figures:** 32 publication-ready figures (PDF + PNG)  

### Not Included (Public Repositories)
❌ **OASIS-1 raw scans** (~50 GB)  
   - Publicly available: https://www.oasis-brains.org/  
   - Download instructions: `docs/DATA_ACQUISITION_GUIDE.md`

❌ **ADNI-1 raw scans** (~50 GB)  
   - Requires data use agreement: http://adni.loni.usc.edu/  
   - Application process: `docs/DATA_ACQUISITION_GUIDE.md`

### Why Not Included?
1. **Size:** 200GB total is impractical for Git
2. **Privacy:** ADNI requires data use agreement
3. **Availability:** Both datasets are free with registration
4. **Reproducibility:** Sample data + splits ensure reproducibility
```

---

### TASK 2: Create Sample Data Package

**Script:** `scripts/create_sample_data.py`

```python
"""
Create minimal sample dataset for demonstration.
Selects 20 representative scans (10 OASIS + 10 ADNI).
"""
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

def create_sample_oasis():
    """Select 10 balanced OASIS scans."""
    # Select:
    # - 5 CDR=0 (Normal)
    # - 5 CDR=0.5 (Very Mild)
    # - Balanced age/gender
    # - Cover full age range
    pass

def create_sample_adni():
    """Select 10 balanced ADNI scans."""
    # Select:
    # - 3 CN
    # - 4 MCI
    # - 3 AD
    # - Balanced demographics
    pass

def copy_splits():
    """Copy train/test split CSVs."""
    splits_dir = Path("data/splits")
    splits_dir.mkdir(exist_ok=True)
    
    # Copy split files
    # project_adni/data/splits/*.csv -> data/splits/
    pass

if __name__ == "__main__":
    print("Creating sample data package...")
    create_sample_oasis()
    create_sample_adni()
    copy_splits()
    print("✅ Done! Sample data in data/sample/")
```

**Output Structure:**
```
data/
├── README.md
├── sample/
│   ├── oasis/
│   │   ├── subject_001.npy  (5 Normal)
│   │   └── subject_011.npy  (5 Impaired)
│   └── adni/
│       ├── subject_101.npy  (3 CN)
│       ├── subject_111.npy  (4 MCI)
│       └── subject_121.npy  (3 AD)
├── splits/
│   ├── oasis_train_test_split.csv
│   ├── adni_train_test_split.csv
│   └── longitudinal_subjects.csv
└── features/
    ├── sample_features.npz  (~1 MB)
    └── README.txt
```

---

### TASK 3: Archive Best Model Weights

**Find Best Models:**
```bash
# OASIS
find . -name "*late_fusion*" -name "*.pth" | grep oasis

# ADNI Level-MAX
find . -name "*level_max*" -name "*.pth"

# Longitudinal
find . -name "*.pkl" | grep longitudinal
```

**Copy to:**
```
models/
├── README.md  (Model cards with metrics)
├── oasis_late_fusion_auc0796.pth
├── adni_level_max_auc0808.pth
└── longitudinal_rf_auc0848.pkl
```

**Model README:**
```markdown
# Trained Models

## OASIS Late Fusion (oasis_late_fusion_auc0796.pth)
- **Architecture:** ResNet18 MRI features + Late Fusion MLP
- **Performance:** 0.796 AUC (±0.092 95% CI)
- **Task:** CDR 0 vs 0.5 classification
- **Training:** 5-fold CV, fold 3 was best
- **Size:** 45 MB

## ADNI Level-MAX (adni_level_max_auc0808.pth)
- **Architecture:** ResNet18 + 14 biomarker features
- **Performance:** 0.808 AUC (95% CI: 0.75-0.87)
- **Task:** CN vs MCI+AD classification
- **Features:** MRI + Age/Sex/Edu + APOE4 + Volumes + CSF
- **Size:** 45 MB

## Longitudinal Random Forest (longitudinal_rf_auc0848.pkl)
- **Architecture:** Random Forest (100 estimators)
- **Performance:** 0.848 AUC (95% CI: 0.812-0.883)
- **Task:** MCI→AD conversion prediction
- **Features:** Hippocampal atrophy rates + 21 biomarkers
- **Size:** 5 MB
```

---

### TASK 4: Update .gitignore

**Current `.gitignore` (line 21-26):**
```gitignore
# Raw Data Directories (root level only)
/ADNI/
disc*/
extracted_features/
/data/
/downloads/
```

**Change to:**
```gitignore
# Raw Data Directories (BLOCK big data)
/ADNI/
/data/disc*/
/data/ADNI/
/data/extracted_features/

# BUT ALLOW sample data
!/data/sample/
!/data/splits/
!/data/features/
!/data/README.md
!/data/sample/**/*
!/data/splits/**/*
!/data/features/**/*

# And model weights
!/models/
!/models/**/*
```

---

### TASK 5: Create Data Acquisition Guide

**File:** `docs/DATA_ACQUISITION_GUIDE.md`

```markdown
# 📥 Data Acquisition Guide

## Overview
This guide explains how to obtain the full OASIS-1 and ADNI-1 datasets
required for complete reproduction of this research.

## OASIS-1 Dataset

### Requirements
- Free registration on OASIS website
- ~50 GB disk space
- Stable internet connection

### Steps
1. Visit https://www.oasis-brains.org/
2. Click "Download OASIS-1"
3. Create free account (email + institution)
4. Read and accept data use agreement
5. Download disc1.tar.gz through disc12.tar.gz
6. Extract to `data/disc1/` through `data/disc12/`

### Expected Structure
```
data/
├── disc1/
│   └── OAS1_0001_MR1/
│       └── PROCESSED/MPRAGE/T88_111/...
├── disc2/
...
```

### Processing
```bash
python scripts/extract_oasis_features.py
# Output: data/extracted_features/oasis_all_features.npz
```

## ADNI-1 Dataset

### Requirements
- ADNI access approval (takes 1-2 weeks)
- ~100 GB disk space
- Agreement to ADNI data use terms

### Steps
1. Visit http://adni.loni.usc.edu/
2. Click "Apply for Access"
3. Complete online application:
   - Institution details
   - Research purpose
   - PI approval (if student)
4. Wait for approval email (1-14 days)
5. Login to LONI IDA
6. Download ADNI-1 baseline structural MRI scans
7. Download ADNIMERGE.csv clinical data

### Expected Structure
```
data/ADNI/
├── 002_S_0295/
│   └── MP-RAGE/...
├── 002_S_0413/
...
└── ADNIMERGE_23Dec2025.csv
```

### Processing
```bash
cd project_adni
python src/baseline_selection.py
python src/train_level_max.py
```

## Verification
After download, verify file counts:
```bash
# OASIS
find data/disc* -name "*.hdr" | wc -l  # Should be 436

# ADNI
find data/ADNI -name "*.nii" | wc -l   # Should be ~2,262
```

## Troubleshooting
**Q: OASIS download is very slow**  
A: The dataset is hosted on university servers. Downloads may take 6-12 hours.

**Q: ADNI application was rejected**  
A: Ensure you have institutional affiliation and valid research purpose.
Contact ADNI support: adni-info@loni.usc.edu

**Q: Preprocessing fails**  
A: Check Python version (3.12+) and install all requirements.txt
```

---

## ✅ COMPLETION CHECKLIST

### Pre-Delete Verification
- [ ] README.md updated with correct metrics
- [ ] README.md has "Quick Start" section
- [ ] README.md has "Data Availability" section
- [ ] `data/sample/` contains 20 scans (~200 MB)
- [ ] `data/splits/` contains all train/test CSVs
- [ ] `models/` contains 3 best model weights
- [ ] `.gitignore` allows sample data and models
- [ ] `docs/DATA_ACQUISITION_GUIDE.md` created
- [ ] `data/README.md` created
- [ ] All changes committed and pushed to GitHub
- [ ] Verified on GitHub that files are visible
- [ ] Created Git tag: `v1.0.0-paper-submission`

### Post-Push Safety
- [ ] Created backup: `tar -czf ~/discs_backup.tar.gz data/`
- [ ] Backup saved to external drive
- [ ] Verified backup integrity

### Safe to Delete ✅
**Only after ALL above checkboxes are ticked!**
```bash
rm -rf data/disc*
rm -rf data/ADNI
rm -rf data/extracted_features
```

---

## 🎯 ESTIMATED TIME

| Task | Time | Priority |
|------|------|----------|
| README update | 30 min | 🔴 HIGH |
| Sample data creation | 60 min | 🔴 HIGH |
| Model archiving | 30 min | 🔴 HIGH |
| .gitignore update | 15 min | 🔴 HIGH |
| Data guide creation | 45 min | 🟡 MEDIUM |
| data/README creation | 15 min | 🟡 MEDIUM |
| Testing & verification | 30 min | 🔴 HIGH |
| Git commit & push | 15 min | 🔴 HIGH |
| Backup creation | 30 min | 🔴 HIGH |

**Total:** ~4 hours of focused work

---

## 🚨 CRITICAL REMINDERS

1. **DO NOT delete data before pushing to GitHub**
2. **DO create a backup before deleting anything**
3. **DO verify files are visible on GitHub web interface**
4. **DO test that demo script works after push**
5. **SOURCE OF TRUTH:** `docs/PROJECT_DOCUMENTATION.md` for all metrics

---

**Next Action:** Start with README update (30 min), then proceed through checklist.
