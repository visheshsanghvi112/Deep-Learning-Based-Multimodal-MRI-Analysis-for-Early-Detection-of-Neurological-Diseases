# 🎯 REPOSITORY CLEANUP & FUTURE-PROOFING PLAN

**Created:** February 9, 2026  
**Objective:** Make repository completely self-contained and runnable without requiring 200GB of local data  
**Timeline:** Immediate (before data deletion)

---

## 📋 EXECUTIVE SUMMARY

### Current Situation
- ✅ **Paper is complete** and ready for presentation
- ⚠️ **~200GB of raw MRI data** stored locally (OASIS + ADNI)
- 🎯 **Need to delete** large data files but keep project functional
- 🔒 **Cannot re-download** 200GB whenever running the project
- 📚 **Product Documentation** (`docs/PROJECT_DOCUMENTATION.md`) is the DIAMOND TRUTH

### Goal
> **Create a self-contained repository where anyone can:**
> 1. Clone the repo
> 2. Install dependencies
> 3. Run demo/inference with minimal sample data
> 4. Reproduce key results without downloading 200GB
> 5. Access complete documentation and figures

---

## 🔍 CURRENT STATE ANALYSIS

### What's Being Ignored (`.gitignore`)
```
❌ /ADNI/                   → Raw ADNI data (huge)
❌ disc*/                   → Raw OASIS data (huge)  
❌ extracted_features/      → Processed features
❌ /data/                   → All root-level data
❌ *.csv, *.xlsx, *.npz     → Data files (with exceptions)
❌ *.nii, *.nii.gz          → NIfTI MRI scans
❌ *.pth, *.pt              → Model weights
```

### What's In the Repo (Already Tracked)
```
✅ docs/                    → All documentation (GOLD)
✅ figures/                 → All visualizations (PNG + PDF)
✅ project/frontend/        → Next.js app (deployed)
✅ project/backend/         → FastAPI backend
✅ scripts/                 → Utility scripts
✅ project_adni/src/        → ADNI pipeline code
✅ project_longitudinal_fusion/  → Final 0.848 AUC results
✅ README.md                → Main documentation
✅ requirements.txt         → Python dependencies
```

### What's Currently LOCAL ONLY ⚠️
```
⚠️ data/ADNI/              → ~100GB raw scans + ADNIMERGE.csv (12.65 MB)
⚠️ data/disc1-12/          → ~100GB raw OASIS scans
⚠️ data/extracted_features/→ Processed .npz files
⚠️ Model checkpoints       → Trained .pth weights
⚠️ Some result CSVs        → May not be tracked
```

---

## 🎯 IMPLEMENTATION PLAN

### PHASE 1: AUDIT & DOCUMENTATION ✅

#### Step 1.1: Verify Product Documentation is Complete
**Source of Truth:** `docs/PROJECT_DOCUMENTATION.md` (2,368 lines)

**Action Items:**
- [x] Confirm all experiments are documented
- [x] Verify all results have confidence intervals
- [x] Check all figures are referenced
- [ ] Cross-reference with README.md for consistency
- [ ] Ensure reproducibility section is complete

**Output:** Documentation audit report

---

#### Step 1.2: Identify Essential vs. Disposable Data
**Classify all data files:**

| Category | Keep in Repo? | Rationale |
|----------|--------------|-----------|
| **Raw OASIS scans** (disc1-12) | ❌ NO | 100GB, publicly downloadable |
| **Raw ADNI scans** (ADNI/) | ❌ NO | 100GB, requires ADNI access |
| **ADNIMERGE.csv** (12.65 MB) | ⚠️ MAYBE | Key metadata, but may have privacy concerns |
| **Extracted features** (.npz) | ✅ YES (sample only) | Small sample for demo/testing |
| **Result CSVs** (train/test splits) | ✅ YES | Essential for reproducibility |
| **Model weights** (.pth) | ✅ YES (best model only) | Needed for inference |
| **Figures** (PDF + PNG) | ✅ YES | Already tracked |
| **Documentation** (MD files) | ✅ YES | Already tracked |

**Action:**
```bash
# Survey what's not tracked but should be
git status --ignored
git ls-files --others --ignored --exclude-standard

# Find all result CSVs and model weights
fd -e csv -e pth -e pt --no-ignore-vcs
```

---

#### Step 1.3: Create Data Acquisition Guide
**File:** `docs/DATA_ACQUISITION_GUIDE.md`

**Contents:**
1. How to download OASIS-1 (direct links)
2. How to apply for ADNI access (step-by-step)
3. Expected folder structure after download
4. Preprocessing requirements
5. Feature extraction commands

**Purpose:** Anyone in future can reconstruct the data pipeline if needed

---

### PHASE 2: MINIMAL DATA PACKAGE 📦

#### Step 2.1: Create Sample Dataset
**Goal:** Provide enough data to run demo without 200GB

**What to Include:**
```
data/
├── README.md                          ← Data package documentation
├── sample/
│   ├── oasis_sample_10_scans/        ← 10 representative OASIS scans (~100MB)
│   ├── adni_sample_10_scans/         ← 10 representative ADNI scans (~100MB)
│   └── features/
│       ├── oasis_sample_features.npz ← Features for 10 scans
│       └── adni_sample_features.npz  ← Features for 10 scans
├── splits/                            ← Train/test split CSVs
│   ├── oasis_train_test_split.csv
│   ├── adni_train_test_split.csv
│   └── longitudinal_split.csv
└── metadata/
    ├── oasis_clinical_features.csv   ← De-identified clinical data
    └── adni_clinical_features.csv    ← De-identified clinical data (if allowed)
```

**Size Estimate:** ~200-300 MB total (acceptable for repo)

**Script to Create:**
```python
# scripts/create_minimal_data_package.py
"""
Extracts 10 representative samples from each dataset:
- Balanced CN/MCI/AD distribution
- Covers age range
- Includes both genders
- Copies raw scans + features
"""
```

---

#### Step 2.2: Update .gitignore Strategically
**Current Issue:** Blanket ignore of `/data/` prevents tracking anything

**Solution:**
```gitignore
# In .gitignore

# Ignore raw data (huge)
/data/disc*/
/data/ADNI/
/data/extracted_features/

# BUT allow sample data
!/data/sample/
!/data/splits/
!/data/metadata/
!/data/README.md

# Allow small data files in sample folder
!/data/sample/**/*.npz
!/data/sample/**/*.csv
!/data/sample/**/*.json
```

---

#### Step 2.3: Archive Best Model Weights
**Keep:** Only the best-performing models

**Structure:**
```
models/
├── README.md              ← Model card with performance metrics
├── oasis/
│   └── late_fusion_best_fold.pth  ← 0.796 AUC model
├── adni/
│   ├── level_max_late.pth         ← 0.808 AUC model
│   └── level_max_attention.pth    ← 0.808 AUC model
└── longitudinal/
    └── random_forest_0848.pkl     ← 0.848 AUC model
```

**Size:** ~50-100 MB per model (ResNet18 is ~45 MB)

**Action:**
```python
# Find best models
fd -e pth -e pkl -e pt | grep best | grep -v checkpoint
```

---

### PHASE 3: README OVERHAUL 📝

#### Step 3.1: Update README.md Based on Product Documentation
**Reference:** `docs/PROJECT_DOCUMENTATION.md` (lines 1-800 reviewed)

**Changes Needed:**

1. **Add "Quick Start Without Data" Section**
```markdown
## 🚀 Quick Start (No Big Data Required)

### Option 1: View Research Only
- Visit live demo: https://neuroscope-mri.vercel.app
- Read documentation: `docs/PROJECT_DOCUMENTATION.md`
- View figures: `figures/` directory

### Option 2: Run Inference Demo (Minimal Data)
```bash
# Uses 10 sample scans included in repo
python scripts/run_demo_inference.py
```

### Option 3: Full Reproduction (Requires Data Download)
See `docs/DATA_ACQUISITION_GUIDE.md` for:
- OASIS-1 download (~50GB)
- ADNI access application
- Preprocessing pipeline
```

2. **Update "How to Run" Section**
**Before:**
```markdown
pip install -r requirements.txt
python train.py
```

**After:**
```markdown
### Prerequisites
- Python 3.12+
- PyTorch 2.0+
- 8GB RAM minimum (for demo)
- 32GB RAM + GPU (for training on full data)

### Demo Mode (No Data Download)
```bash
pip install -r requirements.txt
python scripts/run_demo.py --mode inference --data sample
# Uses included sample data, runs in <5 minutes
```

### Training Mode (Requires Full Data)
```bash
# 1. Download data (see docs/DATA_ACQUISITION_GUIDE.md)
# 2. Extract features
python scripts/extract_features.py --dataset oasis --input data/disc* --output data/extracted_features/
# 3. Train models
cd project_adni
python src/train_level_max.py
```
```

3. **Add Data Availability Section**
```markdown
## 📊 Data Availability

### Included in This Repository
✅ Sample dataset (20 scans, ~200MB)
✅ Train/test split files (CSVs)
✅ Extracted features for all experiments
✅ Pre-trained model weights (best models)
✅ Complete documentation
✅ All figures (PNG + PDF)

### Not Included (Download Required for Full Reproduction)
❌ OASIS-1 raw scans (~50GB) - [Download here](https://www.oasis-brains.org/)
❌ ADNI-1 raw scans (~50GB) - [Apply for access](http://adni.loni.usc.edu/)

### Why Not Included?
- **Size:** 200GB total is impractical for version control
- **Privacy:** ADNI requires data use agreement
- **Availability:** Both datasets are publicly accessible (with application)
```

4. **Cross-Verify All Metrics with Product Documentation**

| Metric in README | Value in README | Value in PROJECT_DOCUMENTATION.md | Match? |
|------------------|-----------------|-----------------------------------|--------|
| OASIS Late Fusion AUC | 0.794 | 0.796±0.092 | ✅ Update README to 0.796 |
| ADNI Level-MAX AUC | 0.808 | 0.808 | ✅ Match |
| Longitudinal AUC | 0.848 | 0.848 (0.812-0.883) | ✅ Match |
| OASIS subjects | 205 | 205 | ✅ Match |
| ADNI subjects | 629 | 629 | ✅ Match |

**Action:** Go through entire README and verify every number against `PROJECT_DOCUMENTATION.md`

---

#### Step 3.2: Create Data Package README
**File:** `data/README.md`

**Structure:**
```markdown
# 📊 Data Package

## What's Included
This folder contains minimal sample data for demonstration purposes.

### Sample Scans (`sample/`)
- 10 OASIS scans (balanced CN/CDR 0.5)
- 10 ADNI scans (balanced CN/MCI/AD)
- Total size: ~200 MB

### Clinical Metadata (`metadata/`)
- De-identified clinical features
- Age, sex, education, volumes
- NO patient identifiers

### Train/Test Splits (`splits/`)
- Exact splits used in paper
- Ensures reproducible results
- Subject IDs are internal (no PHI)

## Getting Full Datasets
See `../docs/DATA_ACQUISITION_GUIDE.md`
```

---

### PHASE 4: SCRIPTS & AUTOMATION 🔧

#### Step 4.1: Create Demo Inference Script
**File:** `scripts/run_demo_inference.py`

**Purpose:** One-command demo that works with included sample data

```python
"""
Demo inference script using sample data included in repository.

Usage:
    python scripts/run_demo_inference.py

Expected output:
    - Loads 10 sample OASIS scans
    - Runs inference with best model
    - Shows predictions and confidence intervals
    - Completes in <5 minutes on CPU
"""

import torch
from pathlib import Path

def main():
    print("🧠 NeuroScope Demo - Early Dementia Detection")
    print("=" * 60)
    
    # Check sample data exists
    sample_dir = Path("data/sample")
    if not sample_dir.exists():
        print("❌ Sample data not found!")
        print("This should be included in the repository.")
        return
    
    # Load model
    model_path = Path("models/oasis/late_fusion_best.pth")
    if not model_path.exists():
        print("❌ Model weights not found!")
        return
    
    print("✅ Loading model...")
    # ... (model loading code)
    
    print("✅ Running inference on 10 sample scans...")
    # ... (inference code)
    
    print("\n📊 Results:")
    print("Sample 1: CDR 0 (Normal) → Predicted: 0.12 (Confidence: 88%)")
    print("Sample 2: CDR 0.5 (Impaired) → Predicted: 0.89 (Confidence: 91%)")
    # ...
    
    print("\n✅ Demo complete!")
    print("For full reproduction, see docs/DATA_ACQUISITION_GUIDE.md")

if __name__ == "__main__":
    main()
```

---

#### Step 4.2: Create Data Download Helper Script
**File:** `scripts/download_oasis.py`

```python
"""
Helper script to download OASIS-1 dataset.

This script automates the download from OASIS servers.
You will still need to register on their website first.
"""
import requests
from pathlib import Path

OASIS_URL = "https://www.oasis-brains.org/..."
# ... download logic with progress bar
```

---

#### Step 4.3: Update All Scripts to Handle Missing Data Gracefully
**Find all scripts that assume data exists:**
```bash
grep -r "data/disc" scripts/
grep -r "data/ADNI" scripts/
```

**Add checks:**
```python
# Before
df = pd.read_csv("data/ADNI/ADNIMERGE.csv")

# After
data_path = Path("data/ADNI/ADNIMERGE.csv")
if not data_path.exists():
    print(f"❌ Data not found: {data_path}")
    print("See docs/DATA_ACQUISITION_GUIDE.md for download instructions.")
    sys.exit(1)
df = pd.read_csv(data_path)
```

---

### PHASE 5: RESULT PRESERVATION 🏆

#### Step 5.1: Archive Key Result Files
**Ensure these are tracked in Git:**

```bash
# Find all result JSONs and CSVs
fd -e json -e csv -e txt . project_adni/results/
fd -e json -e csv -e txt . project_longitudinal_fusion/results/

# Check if tracked
git ls-files | grep results
```

**Structure:**
```
project_adni/results/
├── level_max/
│   ├── test_results.json          ← 0.808 AUC results
│   ├── confusion_matrix.csv
│   └── roc_curve_data.csv
└── level_1/
    └── test_results.json           ← 0.598 AUC results

project_longitudinal_fusion/results/
├── random_forest_full_cohort.json  ← 0.848 AUC (CI: 0.812-0.883)
├── feature_importance.csv
└── progression_predictions.csv
```

**Update `.gitignore`:**
```gitignore
# Instead of blanket ignore:
# project_adni/results/

# Be selective:
project_adni/results/checkpoints/
project_adni/results/logs/
!project_adni/results/**/*.json
!project_adni/results/**/*.csv
!project_adni/results/**/*.md
```

---

### PHASE 6: FRONTEND & DOCUMENTATION SYNC 🌐

#### Step 6.1: Ensure Frontend Has All Necessary Files
**Check:**
```bash
cd project/frontend
ls -lh public/figures/
ls -lh public/docs/
```

**Required:**
- All figures referenced in pages
- Documentation markdown files (if served statically)
- Sample data for demo features

---

#### Step 6.2: Verify All Documentation Links
**Script:** `scripts/verify_doc_links.py`

```python
"""
Verify all documentation cross-references are valid.
"""
import re
from pathlib import Path

def check_markdown_links(md_file):
    content = md_file.read_text()
    links = re.findall(r'\[.*?\]\((.*?)\)', content)
    
    for link in links:
        if link.startswith('http'):
            continue  # Skip external links
        
        target = (md_file.parent / link).resolve()
        if not target.exists():
            print(f"❌ Broken link in {md_file}: {link}")

docs_dir = Path("docs")
for md_file in docs_dir.glob("*.md"):
    check_markdown_links(md_file)
```

---

### PHASE 7: TESTING & VALIDATION ✅

#### Step 7.1: Test Clone-to-Run Workflow
**Simulate fresh user experience:**

```bash
# 1. Clone repo (simulate)
cd /tmp
git clone <repo_url> test_clone
cd test_clone

# 2. Install dependencies
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Run demo
python scripts/run_demo_inference.py

# 4. Expected: Demo runs successfully without errors

# 5. Check docs are accessible
ls docs/
cat docs/PROJECT_DOCUMENTATION.md | head

# 6. Verify figures
ls figures/ | wc -l  # Should show 29+ figures
```

---

#### Step 7.2: Verify Reproducibility Claims
**Check all commands in README work:**

```bash
# For each command in README
cd project/frontend
npm install
npm run dev  # ← Should work

cd ../../project_adni
# Commands should gracefully handle missing data with helpful messages
```

---

### PHASE 8: FINAL CLEANUP 🧹

#### Step 8.1: Commit Everything Essential
```bash
# Check what's not tracked
git status

# Stage important files
git add data/sample/
git add data/splits/
git add data/metadata/
git add models/
git add scripts/run_demo_inference.py
git add docs/DATA_ACQUISITION_GUIDE.md
git add data/README.md

# Commit
git commit -m "Add minimal data package for reproducibility

- Sample dataset (20 scans, ~200MB)
- Train/test split CSVs
- Best model weights
- Demo inference script
- Data acquisition guide
"

# Push
git push origin main
```

---

#### Step 8.2: Create Release/Archive
**Tag the current state:**
```bash
git tag -a v1.0.0-paper-submission -m "Final version for paper submission

All experiments complete:
- OASIS: 0.796 AUC
- ADNI Level-MAX: 0.808 AUC  
- Longitudinal: 0.848 AUC
- Complete documentation
- Minimal reproducible data package
"

git push origin v1.0.0-paper-submission
```

---

#### Step 8.3: Safe to Delete Local Data
**Only after verifying:**
- [ ] All essential data is in repo or reproducible
- [ ] README updated with download instructions
- [ ] Demo works without big data
- [ ] All results are preserved
- [ ] Documentation is complete
- [ ] Repository is pushed to GitHub
- [ ] Release tag is created

**Then you can safely:**
```bash
# Backup first (just in case)
tar -czf ~/Desktop/discs_backup_$(date +%Y%m%d).tar.gz data/

# Delete large data
rm -rf data/disc*
rm -rf data/ADNI/
rm -rf data/extracted_features/

# Keep sample data (already staged in git)
# Keep splits (already staged in git)
```

---

## 📊 SIZE COMPARISON

### Before Cleanup
```
Total: ~200 GB
├── data/disc1-12/      → ~100 GB (OASIS raw)
├── data/ADNI/          → ~100 GB (ADNI raw + metadata)
├── extracted_features/ → ~500 MB (processed .npz)
├── models/checkpoints/ → ~2 GB (all training checkpoints)
└── Code + Docs         → ~100 MB
```

### After Cleanup (In Repo)
```
Total: ~1-2 GB (Git LFS optional for models)
├── data/sample/        → ~200-300 MB (sample scans + features)
├── data/splits/        → ~1 MB (CSVs)
├── data/metadata/      → ~5 MB (clinical data)
├── models/             → ~200 MB (best models only)
├── figures/            → ~50 MB (PNG + PDF)
├── docs/               → ~5 MB (markdowns)
├── Frontend            → ~500 MB (with node_modules → but .gitignored)
└── Code                → ~50 MB (Python scripts)
```

---

## 🎯 SUCCESS CRITERIA

### Must Have ✅
- [ ] **README matches** `PROJECT_DOCUMENTATION.md` (all metrics verified)
- [ ] **Demo runs** without downloading 200GB
- [ ] **All figures** are in repo and referenced
- [ ] **All results** (0.848, 0.808, 0.796) are preserved with confidence intervals
- [ ] **Documentation** is complete and cross-referenced
- [ ] **Data acquisition guide** exists for full reproduction
- [ ] **Best model weights** are tracked
- [ ] **.gitignore** allows sample data but blocks big data
- [ ] **Git tagged** with paper submission version
- [ ] **Can safely delete** local 200GB data

### Nice to Have 🌟
- [ ] Git LFS setup for model weights (if >100 MB)
- [ ] Automated tests for demo script
- [ ] Docker container for full environment
- [ ] Zenodo archive for long-term preservation
- [ ] Scripts pass linting/type checking

---

## 📝 CHECKLIST FOR EXECUTION

### Documentation Phase
- [ ] Audit `PROJECT_DOCUMENTATION.md` completeness
- [ ] Cross-verify all metrics in README
- [ ] Create `docs/DATA_ACQUISITION_GUIDE.md`
- [ ] Create `data/README.md`
- [ ] Update README.md "Quick Start" section
- [ ] Update README.md "Data Availability" section

### Data Phase
- [ ] Create sample dataset (10 OASIS + 10 ADNI scans)
- [ ] Extract and preserve train/test split CSVs
- [ ] Save de-identified clinical metadata
- [ ] Archive best model weights only
- [ ] Update `.gitignore` to allow sample data

### Code Phase
- [ ] Create `scripts/run_demo_inference.py`
- [ ] Create `scripts/download_oasis.py`
- [ ] Update all scripts to gracefully handle missing data
- [ ] Add data existence checks everywhere

### Testing Phase
- [ ] Test fresh clone workflow
- [ ] Run demo inference successfully
- [ ] Verify all README commands work
- [ ] Check all documentation links

### Finalization Phase
- [ ] Stage and commit all essential files
- [ ] Create Git tag for paper version
- [ ] Push to GitHub
- [ ] Verify on GitHub that everything is visible
- [ ] **ONLY THEN** delete local 200GB data

---

## ⚠️ RISKS & MITIGATION

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Delete data before committing** | 🔴 HIGH | Checklist above, create backup first |
| **Forget to track result files** | 🟡 MEDIUM | Run `fd -e json -e csv` before deleting |
| **README out of sync with docs** | 🟡 MEDIUM | Script to cross-verify metrics |
| **Sample data has privacy issues** | 🔴 HIGH | Use synthetic/heavily anonymized data only |
| **Model weights too large for Git** | 🟢 LOW | Use Git LFS or external hosting |
| **Demo script broken** | 🟡 MEDIUM | Test in fresh environment before committing |

---

## 🚀 EXECUTION ORDER

**DO THESE IN ORDER:**

1. ✅ **READ THIS PLAN FULLY**
2. **Phase 1:** Documentation audit (verify SOURCE OF TRUTH)
3. **Phase 3:** Update README.md based on Product Documentation
4. **Phase 2:** Create minimal data package
5. **Phase 4:** Create demo scripts
6. **Phase 5:** Archive results
7. **Phase 7:** TEST EVERYTHING in fresh environment
8. **Phase 8:** Commit, tag, push
9. **VERIFY** on GitHub
10. **BACKUP** local data (tar.gz to external drive)
11. **DELETE** local 200GB data

---

## 📞 QUESTIONS TO RESOLVE

- [ ] Is ADNIMERGE.csv (12.65 MB) allowed to be shared? (Privacy/DUA concerns)
  - If YES: Include de-identified version
  - If NO: Provide script to reconstruct from ADNI download

- [ ] Should we use Git LFS for model weights?
  - If models >100 MB: YES
  - If models <100 MB: Direct commit is fine

- [ ] Do we want a Docker image for reproducibility?
  - Adds complexity but ensures environment consistency

- [ ] Should sample scans be real or synthetic?
  - Real (anonymized): More authentic but privacy risk
  - Synthetic/augmented: Safer but less realistic

---

**NEXT STEP:** Start with Phase 1 - Documentation Audit, then proceed systematically through the phases.
