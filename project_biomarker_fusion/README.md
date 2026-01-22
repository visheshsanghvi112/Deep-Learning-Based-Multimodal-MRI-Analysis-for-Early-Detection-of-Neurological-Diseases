# 🧬 Biomarker + MRI Longitudinal Fusion (NEW EXPERIMENT)

**Date Created:** January 22, 2026  
**Status:** 🚧 NEW - In Development

---

## ⚠️ IMPORTANT: This is a SEPARATE experiment

This directory contains a **brand new experiment** that does NOT modify any existing work in:
- `project/` (original OASIS cross-sectional)
- `project_adni/` (ADNI baseline experiments)  
- `project_longitudinal/` (ResNet longitudinal experiments)

**All previous results remain untouched and valid!**

---

## 🎯 Research Question

**Can we achieve >0.85 AUC by combining:**
1. 🧠 **MRI features** (ResNet18, 512-dim) - captures brain structure
2. 🧬 **Biomarker features** (Hippocampus, Ventricles, APOE4) - captures biology
3. 📈 **Longitudinal deltas** (change over time) - captures progression

### Hypothesis

| Component | Alone | Combined |
|-----------|-------|----------|
| ResNet longitudinal | 0.52 | ❌ Too weak |
| Biomarkers + delta | 0.83 | ✅ Strong but lacks imaging |
| **ResNet + Biomarkers + Delta** | **???** | 🎯 **Target: 0.85+** |

---

## 📊 What's Different from Existing Work?

### Existing Work (PRESERVED):
1. **project_adni/**: Cross-sectional ADNI (Level-1, Level-MAX)
   - Uses: Baseline scans only
   - Best: 0.808 AUC (Level-MAX with 14 biomarkers)

2. **project_longitudinal/**: ResNet longitudinal
   - Uses: ResNet features + temporal change
   - Best: 0.52 AUC (ResNet doesn't capture atrophy)

### This New Work:
3. **project_biomarker_fusion/** (THIS FOLDER):
   - Uses: **ResNet + Real biomarkers + Longitudinal**
   - Goal: Build actual trained PyTorch fusion model
   - Target: 0.85+ AUC

---

## 🔬 Methodology

### Input Features (Per Subject):

**Baseline Visit:**
- MRI ResNet features: 512-dim
- Hippocampus volume: 1
- Ventricles volume: 1
- Entorhinal volume: 1
- APOE4 status: 1
- Age: 1
- Sex: 1

**Follow-up Visit:**
- MRI ResNet features: 512-dim
- Hippocampus volume: 1
- Ventricles volume: 1
- Entorhinal volume: 1

**Delta (Change):**
- ResNet delta: 512-dim
- Hippocampus delta: 1
- Ventricles delta: 1
- Entorhinal delta: 1

**Total: 512*3 + 3*3 + 3 = 1,548 features**

### Architecture

```
Baseline MRI (512)  ──┐
Baseline Bio (6)    ──┼──> Early Fusion ──> MLP ──> 2 classes
                      │
Followup MRI (512)  ──┤
Followup Bio (3)    ──┤
                      │
Delta MRI (512)     ──┤
Delta Bio (3)       ──┘
```

### Training Protocol
- 5-fold cross-validation
- Subject-wise splits (no leakage)
- Same train/test subjects as project_longitudinal/
- Adam optimizer, learning rate scheduling
- Early stopping on validation AUC

---

## 📁 Directory Structure

```
project_biomarker_fusion/
├── README.md                          ← This file
├── data/
│   ├── biomarker_longitudinal.npz     ← Extracted features
│   └── train_test_splits.csv          ← Subject splits
├── src/
│   ├── 01_extract_biomarkers.py       ← Get data from ADNIMERGE
│   ├── 02_prepare_fusion_data.py      ← Combine ResNet + biomarkers
│   ├── 03_train_fusion.py             ← Train PyTorch model
│   └── 04_evaluate.py                 ← Full evaluation
├── results/
│   ├── checkpoints/                   ← Saved models
│   ├── metrics.json                   ← AUC, accuracy, etc.
│   └── comparison.json                ← vs existing baselines
└── docs/
    └── RESULTS.md                     ← Full writeup
```

---

## 🚀 How to Run

```bash
# 1. Extract biomarkers from ADNIMERGE
python src/01_extract_biomarkers.py

# 2. Prepare fusion dataset
python src/02_prepare_fusion_data.py

# 3. Train model
python src/03_train_fusion.py

# 4. Evaluate
python src/04_evaluate.py
```

---

## 📊 Expected Results

| Model | AUC | Source |
|-------|-----|--------|
| ResNet-only (longitudinal) | 0.52 | project_longitudinal/ |
| Biomarkers-only (statistical) | 0.83 | project_longitudinal/biomarker_analysis/ |
| **THIS: Full Fusion** | **0.85+?** | **NEW - To be determined** |

---

## ✅ Safety Checklist

- [x] New separate directory created
- [x] Does NOT modify project/ code
- [x] Does NOT modify project_adni/ code
- [x] Does NOT modify project_longitudinal/ code
- [x] Uses same train/test splits (reproducible)
- [x] All existing results remain valid

---

## 📝 Notes

This experiment will answer the key question: **Does deep learning fusion of MRI + biomarkers beat simple logistic regression?**

If YES (>0.85): We have a strong publication-ready model!  
If NO (~0.83): Simple models sufficient, but we validated the approach

**No existing work will be altered or invalidated.**
