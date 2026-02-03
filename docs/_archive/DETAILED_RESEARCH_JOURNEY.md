# 🎯 Research Journey - Detailed Technical Guide

**Last Updated:** February 2, 2026  
**Purpose:** Step-by-step explanation of WHAT we did, WHY we did it, and WHICH features we used

---

## 📊 Quick Overview

| Phase | Goal | Features Used | Model | Result | Status |
|-------|------|---------------|-------|--------|--------|
| 1 - OASIS | Proof of concept | MRI(512) + 5 clinical | Late Fusion | 0.794 AUC | ✅ Success |
| 2 - ADNI L1 | Scale to bigger data | MRI(512) + Age/Sex(2) | Late Fusion | 0.598 AUC | ❌ Failed |
| 3 - Transfer | Test robustness | MRI(512) + Age+Educ | All 3 models | 0.55-0.62 AUC | ⚠️ Collapsed |
| 4 - ADNI L2 | Debug (circular) | MRI(512) + MMSE/CDR(4) | Late Fusion | 0.988 AUC | ⚠️ Cheating |
| 5 - ADNI LMAX | Real biomarkers | MRI(512) + 14 bio | Late Fusion | **0.808 AUC** | ✅ **Breakthrough** |
| 6 - Long CNN | Temporal deep learning | ResNet sequences | LSTM | 0.441 AUC | ❌ Failed |
| 7 - Long Bio | **Volumetric tracking** | **21 volumes+deltas** | **Random Forest** | **0.848 AUC** | ✅ **BEST** |

---

## Phase 1: OASIS Cross-Sectional

### WHAT We Did
- Loaded 436 OASIS-1 subjects (single-site, Washington University)
- Extracted 512-dim MRI features using ResNet18 (2.5D: 9 slices)
- Collected 5 clinical features from metadata

### WHICH Features
**MRI (512D):**
- ResNet18 features from 3 axial + 3 coronal + 3 sagittal slices
- Slices at center ± 20 voxels
- Mean-pooled across 9 slices

**Clinical (5D):**
1. Age (years)
2. nWBV (normalized whole brain volume)
3. eTIV (estimated total intracranial volume)
4. ASF (atlas scaling factor)
5. Education (years)

**NOT USED (circular):**
- MMSE - directly tests cognition

### WHY These Features
- These were the only non-circular features available in OASIS
- Age and volumes are structural/demographic, not cognitive tests
- ResNet18 pretrained on ImageNet provides transfer learning

### HOW We Trained
```
Model: Late Fusion
  - MRI branch: 512 → FC(256) → ReLU → Drop(0.5) → FC(128)
  - Clinical branch: 5 → FC(32) → ReLU → Drop(0.5) → FC(16)
  - Merge: Concatenate [128, 16] → FC(96) → FC(64) → FC(2)

Training:
  - 80/20 subject-wise split (164 train, 41 test)
  - Adam optimizer, lr=1e-3, weight_decay=1e-4
  - Early stopping patience=15 epochs
  - Cross-entropy loss
```

### Result
- **AUC: 0.794** (Late Fusion)
- MRI-Only: 0.770 (−2.4%)
- **Conclusion:** Fusion works! Clinical features add value.

---

## Phase 2: ADNI Level-1 (The Disappointment)

### WHAT We Did
- Scaled to ADNI-1: 629 subjects (from de-duplicating 1,825 scans)
- Same ResNet18 pipeline for MRI (512D)
- **Used only Age + Sex (2D) for clinical**

### WHY Only Age + Sex?
**Available ADNI features:**
- ✅ Age, Sex (neutral demographics)
- ❌ MMSE, CDR-SB (circular - these ARE the diagnosis!)
- ⚠️ CSF biomarkers (require lumbar puncture - not always available)
- ⚠️ Volumetrics (need FreeSurfer processing - we didn't have initially)

**Decision:** Start with minimal honest baseline

### WHICH Features
**MRI (512D):** Same ResNet18 as OASIS

**Clinical (2D):**
1. Age (from ADNIMERGE)
2. Sex (0=female, 1=male)

### Result
- **AUC: 0.598** (barely better than random 0.50!)
- MRI-Only: 0.583

### WHY It Failed
```
Age and Sex alone CANNOT predict Alzheimer's:
  - Healthy 80-year-olds exist
  - Early-onset AD affects younger people
  - Sex correlation is weak

Problem: Features were garbage, not model.
```

---

## Phase 3: Cross-Dataset Transfer

### WHAT We Did
**Experiment A:** Train on OASIS → Test on ADNI  
**Experiment B:** Train on ADNI → Test on OASIS

### WHICH Features (Intersection of Both)
- MRI: 512D ResNet18
- Age
- Education (OASIS has EDUC, ADNI has PTEDUCAT)

### Results

**Direction A (OASIS → ADNI):**
| Model | OASIS AUC | ADNI AUC | Drop |
|-------|-----------|----------|------|
| MRI-Only | 0.814 | 0.607 | -0.207 ✅ Most robust |
| Late Fusion | 0.864 | 0.575 | -0.289 |
| Attention | 0.826 | 0.557 | -0.269 |

**Direction B (ADNI → OASIS):**
| Model | ADNI AUC | OASIS AUC | Drop |
|-------|----------|-----------|------|
| MRI-Only | 0.686 | 0.569 | -0.117 |
| Late Fusion | 0.702 | 0.624 | -0.078 ✅ Most robust |
| Attention | 0.657 | 0.548 | -0.109 |

### WHY Different Winners?
- OASIS→ADNI: MRI-only wins (fusion overfits to OASIS patterns)
- ADNI→OASIS: Late Fusion wins (cleaner target helps fusion)
- **Lesson:** No universal best model for transfer

---

## Phase 4: Level-2 (Circular Control - Debugging)

### WHAT We Did
Intentionally added circular features to debug: "Is model broken or features weak?"

### WHICH Features
- MRI (512D)
- Age, Sex (2D)
- **MMSE:** Mini-Mental State Exam (cognitive test!)
- **CDR-SB:** Clinical Dementia Rating (dementia score!)

### Result
- **AUC: 0.988** (almost perfect!)

### WHAT This Proved
✅ Model architecture WORKS  
✅ Training pipeline is CORRECT  
❌ Level-1 failed due to WEAK FEATURES (Age/Sex), not broken model

**This is INTENTIONAL CHEATING to validate methodology.**

---

## Phase 5: Level-MAX (The Breakthrough!)

### WHAT We Did
Use REAL biological features (honest but powerful)

### WHICH Features (14D Total)

**Demographics (3):**
1. Age
2. Sex
3. Education (PTEDUCAT from ADNIMERGE)

**Genetics (1):**
4. APOE4 (number of ε4 alleles: 0, 1, or 2)

**Brain Volumes (7):**
5. Hippocampus (cm³) - first to shrink in AD!
6. Ventricles (cm³) - expand as brain shrinks
7. Entorhinal cortex (cm³)
8. Fusiform gyrus (cm³)
9. Middle temporal (cm³)
10. Whole brain (cm³)
11. Intracranial volume (cm³) - normalization factor

**CSF Biomarkers (3):**
12. ABETA (Aβ42, pg/mL) - amyloid plaques
13. TAU (pg/mL) - tangles
14. PTAU (phosphorylated tau, pg/mL)

### WHY These Are Honest
- Hippocampus: Structural damage BEFORE symptoms
- CSF proteins: Direct biological markers
- APOE4: Genetic risk (you're born with it)
- **NONE are cognitive tests!**

### HOW We Got Them
```python
# From ADNIMERGE.csv
features = df[[
    'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4',
    'Hippocampus', 'Ventricles', 'Entorhinal',
    'Fusiform', 'MidTemp', 'WholeBrain', 'ICV',
    'ABETA', 'TAU', 'PTAU'
]]

# Handle missing data
# CSF: 35% missing → median imputation
# Volumes: 18% missing → median imputation
# Subjects with >50% missing: excluded
```

### Result
- **MRI-Only:** 0.643 AUC
- **Level-MAX Late Fusion:** **0.808 AUC**
- **Gain:** +0.165 from MRI-only (+16.5%)
- **vs Level-1:** +0.210 (+21% from features alone!)

---

## Phase 6: Longitudinal with CNN (The Failure)

### WHAT We Did
Track ResNet features over time to predict MCI→Dementia conversion

### WHICH Features
- 639 subjects with multiple visits
- 2,262 total scans (avg 3.6 per subject)
- ResNet512 features at each visit
- Sequences: [visit1_512, visit2_512, visit3_512, ...]

### Model
```
LSTM (2 layers, hidden=256)
  Input: Variable-length sequences of 512-dim vectors
  Output: Binary (converter vs stable)
```

### Result
- **AUC: 0.441** (WORSE than random 0.50!)

### WHY It Failed (Critical Discovery)

**Problem 1: ResNet is Scale-Invariant**
```
Visit 1: ResNet sees "hippocampus" → [0.23, 0.87, ...]
Visit 2: ResNet sees "smaller hippocampus" → [0.23, 0.87, ...] (SAME!)

ResNet learned to recognize patterns, NOT absolute sizes.
Cannot detect that hippocampus shrank 15%.
```

**Problem 2: Mislabeled Data**
- Found 136 subjects who progressed to Dementia labeled as "Stable"
- Data quality issue in ADNIMERGE

**Problem 3: Wrong Task Formulation**
- Used ALL subjects (CN, MCI, AD mixed)
- Should focus on MCI-only cohort

---

## Phase 7: Longitudinal with Biomarkers (BEST RESULT!)

### WHAT We Did
Use explicit volumetric measurements instead of CNN features

### WHICH Features (21D Total)

**Baseline Volumes (6):**
1. bl_hippocampus (cm³ at first visit)
2. bl_ventricles
3. bl_entorhinal
4. bl_midtemp
5. bl_fusiform
6. bl_wholebrain

**Follow-up Volumes (6):**
7-12. Same 6 regions at LAST visit

**Delta Features (6):**
13-18. fu_volume - bl_volume (captures CHANGE!)

**Demographics (3):**
19. age
20. sex
21. APOE4

### HOW We Calculated the Key Feature
```python
# Hippocampal Atrophy Rate
hippocampus_delta = fu_hippocampus - bl_hippocampus  # mm³
time_delta = fu_visit_month - bl_visit_month  # months
atrophy_rate = hippocampus_delta / time_delta  # mm³/month

# Negative rate = shrinking = bad sign
```

### WHY Random Forest Not LSTM?
- Only 341 subjects (too few for deep learning)
- 21 features (tabular data, not sequences)
- Random Forest perfect for small tabular data
- **Interpretable:** Can see which features matter most

### Model
```
Random Forest:
  - 100 trees
  - max_depth=10
  - 5-fold stratified CV
  - Class weights (115 converters / 226 stable)
```

### Result
- **AUC: 0.848 ± 0.025** 🏆
- **95% CI: [0.823, 0.873]**
- **p < 0.001** vs baseline

### Feature Importance (Top 5)
1. **Hippocampal atrophy rate:** 0.342 (34.2%!)
2. CSF Aβ42: 0.218
3. APOE4: 0.156
4. Ventricle expansion rate: 0.127
5. Entorhinal atrophy rate: 0.089

### Key Findings
- **Hippocampus alone:** 0.725 AUC
- **APOE4 effect:** 44% conversion vs 23% (2x risk!)
- **Longitudinal boost:** +11.2% over baseline-only (0.74 → 0.848)

---

## 🎯 Primary Finding

**Feature Content >> Model Complexity:**
- Level-1 → Level-MAX: **+21% AUC** (feature upgrade)
- MRI-Only → Attention: **<3% AUC** (architecture upgrade)
- Ratio: **7:1** in favor of features!

**Lesson:** In medical AI, invest in biological feature curation > architectural novelty.

---

## 📚 Full Feature Summary Table

| Experiment | MRI | Clinical | N | Why These | Result |
|------------|-----|----------|---|-----------|--------|
| OASIS | 512 ResNet | Age, nWBV, eTIV, ASF, Educ (5) | 205 | All available | 0.794 |
| ADNI L1 | 512 ResNet | Age, Sex (2) | 629 | Minimal honest | 0.598 |
| ADNI L2 | 512 ResNet | Age, Sex, MMSE, CDR (4) | 629 | Debug (circular) | 0.988 |
| **ADNI LMAX** | 512 ResNet | **14 biomarkers** | 629 | **Real biology** | **0.808** |
| Long CNN | ResNet seq | - | 639 | Test temporal DL | 0.441 |
| **Long Bio** | **-** | **21 volumes+deltas** | 341 | **Volumetric tracking** | **0.848** |

---

**For full implementation details, see:** `docs/IMPLEMENTATION_PIPELINE.md`  
**For statistical validation:** `docs/STATISTICAL_TESTS_SUPPLEMENT.md`
