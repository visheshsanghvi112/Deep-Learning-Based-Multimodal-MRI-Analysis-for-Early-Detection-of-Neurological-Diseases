# Data Cleaning & Preprocessing Documentation

**Deep Learning-Based Multimodal MRI Analysis for Early Detection of Neurological Diseases**

**Document Purpose:** Comprehensive enumeration of all data cleaning, preprocessing, and feature engineering steps applied to OASIS-1 and ADNI-1 datasets.  
**Date:** December 24, 2025  
**Status:** Documentation-Only (No Code Changes)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Data Cleaning Steps](#data-cleaning-steps)
3. [Before vs. After Comparison Tables](#before-vs-after-comparison-tables)
4. [Feature Engineering & Selection](#feature-engineering--selection)
5. [What Was NOT Done (And Why)](#what-was-not-done-and-why)
6. [Leakage Prevention Measures](#leakage-prevention-measures)
7. [References to Implementation](#references-to-implementation)

---

## Executive Summary

This document enumerates all structural and semantic data cleaning steps applied to **OASIS-1** (N=436 scans) and **ADNI-1** (N=629 baseline scans) datasets before training early-detection dementia models. 

**Key Principle:** Most cleaning was **structural** (de-duplication, visit selection, subject-wise splits) and **semantic** (feature exclusion, level separation), not numerical (no imputation, no oversampling, no domain adaptation).

**Critical Distinction:**
- **Level-1 Models:** "Honest" early-detection experiments using only MRI + weak demographics (Age, Sex, Education)
- **Level-2 Models:** "Upper-bound reference" experiments including downstream cognitive scores (MMSE, CDR-SB)

---

## Data Cleaning Steps

### 1. Subject-Level De-duplication

#### 1.1 OASIS-1 De-duplication

**What Was Done:**
- OASIS-1 dataset contains 436 raw scan entries across 12 disc folders (`disc1` through `disc12`)
- Each scan represents a unique MRI session for a subject
- Scans were treated as independent cross-sectional samples (no longitudinal structure in OASIS-1)
- Subject IDs extracted from folder naming convention: `OAS1_XXXX_MR1`

**Why It Was Necessary:**
- OASIS-1 is inherently cross-sectional, so each scan is a unique baseline
- No subject appears multiple times in the dataset
- Verification: 436 unique MRI folders = 436 unique subjects

**What Error/Leakage It Prevents:**
- Prevents counting the same subject multiple times in different experimental conditions
- Ensures each data point represents an independent biological sample

**Implementation:**
```
Source: project/scripts/mri_feature_extraction.py
Method: Folder-based enumeration with Subject ID parsing
Verification: len(subject_ids) == 436 unique values
```

---

#### 1.2 ADNI-1 De-duplication

**What Was Done:**
- Raw ADNI registry contained **1,825 scans** across multiple visits per subject
- Identified **629 unique subjects** by PTID (Patient ID)
- Applied strict baseline selection (see Section 2 below)
- Final dataset: **629 unique subjects, one scan each**

**Why It Was Necessary:**
- ADNI is a **longitudinal study** with multiple follow-up visits (bl, m06, m12, m24, etc.)
- Using multiple time points from the same subject creates **temporal leakage**
- Early-detection task requires baseline-only data (pre-diagnosis progression)

**What Error/Leakage It Prevents:**
- **Temporal leakage:** Prevents model from learning disease progression patterns instead of baseline characteristics
- **Subject leakage:** Ensures no subject contributes both to training and testing sets via different time points
- **Label leakage:** Prevents using future diagnosis labels to inform baseline predictions

**Implementation:**
```
Source: project_adni/src/baseline_selection.py
Method: Group-by PTID, select earliest baseline scan
Verification: df['Subject'].nunique() == len(df) == 629
```

---

### 2. Baseline-Only Visit Selection

**What Was Done:**

**Selection Priority (ADNI-1):**
1. **Priority 1:** Select scans with visit code `'sc'` (screening/baseline)
2. **Priority 2:** If no `'sc'` available, select scan with earliest `Acq_Date`

**Selection Results:**
- 611 subjects had explicit `'sc'` visits (97.1%)
- 18 subjects selected via earliest-date fallback (2.9%)
- Total: 629 subjects with exactly one baseline scan each

**Why It Was Necessary:**
- Longitudinal datasets include follow-up visits (m06, m12, m24) that contain **progressed disease states**
- Early detection requires **baseline characteristics only**, before substantial cognitive decline
- Using follow-up visits would answer "Can we detect progression?" (different research question)

**What Error/Leakage It Prevents:**
- **Disease progression leakage:** Prevents model from learning late-stage disease markers instead of early-stage subtle changes
- **Temporal ordering leakage:** Prevents model from "seeing the future" (i.e., using disease trajectory to inform baseline prediction)
- **Circular reasoning:** Avoids training on data where the outcome (cognitive decline) has already manifested visibly

**Implementation:**
```
Source: project_adni/src/baseline_selection.py
Function: select_baseline_scans()
Input:  ADNI1_Complete_1Yr_1.5T_12_19_2025.csv (1,825 scans)
Output: adni_baseline_selection.csv (629 subjects)
```

**Code Excerpt:**
```python
for subject_id, group_df in df.groupby('Subject'):
    # Priority 1: Look for 'sc' (screening/baseline) visit
    sc_scans = group_df[group_df['Visit'] == 'sc']
    if len(sc_scans) > 0:
        selected = sc_scans.sort_values('ParsedDate').iloc[0]
    else:
        # Priority 2: No 'sc' available, select earliest date
        selected = group_df.sort_values('ParsedDate').iloc[0]
```

---

### 3. Removal of Longitudinal Leakage

**What Was Done:**
- All non-baseline visits (m06, m12, m18, m24, m36, etc.) were **completely discarded**
- Only baseline scans (sc/bl/earliest) were retained for feature extraction
- Train/test splits were created **after** baseline selection, not before

**Why It Was Necessary:**
- In longitudinal studies, disease progression creates a **temporal dependency**
- Using longitudinal data mixes "early detection" with "progression modeling"
- Research question is: "Can baseline MRI predict future cognitive decline?" not "Can we track decline over time?"

**What Error/Leakage It Prevents:**
- **Future information leakage:** Model cannot use knowledge of disease trajectory
- **Label shift:** Prevents diagnostic labels from changing over time within the same training set
- **Confounding:** Removes age-related progression effects that confound early detection signals

**Implementation:**
```
Source: project_adni/src/baseline_selection.py (implicit via visit filtering)
Verification: No subject has >1 scan in baseline selection output
```

---

### 4. Subject-Wise Train/Test Splitting

#### 4.1 OASIS-1 Splitting Strategy

**What Was Done:**
- Filtered dataset to **binary classification task:** CDR=0 (Normal) vs CDR=0.5 (Very Mild Dementia)
- Excluded CDR=1.0, CDR=2.0 (manifest dementia), and CDR=NaN (young controls without dementia screening)
- Usable dataset: **205 subjects** (135 CDR=0, 70 CDR=0.5)
- Split: **80% train (164 subjects), 20% test (41 subjects)**
- Stratified by CDR label to preserve class balance
- Random seed: 42 (for reproducibility)

**Why It Was Necessary:**
- Subject-wise splitting ensures **independence** between train and test sets
- Prevents data leakage where different scans of the same subject appear in both sets
- Stratification ensures both sets have similar class distributions

**What Error/Leakage It Prevents:**
- **Subject identity leakage:** Model cannot memorize per-subject idiosyncrasies
- **Data snooping:** Test set remains completely unseen during training
- **Optimistic bias:** Prevents inflated performance due to subject overlap

**Implementation:**
```
Source: project_adni/src/cross_dataset_robustness.py (lines 168-170)
Method: sklearn.model_selection.train_test_split with stratify
Code:
  mri_train, mri_test, clin_train, clin_test, y_train, y_test = train_test_split(
      mri, clinical_selected, y, test_size=0.2, stratify=y, random_state=42
  )
```

---

#### 4.2 ADNI-1 Splitting Strategy

**What Was Done:**
- Binary classification: **CN (Cognitively Normal) = 0** vs **MCI+AD (Impaired) = 1**
- Total subjects: 629 (after baseline selection)
- Split: **80% train (503 subjects), 20% test (126 subjects)**
- Stratified by diagnosis Group (CN, MCI, AD) to preserve class proportions
- Random seed: 42 (consistent with OASIS)

**Why It Was Necessary:**
- Subject-wise splitting is **mandatory** for valid generalization testing
- ADNI subjects have rich metadata; splitting by scan would leak subject-level information
- Stratification prevents class imbalance artifacts (e.g., all AD cases in training set)

**What Error/Leakage It Prevents:**
- **Subject leakage:** Ensures no subject contributes to both training and testing
- **Distribution shift:** Maintains similar class ratios across train/test
- **Reproducibility:** Fixed random seed allows exact replication

**Implementation:**
```
Source: project_adni/src/data_split.py (lines 34-39)
Method: sklearn.model_selection.train_test_split with stratification
Verification:
  - Assert df['Subject'].nunique() == len(df) (one row per subject)
  - Overlap check: len(train_subjects ∩ test_subjects) == 0
Code:
  train_df, test_df = train_test_split(
      df, test_size=0.20, stratify=df['Group'], random_state=42
  )
```

**Post-Split Verification:**
```
✓ Verified: NO subject overlap between train and test
Train: CN=40.2%, MCI=39.8%, AD=20.0%
Test:  CN=40.5%, MCI=39.7%, AD=19.8%
```

---

### 5. Feature Intersection Enforcement for Cross-Dataset Experiments

**What Was Done:**

**Cross-Dataset Transfer Experiments (OASIS ↔ ADNI):**
- Identified **feature intersection** across both datasets:
  - **MRI Features:** 512-dimensional ResNet18 embeddings (identical architecture for both datasets)
  - **Clinical Features:** Age, Education (available in both OASIS and ADNI)
  - **Excluded:** Sex (missing in some subjects), MMSE (not in OASIS baseline), CDR-SB (not in OASIS)

**OASIS Feature Mapping:**
```
clinical_features index 0 = Age
clinical_features index 5 = Education
→ clinical_selected = clin[:, [0, 5]]  # Shape: (N, 2)
```

**ADNI Feature Mapping:**
```
Age from CSV column 'Age'
Education from ADNIMERGE.csv column 'PTEDUCAT' (merged on PTID)
→ clinical = df[['Age', 'PTEDUCAT']].values  # Shape: (N, 2)
```

**Why It Was Necessary:**
- Cross-dataset experiments test **domain robustness**, not feature availability
- Using dataset-specific features (e.g., ADNI's CDR-SB) would create unfair comparisons
- Intersection ensures both source and target domains have identical feature spaces

**What Error/Leakage It Prevents:**
- **Feature leakage:** Prevents using privileged information only available in one dataset
- **Domain-specific overfitting:** Ensures models learn generalizable patterns, not dataset artifacts
- **Unfair comparison:** Enables direct AUC comparison across domains

**Implementation:**
```
Source: project_adni/src/cross_dataset_robustness.py
OASIS: Lines 160-162 (select Age, Educ from indices [0, 5])
ADNI:  Lines 186-203 (merge Education from ADNIMERGE, extract Age, Educ)
```

**Code Excerpt:**
```python
# OASIS
clinical_selected = clin[:, [0, 5]]  # Age (0), Education (5)

# ADNI
bl_merge = merge_df[merge_df['VISCODE'].isin(['bl', 'sc', 'scmri', 'm00'])].copy()
educ_map = bl_merge[['PTID', 'PTEDUCAT']].dropna().drop_duplicates(subset='PTID')
df = df.merge(educ_map, left_on='Subject', right_on='PTID', how='left')
clinical = df[['Age', 'PTEDUCAT']].values  # Shape: (N, 2)
```

---

### 6. Explicit Exclusion of Cognitively Downstream Features

**What Was Done:**

**Level-1 Experiments (Early Detection - Primary Claims):**
- **Excluded Features:**
  - MMSE (Mini-Mental State Examination, 0-30 scale)
  - CDR-SB (Clinical Dementia Rating - Sum of Boxes)
  - ADAS-Cog (Alzheimer's Disease Assessment Scale - Cognitive)
  - MoCA (Montreal Cognitive Assessment)
  
- **Included Features (Level-1):**
  - MRI: 512-dimensional ResNet18 embeddings
  - Demographics: Age, Sex
  - Education (when available in feature intersection)

**Level-2 Experiments (Upper-Bound Reference - NOT for Early Detection Claims):**
- **Included Features (Level-2 only):**
  - MMSE, CDR-SB (cognitively downstream measures)
  - APOE4 (genetic risk marker)
  - Age, Education

**Why It Was Necessary:**
- MMSE/CDR-SB **directly measure cognitive impairment** - the outcome we're trying to predict
- Including them creates **circular reasoning:** "Can we predict cognitive impairment using cognitive impairment scores?"
- Early detection requires predicting **before** clinical cognitive tests show impairment

**What Error/Leakage It Prevents:**
- **Circular feature leakage:** MMSE ≈ 0.87 correlation with CDR (the label)
- **Outcome contamination:** Cognitive scores are proxy measures of the diagnosis itself
- **Invalid claims:** Prevents claiming "early detection" when using post-diagnosis measurements

**Real-World Impact:**
```
Level-1 (No MMSE):  AUC = 0.598 (realistic early-detection performance)
Level-2 (With MMSE): AUC = 0.988 (proves model capacity, but circular)
```

**Implementation:**
```
Source: 
  - Level-1: project_adni/src/train_level1.py (lines 74-76, only Age and Sex)
  - Level-2: project_adni/src/train_level2.py (lines 136, includes MMSE, CDRSB)
  - Cross-Dataset: project_adni/src/cross_dataset_robustness.py (lines 160-162, only Age, Educ)
```

**Code Verification:**
```python
# Level-1 (train_level1.py)
sex_encoded = (df['Sex'] == 'M').astype(np.float32).values
age = df['Age'].values.astype(np.float32)
clinical = np.column_stack([age, sex_encoded])  # ONLY Age + Sex

# Level-2 (train_level2.py)
clinical_cols = ['MMSE', 'CDRSB', 'PTEDUCAT', 'AGE', 'APOE4']
train_clinical = train_df[clinical_cols].values  # Includes cognitive scores
```

---

### 7. Separation of Level-1 (MRI + Weak Demographics) vs Level-2 (Clinical-Informed Upper Bound)

**What Was Done:**

**Two-Tier Experimental Design:**

| Tier | Purpose | Features | Valid Claims |
|------|---------|----------|--------------|
| **Level-1** | Realistic early detection | MRI + Age + Sex (+ Education for cross-dataset) | ✅ Primary research claims |
| **Level-2** | Upper-bound reference | MRI + MMSE + CDR-SB + Age + APOE4 + Education | ⚠️ Proof of model capacity only |

**Level-1 Details:**
- **Input:** ResNet18 MRI features (512-dim) + Age + Sex
- **Performance:** AUC = 0.583–0.607 (ADNI), AUC = 0.78 (OASIS)
- **Interpretation:** This is the **honest baseline** for early detection
- **Clinical Scenario:** Screening-level prediction using only imaging + basic demographics

**Level-2 Details:**
- **Input:** ResNet18 MRI features (512-dim) + MMSE + CDR-SB + Age + APOE4 + Education
- **Performance:** AUC = 0.988 (ADNI)
- **Interpretation:** This proves the model **can learn** when given diagnostic-level features
- **Clinical Scenario:** NOT applicable for early detection (circular reasoning)

**Why It Was Necessary:**
- **Honest reporting:** Separates realistic performance from artificially inflated results
- **Methodological transparency:** Acknowledges the circular nature of cognitive scores
- **Benchmark establishment:** Level-2 serves as a sanity check (if model fails even with MMSE, architecture is broken)

**What Error/Leakage It Prevents:**
- **Claim inflation:** Prevents reporting Level-2 AUC (0.988) as "early detection" performance
- **Methodological confusion:** Clearly delineates which features are acceptable for early detection
- **Literature comparison:** Allows fair comparison with papers that do/don't use cognitive scores

**Implementation:**
```
Level-1 Training: project_adni/src/train_level1.py
Level-2 Training: project_adni/src/train_level2.py
Cross-Dataset:    project_adni/src/cross_dataset_robustness.py (uses Level-1 features only)
```

**Results Documentation:**
```
FINAL_PAPER_DRAFT.md:
  - Line 42-46: Level-1 results (AUC 0.583-0.598, "Honest Baseline")
  - Line 48-52: Level-2 results (AUC 0.988, "Circular Upper Bound")
  - Line 29: Explicit warning about label shift and circular features
```

---

## Before vs. After Comparison Tables

### Table 1: OASIS-1 Subject Counts

| Processing Stage | Count | Notes |
|------------------|-------|-------|
| **Raw MRI Scans** | 436 | All subjects across disc1-disc12 |
| **After Baseline Selection** | 436 | No change (OASIS is already cross-sectional) |
| **After Label Filtering** | 205 | Filtered to CDR=0 vs CDR=0.5 only |
| **Training Set** | 164 | 80% split, stratified by CDR |
| **Test Set** | 41 | 20% split, stratified by CDR |

**Label Distribution (Filtered Dataset):**
```
CDR = 0.0  (Normal):        135 subjects (65.9%)
CDR = 0.5  (Very Mild):      70 subjects (34.1%)
─────────────────────────────────────────────
Total Usable:               205 subjects
Excluded (CDR=1, 2, NaN):   231 subjects
```

**Feature Counts:**
```
Raw clinical features:   6 (Age, MMSE, nWBV, eTIV, ASF, Education)
MRI features:          512 (ResNet18 embeddings)
Level-1 features:        2 (Age, Education) for cross-dataset
Total feature vector:  514 (512 MRI + 2 clinical)
```

---

### Table 2: ADNI-1 Subject Counts

| Processing Stage | Count | Reduction | Notes |
|------------------|-------|-----------|-------|
| **Raw CSV Scans** | 1,825 | — | All visits, all subjects |
| **Unique Subjects (PTID)** | 629 | -65.5% | De-duplicated by subject ID |
| **After Baseline Selection** | 629 | 0% | One scan per subject (sc/earliest) |
| **Training Set** | 503 | — | 80% split, stratified by Group |
| **Test Set** | 126 | — | 20% split, stratified by Group |

**Visit Distribution (Before Baseline Selection):**
```
Baseline (sc):      611 scans
Follow-ups (m06-m36): 1,214 scans  ← DISCARDED
─────────────────────────────────
Total:            1,825 scans
Selected:           629 baseline-only
```

**Diagnosis Distribution (After Baseline Selection):**
```
CN (Cognitively Normal):  254 subjects (40.4%)
MCI (Mild Cognitive Imp): 250 subjects (39.7%)
AD (Alzheimer's Disease): 125 subjects (19.9%)
─────────────────────────────────────────────
Total:                    629 subjects
```

**Feature Counts (Level-1 vs Level-2):**
```
                    Level-1    Level-2
─────────────────────────────────────
MRI features         512        512
Clinical features      2          5
  - Age               ✓          ✓
  - Sex               ✓          —
  - Education         —          ✓
  - MMSE              ✗          ✓  ← Circular
  - CDR-SB            ✗          ✓  ← Circular
  - APOE4             ✗          ✓
─────────────────────────────────────
Total features       514        517
```

---

### Table 3: Cross-Dataset Feature Intersection

| Feature Type | OASIS | ADNI | Intersection | Used in Cross-Dataset? |
|--------------|-------|------|--------------|------------------------|
| **MRI (ResNet18)** | ✓ (512-dim) | ✓ (512-dim) | ✓ | ✅ Yes (identical architecture) |
| **Age** | ✓ | ✓ | ✓ | ✅ Yes |
| **Education** | ✓ | ✓ | ✓ | ✅ Yes |
| **Sex** | ✓ | ✓ | ✓ | ❌ No (missing in some ADNI subjects) |
| **MMSE** | ⚠️ (available but excluded) | ⚠️ (available but excluded) | — | ❌ No (circular feature) |
| **CDR-SB** | ✗ | ✓ | — | ❌ No (not in OASIS) |
| **nWBV** | ✓ | ✗ | — | ❌ No (not in ADNI baseline) |

**Final Cross-Dataset Feature Set:**
- MRI: 512 features
- Clinical: 2 features (Age, Education)
- **Total: 514 features**

---

## Feature Engineering & Selection

### 1. MRI Feature Extraction

**Method:** 2.5D Multi-Slice ResNet18

**What Was Done:**
- Pretrained ResNet18 (ImageNet weights) used for transfer learning
- Extract features from **3 anatomical planes:** axial, coronal, sagittal
- **3 slices per plane:** center slice ± 20 voxels
- **Total: 9 slices per subject**
- Output: 512-dimensional feature vector per slice
- Aggregation: **Mean pooling** across all 9 slices → final 512-dim embedding

**Why This Approach:**
- Full 3D CNNs require massive GPU memory (6.4M voxels per scan)
- Small dataset (N=205–629) → high overfitting risk with 3D CNNs
- 2.5D captures 3D spatial information efficiently while leveraging pretrained 2D models
- Mean pooling reduces variance and creates permutation-invariant representation

**Normalization:**
- Per-slice intensity normalization to [0, 1] range
- StandardScaler applied to final 512-dim features before training (fit on train, transform on test)

**Implementation:**
```
Source: project/scripts/mri_feature_extraction.py
Architecture: torchvision.models.resnet18(pretrained=True)
Extraction: features_512 = model.avgpool(conv_features).squeeze()
```

---

### 2. Clinical Feature Engineering

#### OASIS-1 Clinical Features

**Raw Features (6 total):**
1. **Age:** Subject age in years
2. **MMSE:** Mini-Mental State Exam (0-30) ← **Excluded in Level-1**
3. **nWBV:** Normalized whole brain volume
4. **eTIV:** Estimated total intracranial volume
5. **ASF:** Atlas scaling factor
6. **EDUC:** Education level (1-5 categorical)

**Level-1 Selection:**
- **Used:** Age, Education (indices [0, 5])
- **Excluded:** MMSE (circular), nWBV/eTIV/ASF (not in ADNI baseline)

**Z-Score Normalization:**
```
Age:  μ=50,   σ=20   → z = (age - 50) / 20
EDUC: μ=3,    σ=1.5  → z = (educ - 3) / 1.5
```

---

#### ADNI-1 Clinical Features

**Level-1 Features (2 total):**
1. **Age:** Acquisition age (from CSV column `'Age'`)
2. **Sex:** Binary encoding (M=1, F=0)

**Level-2 Features (5 total):**
1. **MMSE:** Mini-Mental State Exam (merged from ADNIMERGE.csv)
2. **CDR-SB:** Clinical Dementia Rating Sum of Boxes (merged from ADNIMERGE.csv)
3. **PTEDUCAT:** Years of education (merged from ADNIMERGE.csv)
4. **AGE:** Subject age
5. **APOE4:** APOE ε4 allele count (0, 1, or 2; missing filled with 0)

**Cross-Dataset Features (2 total):**
- **Age:** Direct extraction
- **Education (PTEDUCAT):** Merged from ADNIMERGE via PTID key

**Normalization:**
- StandardScaler fit on training set, applied to test set
- APOE4 missing values imputed with 0 (= no risk allele)

**Implementation:**
```
Source: project_adni/src/train_level1.py (Level-1)
        project_adni/src/train_level2.py (Level-2)
        project_adni/src/cross_dataset_robustness.py (Cross-dataset)
```

---

### 3. Label Engineering

#### OASIS-1 Binary Labels

**Raw Labels:** CDR (Clinical Dementia Rating) {0, 0.5, 1.0, 2.0, NaN}

**Binary Conversion:**
```
CDR = 0   → Label = 0 (Normal)
CDR = 0.5 → Label = 1 (Very Mild Dementia / MCI-like)
CDR ≥ 1.0 → EXCLUDED (manifest dementia, not early detection)
CDR = NaN → EXCLUDED (young controls, no dementia screening)
```

**Final Distribution:** 135 Normal (0) vs 70 Very Mild (1)

---

#### ADNI-1 Binary Labels

**Raw Labels:** Group {CN, MCI, AD}

**Binary Conversion:**
```
CN (Cognitively Normal) → Label = 0
MCI (Mild Cog. Impair)  → Label = 1
AD (Alzheimer's Disease) → Label = 1
```

**Rationale:**
- CN = healthy baseline
- MCI+AD = impaired (both represent disease presence)
- Task: "Does baseline MRI show ANY cognitive impairment signal?"

**Final Distribution:** 254 CN (0) vs 375 MCI+AD (1)

---

## What Was NOT Done (And Why)

This section documents deliberate **non-actions** taken to preserve data integrity and ensure honest evaluation.

---

### 1. No Imputation of Missing Values

**What We Did NOT Do:**
- NO mean/median imputation for missing clinical features
- NO model-based imputation (KNN, MICE, etc.)
- NO forward/backward fill for longitudinal data

**What We Did Instead:**
- **Dropped rows with missing values** in required features
- Level-2 APOE4: Missing values filled with 0 (biologically justified: "no risk allele detected")
- Education (ADNIMERGE merge): Subjects without education data excluded from Level-2 experiments

**Why We Chose This:**
1. **Small dataset size:** Imputation in small samples (N=200–600) creates spurious correlations
2. **Honest uncertainty:** Missing data reflects real-world data quality issues
3. **Leakage prevention:** Imputation can leak label information via distributional patterns
4. **Reproducibility:** Explicit handling is more transparent than hidden imputation

**Impact:**
```
ADNI Level-2: 629 subjects → 623 subjects with complete clinical data (6 excluded)
OASIS: No impact (clinical data already complete for CDR=0/0.5 subjects)
```

**Alternative Considered (Rejected):**
- Multiple imputation (MICE): Too complex for small sample size
- Mean imputation: Creates artificial central tendency, reduces variance

---

### 2. No Oversampling or Class Rebalancing

**What We Did NOT Do:**
- NO SMOTE (Synthetic Minority Over-sampling Technique)
- NO ADASYN (Adaptive Synthetic Sampling)
- NO random oversampling of minority class
- NO random undersampling of majority class

**What We Did Instead:**
- Used **class-weighted loss functions** during training
- Weighted loss inversely proportional to class frequency
- Stratified sampling during train/test splits

**Why We Chose This:**
1. **Synthetic data skepticism:** SMOTE can create unrealistic "synthetic" brain scans (especially with high-dimensional MRI features)
2. **Overfitting risk:** Oversampling amplifies noise in minority class
3. **Realistic evaluation:** Test set reflects true class imbalance (clinically relevant)
4. **Metric choice:** AUC (our primary metric) is insensitive to class imbalance

**Impact:**
```
OASIS: 135 Normal vs 70 Very Mild (1.93:1 ratio) → Retained as-is
ADNI:  254 CN vs 375 Impaired (1:1.48 ratio)    → Retained as-is

Loss weighting:
  CN weight:  1 / 254 = 0.00394 → normalized → 0.60
  MCI+AD weight: 1 / 375 = 0.00267 → normalized → 0.40
```

**Class Weighting Implementation:**
```python
# Source: train_level1.py, lines 270-275
train_labels = np.concatenate([batch['label'].numpy() for batch in train_loader])
class_counts = np.bincount(train_labels)
class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
class_weights = class_weights / class_weights.sum()  # Normalize
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Alternative Considered (Rejected):**
- SMOTE: Risk of generating biologically implausible MRI feature vectors
- Undersampling: Would discard valuable CN samples (already limited data)

---

### 3. No Domain Adaptation or Transfer Learning Tuning

**What We Did NOT Do:**
- NO fine-tuning of ResNet18 backbone on MRI data
- NO domain adversarial training (DANN, CORAL, etc.)
- NO domain-specific batch normalization adjustments
- NO data augmentation beyond inherent 2.5D multi-slice extraction

**What We Did Instead:**
- Used **frozen pretrained ResNet18** (ImageNet weights) as feature extractor
- Trained only small classifier heads (MLP with 32–64 hidden units)
- Applied **same preprocessing** to both OASIS and ADNI (intensity normalization, slice selection)

**Why We Chose This:**
1. **Honest evaluation:** Domain adaptation would artificially improve cross-dataset transfer AUC
2. **Dataset size:** Fine-tuning 11M-parameter ResNet18 on N=200–600 samples → severe overfitting
3. **Transfer learning validity:** Pretrained features capture low-level visual patterns (edges, textures) applicable to MRI
4. **Reproducibility:** Frozen features provide deterministic, reusable embeddings

**Impact:**
```
Cross-dataset transfer AUC without domain adaptation:
  OASIS → ADNI: 0.607 (MRI-Only)
  ADNI → OASIS: 0.569 (MRI-Only)

This is LOWER than in-dataset AUC (0.78–0.81), which is expected and honest.
Domain adaptation could boost these by ~5-10%, but would obscure true generalization gap.
```

**Alternative Considered (Rejected):**
- Fine-tuning ResNet18: 11M parameters vs 200–600 samples → overfitting guaranteed
- CycleGAN-based MRI harmonization: Adds complexity, potential for synthetic artifacts

---

### 4. No Feature Selection or Dimensionality Reduction

**What We Did NOT Do:**
- NO PCA (Principal Component Analysis) on MRI features
- NO LASSO/Elastic Net for feature selection
- NO manual removal of correlated features
- NO feature importance-based pruning

**What We Did Instead:**
- Used **all 512 ResNet18 features** as-is
- Applied dropout (p=0.5) in classifier to implicitly handle redundancy
- Used L2 regularization (weight_decay=1e-4) to penalize large weights

**Why We Chose This:**
1. **Pretrained features are curated:** ResNet18 learned 512 meaningful features from ImageNet
2. **Biological complexity:** Brain pathology is multifactorial; aggressive feature selection risks discarding subtle signals
3. **Regularization suffices:** Dropout + weight decay handle collinearity without explicit removal
4. **Reproducibility:** Using raw ResNet features allows direct comparison across studies

**Impact:**
```
No dimensionality reduction → Full 512-dim MRI features retained
Classifier parameters: ~18K (512→32→16→2 layers)
Small classifier prevents overfitting despite high-dimensional input
```

**Alternative Considered (Rejected):**
- PCA: Would discard biological interpretability (PC1, PC2, etc. are abstract)
- LASSO: Sparse models may discard complementary weak signals

---

### 5. No Multi-Task Learning or Auxiliary Losses

**What We Did NOT Do:**
- NO multi-task learning (e.g., jointly predict CDR + age)
- NO contrastive losses (SimCLR, supervised contrastive, etc.)
- NO reconstruction losses (autoencoder-style)
- NO triplet losses or metric learning

**What We Did Instead:**
- Simple **binary cross-entropy loss** (2-class softmax)
- Class-weighted to handle imbalance
- Early stopping based on training loss (no separate validation split due to small N)

**Why We Chose This:**
1. **Simplicity:** Multi-task losses add hyperparameters (loss weighting, auxiliary task design)
2. **Small datasets:** Auxiliary tasks require additional labeled data (e.g., age regression needs age labels for all subjects)
3. **Honest claims:** Complex losses can obscure what the model actually learned ("Did it learn dementia or age?")

**Impact:**
```
Single-task binary classification → Clear interpretation
AUC directly measures dementia detection ability
No confounding from auxiliary task performance
```

---

## Leakage Prevention Measures

This section enumerates all **active measures** taken to prevent data leakage.

---

### 1. Temporal Leakage Prevention

**Threat:** Using future time points to predict baseline outcomes.

**Prevention Measures:**
- ✅ **Baseline-only selection:** Only sc/bl visits used (no m06, m12, etc.)
- ✅ **No longitudinal features:** Discarded all follow-up scans
- ✅ **Label freeze:** Diagnosis labels taken from baseline visit only

**Verification:**
```python
# Source: baseline_selection.py, line 49
sc_scans = group_df[group_df['Visit'] == 'sc']
# Only 'sc' or earliest visit selected, never follow-up visits
```

---

### 2. Subject Identity Leakage Prevention

**Threat:** Same subject appearing in both training and test sets.

**Prevention Measures:**
- ✅ **Subject-wise splitting:** Splits performed after grouping by Subject ID
- ✅ **Overlap verification:** Explicit check that train_subjects ∩ test_subjects = ∅
- ✅ **Stratified sampling:** Maintains class balance while preserving subject independence

**Verification:**
```python
# Source: data_split.py, lines 62-69
train_subjects = set(train_df['Subject'])
test_subjects = set(test_df['Subject'])
overlap = train_subjects.intersection(test_subjects)
assert len(overlap) == 0, "Subject overlap detected!"
```

---

### 3. Label Leakage Prevention

**Threat:** Features that directly encode the outcome variable.

**Prevention Measures:**
- ✅ **MMSE excluded from Level-1:** MMSE ≈ 0.87 correlation with CDR
- ✅ **CDR-SB excluded from Level-1:** CDR-SB is derived from CDR (the label)
- ✅ **Level-2 marked as circular:** Explicit warnings in documentation and code comments

**Verification:**
```python
# Source: train_level1.py, lines 74-76
clinical = np.column_stack([age, sex_encoded])  # ONLY Age + Sex
# MMSE NOT in feature set
```

**Documentation:**
```markdown
# Source: FINAL_PAPER_DRAFT.md, line 21
"Exclusion of Circular Features: For all early-detection experiments (Level-1 
and Cross-Dataset), we explicitly EXCLUDED cognitively downstream measures 
including MMSE, CDR-SB, ADAS-Cog, and MoCA."
```

---

### 4. Data Snooping Prevention

**Threat:** Using test set for model selection, hyperparameter tuning.

**Prevention Measures:**
- ✅ **No validation split:** Early stopping based on training loss (small N doesn't allow train/val/test split)
- ✅ **Fixed hyperparameters:** Same hyperparameters for OASIS and ADNI (no tuning per dataset)
- ✅ **Test set touched ONCE:** Final evaluation only; no iterative tuning

**Hyperparameter Freeze:**
```python
# Source: train_level1.py, lines 47-52
# IDENTICAL to OASIS (no ADNI-specific tuning)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.5
PATIENCE = 15
BATCH_SIZE = 16
```

---

### 5. Feature Distribution Leakage Prevention

**Threat:** Normalizing across train+test jointly (leaks test statistics).

**Prevention Measures:**
- ✅ **Fit scaler on train only:** StandardScaler.fit(train_data)
- ✅ **Transform test separately:** scaler.transform(test_data)
- ✅ **Cross-dataset freezing:** Scaler fit on source domain, applied to target domain

**Implementation:**
```python
# Source: train_level1.py, lines 344-351
mri_scaler = StandardScaler()
clinical_scaler = StandardScaler()

train_mri_scaled = mri_scaler.fit_transform(train_mri)      # FIT on train
train_clinical_scaled = clinical_scaler.fit_transform(train_clinical)

test_mri_scaled = mri_scaler.transform(test_mri)            # TRANSFORM test
test_clinical_scaled = clinical_scaler.transform(test_clinical)
```

---

### 6. Cross-Dataset Leakage Prevention

**Threat:** Using features available only in one dataset.

**Prevention Measures:**
- ✅ **Feature intersection enforcement:** Only MRI + Age + Education used
- ✅ **No dataset-specific features:** CDR-SB (ADNI-only), nWBV (OASIS-only) excluded
- ✅ **Frozen source scaler:** Normalization statistics from source applied to target (no target statistics used)

**Verification:**
```python
# Source: cross_dataset_robustness.py, lines 276-286
# Scaler frozen on Source
scaler_mri = StandardScaler().fit(src_mri)     # FIT on source
scaler_clin = StandardScaler().fit(src_clin)

src_mri_s = scaler_mri.transform(src_mri)
src_clin_s = scaler_clin.transform(src_clin)
tgt_mri_s = scaler_mri.transform(tgt_mri)      # TRANSFORM target (no fit)
tgt_clin_s = scaler_clin.transform(tgt_clin)
```

---

## References to Implementation

All cleaning steps are implemented across multiple scripts. Below is a mapping of each step to its source code.

### Scripts by Purpose

| Script | Purpose | Key Functions |
|--------|---------|---------------|
| `project_adni/src/baseline_selection.py` | Step 2: Baseline visit selection | `select_baseline_scans()` |
| `project_adni/src/data_split.py` | Step 4: Subject-wise train/test split | `perform_split()` |
| `project_adni/src/train_level1.py` | Level-1 training (early detection) | `load_adni_data()`, `main()` |
| `project_adni/src/train_level2.py` | Level-2 training (upper-bound) | `merge_clinical_features()`, `load_data_level2()` |
| `project_adni/src/cross_dataset_robustness.py` | Cross-dataset transfer experiments | `load_oasis_with_split()`, `load_adni_splits()` |
| `project/scripts/mri_feature_extraction.py` | MRI feature extraction (OASIS) | 2.5D ResNet18 extraction |
| `project_adni/src/feature_extraction.py` | MRI feature extraction (ADNI) | ResNet18 on ADNI scans |

---

### Line-Level References

#### Subject De-duplication
```
OASIS: project/scripts/mri_feature_extraction.py (implicit via folder enumeration)
ADNI:  project_adni/src/baseline_selection.py, line 47 (for subject_id, group_df in df.groupby('Subject'))
```

#### Baseline Visit Selection
```
project_adni/src/baseline_selection.py:
  - Line 49: sc_scans = group_df[group_df['Visit'] == 'sc']
  - Line 58: sorted_by_date = group_df.sort_values('ParsedDate')
```

#### Subject-Wise Splitting
```
OASIS: project_adni/src/cross_dataset_robustness.py, line 168-170
ADNI:  project_adni/src/data_split.py, line 34-39
```

#### Feature Intersection
```
project_adni/src/cross_dataset_robustness.py:
  - Line 162: OASIS clinical_selected = clin[:, [0, 5]]
  - Line 203: ADNI clinical = df[['Age', 'PTEDUCAT']].values
```

#### Feature Exclusion
```
Level-1: project_adni/src/train_level1.py, line 74-76 (Age, Sex only)
Level-2: project_adni/src/train_level2.py, line 136 (includes MMSE, CDRSB)
```

#### Class Weighting (NO oversampling)
```
project_adni/src/train_level1.py, line 270-275
project_adni/src/train_level2.py, line 256-261
```

---

## Conclusion

This document has enumerated **all** data cleaning and preprocessing steps applied to OASIS-1 and ADNI-1 datasets. Key principles:

1. **Structural cleaning** (de-duplication, baseline selection, subject-wise splits) was **mandatory** and **strictly enforced**
2. **Semantic cleaning** (feature exclusion, level separation) was **critical** to prevent circular reasoning
3. **No numerical imputation, oversampling, or domain adaptation** was performed to ensure **honest evaluation**

**Before vs. After Summary:**
- OASIS: 436 raw scans → 205 usable subjects (CDR 0 vs 0.5)
- ADNI: 1,825 raw scans → 629 baseline subjects → 503 train / 126 test

**Feature Sets:**
- Level-1 (Early Detection): MRI (512) + Age + Sex/Education → **514 features**
- Level-2 (Upper-Bound): MRI (512) + MMSE + CDR-SB + Age + APOE4 + Education → **517 features**

**Leakage Prevention:**
- ✅ No temporal leakage (baseline-only)
- ✅ No subject leakage (subject-wise splits)
- ✅ No label leakage (MMSE/CDR-SB excluded from Level-1)
- ✅ No data snooping (test set touched once)
- ✅ No distribution leakage (scaler fit on train only)

This documentation is ready for inclusion in:
- **Thesis Methods Section:** Copy Sections 2–6 verbatim
- **Paper Supplementary Material:** Use Tables 1–3 and Section 5
- **Code Documentation:** Reference line numbers for reproducibility

---

**Document Version:** 1.0  
**Last Updated:** December 24, 2025  
**Authors:** Research Team  
**Approval Status:** Ready for Publication
