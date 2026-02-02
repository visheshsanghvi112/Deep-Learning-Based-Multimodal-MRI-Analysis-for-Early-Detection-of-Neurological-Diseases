# 🔧 Implementation Pipeline - The Complete Research Journey

**Purpose:** Step-by-step record of WHAT we did, in WHAT ORDER, and WHY we made each decision.

**Last Updated:** January 30, 2026

---

## 📊 Technical Specifications & Implementation Details

### 1. Data Sources

**OASIS-1 (Cross-Sectional)**
- N = 205 subjects (CDR 0/0.5 filtered from 436 total)
- Single-site: Washington University (Siemens 1.5T)
- Format: ANALYZE (.hdr/.img), 176×208×176 voxels, 1mm³ isotropic
- Task: Binary classification CDR=0 vs CDR=0.5
- Label distribution: 138 normal / 67 very mild dementia (2.06:1 imbalance)

**ADNI-1 (Baseline Scans)**
- N = 629 subjects (de-duplicated from 1,825 longitudinal scans)
- Multi-site: 57 sites, heterogeneous scanners (GE/Siemens/Philips 1.5T)
- Format: NIfTI (.nii), variable resolution (0.94-1.25mm³)
- Task: Binary classification CN vs (MCI+AD)
- Label distribution: 194 CN / 435 MCI+AD (2.24:1 imbalance)

**ADNI Longitudinal Subset**
- N = 639 subjects, 2,262 total scans
- Progression task: MCI stable vs MCI→AD conversion (2-year window)
- Label: 226 converters / 403 stable

---

### 2. Feature Engineering

**MRI Features (512-dimensional)**
- Architecture: ResNet18 (ImageNet pretrained)
- Extraction: 2.5D approach (9 slices: 3 axial + 3 coronal + 3 sagittal)
- Slice indices: Center ± 20 voxels along each axis
- Preprocessing: Resize to 224×224, min-max normalization
- Aggregation: Mean pooling across slices
- Output: 512-dim embedding per subject

**Clinical Feature Levels**

| Level | Feature Set | Dimensionality | Rationale |
|-------|-------------|----------------|-----------|
| **Level-1** | Age, Sex | 2D | Minimal honest baseline (always available) |
| **Level-MAX** | Age, Sex, Educ, APOE4, 7 volumes (Hippo, Vent, Ent, Fusi, MidT, WB, ICV), 3 CSF (Aβ42, Tau, pTau) | 14D | Biological markers, no circular features |
| **Level-2** | Level-1 + MMSE, CDR-SB | 4D | Circular reference (diagnostic scores) |

**Longitudinal Features (21-dimensional)**
- 6 baseline volumes (bl_hippocampus, bl_ventricles, bl_entorhinal, bl_midtemp, bl_fusiform, bl_wholebrain)
- 6 follow-up volumes (fu_*)
- 6 delta features (fu - bl)
- 3 demographics (age, sex, APOE4)
- Key feature: Hippocampal atrophy rate = Δvolume / Δtime

---

### 3. Model Architectures

**MRI-Only Baseline**
```
Input: 512-dim MRI features
Layers: 512 → FC(256) → ReLU → Dropout(0.5) → FC(128) → ReLU → Dropout(0.5) → FC(2)
Loss: Binary cross-entropy
Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
```

**Late Fusion**
```
MRI branch:     512 → FC(256) → FC(128) → FC(2)  [logits_mri]
Clinical branch:  N → FC(32)  → FC(16)  → FC(2)  [logits_clin]
Fusion: logits_final = (logits_mri + logits_clin) / 2
```

**Attention-Gated Fusion**
```
MRI branch:     512 → FC(256) → FC(128)  [h_mri]
Clinical branch:  N → FC(32)  → FC(16)   [h_clin]
Attention: α_mri = sigmoid(W_mri @ h_mri), α_clin = sigmoid(W_clin @ h_clin)
Fusion: h = α_mri * h_mri + α_clin * h_clin → FC(64) → FC(2)
```

**Longitudinal Model**
```
Algorithm: Random Forest (100 trees, max_depth=10)
Input: 21 longitudinal features
Target: Binary (converter vs stable)
Validation: 5-fold stratified CV
```

---

### 4. Experimental Results Summary

| Experiment | Dataset | Model | Features | AUC | Key Finding |
|------------|---------|-------|----------|-----|-------------|
| OASIS In-Dataset | OASIS (N=205) | Late Fusion | MRI + 5 clinical | 0.794±0.083 | Fusion baseline established |
| ADNI Level-1 | ADNI (N=629) | Late Fusion | MRI + Age/Sex | 0.598 | Weak features fail |
| ADNI Level-MAX | ADNI (N=629) | Late Fusion | MRI + 14 biomarkers | **0.808** | **+21% gain from features** |
| ADNI Level-2 | ADNI (N=629) | Late Fusion | MRI + MMSE/CDR | 0.988 | Circular ceiling |
| Cross: OASIS→ADNI | OASIS→ADNI | MRI-Only | MRI + Age/Sex | 0.607 | MRI-only most robust |
| Cross: ADNI→OASIS | ADNI→OASIS | Late Fusion | MRI + Age/Sex | 0.624 | Fusion helps on clean target |
| Longitudinal (CNN) | ADNI Long (N=639) | LSTM | ResNet sequences | 0.441 | Generic features fail |
| Longitudinal (Bio) | ADNI Long (N=639) | Random Forest | 21 volume features | **0.848** | **Atrophy rate wins** |

**Primary Finding:**
- Feature content gap (Level-1 → Level-MAX): **+21.0% AUC**
- Architecture gap (MRI-only → Attention): **<3% AUC**
- Conclusion: Feature engineering >> model complexity (within tested families)

---

### 5. Computational Details

**Hardware:**
- GPU: NVIDIA RTX 3060 (12GB VRAM) or CPU fallback
- RAM: 16GB minimum for batch processing
- Storage: 50GB for extracted features + models

**Processing Time:**
- MRI feature extraction: 3-7 sec/subject (GPU: 3s, CPU: 7s)
- Full ADNI extraction (629 subjects): ~30-45 minutes
- Model training (single fold): 5-10 minutes
- 5-fold CV: 30-50 minutes

**Software Stack:**
- Python 3.8+
- PyTorch 1.10+ (ResNet18, custom fusion models)
- scikit-learn 1.0+ (Random Forest, metrics)
- nibabel 3.2+ (ANALYZE/NIfTI loading)
- pandas 1.3+ (ADNIMERGE processing)

---

### 6. Implementation Challenges & Solutions

**Challenge 1: Label Shift (OASIS vs ADNI)**
- Problem: CDR 0/0.5 vs CN/(MCI+AD) are different diagnostic criteria
- Solution: Preserved native labels; cross-dataset results interpreted with label-shift caveat
- Impact: Transfer AUC drops reflect both domain shift AND label mismatch

**Challenge 2: Missing Biomarkers**
- Problem: ADNIMERGE has 35% missing CSF, 18% missing volumes
- Solution: Median imputation on training set; subjects with >50% missing excluded
- Impact: Level-MAX N=518 usable (from 629 baseline)

**Challenge 3: ResNet Scale Invariance**
- Problem: ImageNet features ignore absolute size → can't detect atrophy
- Solution: Switched to explicit volumetric measurements (mm³) for longitudinal
- Impact: Longitudinal AUC: 0.44 (CNN) → 0.83 (volumes)

**Challenge 4: Class Imbalance**
- Problem: ADNI 69% positive class
- Solution: Stratified splits + AUC as primary metric (threshold-independent)
- Impact: Accuracy misleading; all comparisons use AUC

**Challenge 5: Leakage Prevention**
- Problem: ADNI has multiple scans per subject (temporal leakage risk)
- Solution: Subject-level splits; baseline-only for cross-sectional experiments
- Impact: 1,825 scans → 629 subjects (-65% data to maintain integrity)

---

### 7. Code Structure

```
project/
├── scripts/
│   ├── mri_feature_extraction.py     # ResNet18 2.5D pipeline
│   └── train_multimodal.py           # OASIS experiments
│
project_adni/
├── src/
│   ├── train_level1.py               # ADNI Level-1 (Age/Sex)
│   ├── train_level_max.py            # ADNI Level-MAX (14 biomarkers)
│   ├── train_level2.py               # Circular reference
│   └── cross_dataset_robustness.py   # OASIS ↔ ADNI transfer
│
project_longitudinal_fusion/
├── scripts/
│   ├── 01_data_preparation.py        # Extract progression labels
│   ├── 03_train_lstm.py              # Phase 1: CNN sequences
│   └── 06_full_cohort_analysis.py    # Phase 2: Volumetric features
```

---

### 8. Key Takeaways for Researchers

**What Worked:**
- Biological biomarkers (Level-MAX) achieve honest 0.81 AUC
- Hippocampal atrophy rate is single strongest progression predictor
- Simple late fusion competitive with complex attention mechanisms
- Smaller homogeneous data (OASIS) generalizes better than larger noisy data (ADNI) in some transfer directions

**What Failed:**
- Demographics alone (Age/Sex) = near-random performance (0.60 AUC)
- Generic CNN features useless for temporal modeling (0.44 AUC)
- Attention fusion: higher variance, worse cross-dataset robustness
- Cross-dataset transfer degrades 15-30% across all architectures

**Methodological Contributions:**
- Three-tier evaluation protocol (Level-1/MAX/2) quantifies circular vs honest performance
- Controlled comparison proving feature content > architecture (within tested families)
- Robustness asymmetry: best model varies by transfer direction
- Negative results (longitudinal CNN failure) inform future feature design

---

## 🎯 The Research Journey (True Chronological Order)

```
START: I have OASIS MRI data, let's try multimodal fusion
  │
  ▼
PHASE 1: OASIS Experiments ──────────────────────────── 0.794 AUC ✅
  │       "Does fusion even work?"
  │
  ▼
PHASE 2: ADNI Level-1 ───────────────────────────────── 0.598 AUC ⚠️
  │       "Let's try a bigger, better dataset"
  │       (But we only used Age + Sex → terrible!)
  │
  ▼
PHASE 3: Cross-Dataset Transfer ─────────────────────── 0.55-0.62 AUC ⚠️
  │       "Do models trained on one dataset work on another?"
  │       (They collapsed! Overfitting to dataset quirks)
  │
  ▼
PHASE 4: ADNI Level-2 (Circular) ────────────────────── 0.988 AUC ❌
  │       "Is the model broken, or are our features weak?"
  │       (Model works! It's the features that suck)
  │
  ▼
PHASE 5: ADNI Level-MAX (Biomarkers) ────────────────── 0.808 AUC ✅
  │       "Let's use REAL biological features"
  │       (BREAKTHROUGH! Features > Architecture)
  │
  ▼
PHASE 6: Longitudinal with CNN ──────────────────────── 0.441 AUC ❌
  │       "Can we predict MCI→Dementia conversion using ResNet over time?"
  │       (FAILED! ResNet can't see atrophy)
  │
  ▼
PHASE 7: Longitudinal with Biomarkers ───────────────── 0.848 AUC ✅
  │       "What if we use volumetric changes instead of CNN?"
  │       (BEST RESULT! Hippocampal atrophy rate wins)
  │
  ▼
END: Primary Finding = Feature Content (+21%) >> Architecture (<3%)
```

---

## 📋 Master Experiment List

| Phase | Experiment | Why We Did It | AUC | Outcome |
|-------|------------|---------------|-----|---------|
| 1 | OASIS In-Dataset | Starting point - test if fusion works | 0.794 | ✅ Baseline |
| 2 | ADNI Level-1 | Bigger dataset, but minimal features | 0.598 | ⚠️ Weak |
| 3a | Cross: OASIS→ADNI | Test generalization | 0.607 | ⚠️ Collapse |
| 3b | Cross: ADNI→OASIS | Test reverse direction | 0.624 | ⚠️ Collapse |
| 4 | ADNI Level-2 (Circular) | Debug: is model or features the problem? | 0.988 | ❌ Cheating |
| 5 | ADNI Level-MAX | Use real biomarkers honestly | 0.808 | ✅ Works! |
| 6 | Longitudinal CNN | Track disease over time with ResNet | 0.441 | ❌ Failed |
| 7 | Longitudinal Biomarkers | Track volumetric changes | 0.848 | ✅ **BEST** |

---

## 🔬 PHASE 1: OASIS Cross-Sectional (WHERE IT ALL STARTED)

### Why We Started Here
```
- I had access to OASIS-1 data (publicly available, no approval needed)
- 205 subjects with MRI scans + clinical info
- Goal: Can we combine MRI + clinical data to detect dementia?
- This was the PROOF OF CONCEPT
```

### What We Did

**Step 1: Load MRI Scans**
```
Source: disc1-disc12 folders
Format: .hdr/.img files (Analyze format)
Location: PROCESSED/MPRAGE/T88_111/
Tool: nibabel library
Output: 3D numpy array (176 × 208 × 176 voxels)
```

**Step 2: Extract MRI Slices (2.5D Approach)**
```
Why 2.5D? 
  - 3D CNN needs too much memory and data
  - Pure 2D loses spatial context
  - 2.5D = sweet spot

How:
  - Take 3 slices from each axis (axial, coronal, sagittal)
  - Slices at center and center ± 20
  - Total: 9 slices per brain
  - Resize to 224×224 for ResNet
```

**Step 3: Extract Deep Features with ResNet18**
```
Why ResNet18?
  - Pre-trained on ImageNet (1.2M images)
  - Transfer learning works for medical imaging
  - Lightweight (11M params) - won't overfit on 205 subjects

How:
  - Remove final classification layer
  - Pass each slice through ResNet
  - Get 512-dim vector per slice
  - Average across 9 slices → 512 numbers per subject
```

**Step 4: Get Clinical Features**
```
From subject .txt files:
  - AGE (years)
  - nWBV (normalized brain volume)
  - eTIV (total intracranial volume)
  - ASF (atlas scaling factor)
  - EDUC (years of education)

NOT used (circular):
  - MMSE (directly measures cognition = cheating)
```

**Step 5: Create Labels**
```
From CDR (Clinical Dementia Rating):
  - CDR = 0 → Healthy (Label = 0)
  - CDR ≥ 0.5 → Dementia (Label = 1)
```

**Step 6: Train Fusion Model**
```
Architecture: Late Fusion
  - MRI branch: 512 → 64
  - Clinical branch: 5 → 32
  - Concatenate → 96 → 64 → 2 (output)
 
Training:
  - 80/20 subject-wise split
  - Adam optimizer, lr=0.001
  - Early stopping at patience=15
```

### Result
```
AUC: 0.794 (Late Fusion)
     0.770 (MRI-Only)
     0.790 (Attention Fusion)

Conclusion: FUSION WORKS! +2.4% gain over MRI-only.
Next step: Try on bigger, richer dataset (ADNI)
```

---

## 🔬 PHASE 2: ADNI Level-1 (THE DISAPPOINTMENT)

### Why We Moved to ADNI
```
- OASIS worked, but only 205 subjects
- ADNI has 629 subjects + rich biomarkers
- ADNI is the "gold standard" for Alzheimer's research
- Goal: Replicate success on bigger dataset
```

### What We Did
```
Same pipeline as OASIS:
  1. Extract MRI features with ResNet18 (512-dim)
  2. Get clinical features from ADNIMERGE.csv
  
BUT: We only used Age + Sex (2 features)
     Why? These are the only "neutral" features
     Everything else in ADNI is either:
       - Circular (MMSE, CDR) 
       - Requires special tests (CSF biomarkers)
```

### Result
```
AUC: 0.598 (Late Fusion)
     0.583 (MRI-Only)

DISASTER! Barely better than random (0.50)
```

### Why It Failed
```
Age and Sex alone can't predict Alzheimer's!
  - Old people can be healthy
  - Young people can have early-onset AD
  - Sex has weak correlation with AD risk

The MRI features were the same quality as OASIS.
The CLINICAL features were garbage.
```

### What This Taught Us
```
FUSION IS ONLY AS GOOD AS ITS INPUTS!
If clinical features are weak, fusion adds nothing.
```

---

## 🔬 PHASE 3: Cross-Dataset Transfer (THE REALITY CHECK)

### Why We Did This
```
After OASIS worked and ADNI failed, we asked:
"Do these models even generalize to new data?"
"Or are they just memorizing dataset-specific patterns?"
```

### Experiment A: OASIS → ADNI
```
Train on: OASIS (205 subjects)
Test on: ADNI (629 subjects)
No fine-tuning - pure zero-shot transfer

Features used (intersection of both datasets):
  - MRI (512 ResNet features)
  - Age
  - Education
```

**Results:**
```
| Model            | OASIS AUC | ADNI AUC | Drop   |
|------------------|-----------|----------|--------|
| MRI-Only         | 0.814     | 0.607    | -0.207 |
| Late Fusion      | 0.864     | 0.575    | -0.289 |
| Attention Fusion | 0.826     | 0.557    | -0.269 |

SHOCKING: Fancier models transferred WORSE!
MRI-Only was the most robust.
Fusion models overfit to OASIS-specific patterns.
```

### Experiment B: ADNI → OASIS
```
Train on: ADNI (503 subjects)
Test on: OASIS (205 subjects)
```

**Results:**
```
| Model            | ADNI AUC | OASIS AUC | Drop   |
|------------------|----------|-----------|--------|
| MRI-Only         | 0.686    | 0.569     | -0.117 |
| Late Fusion      | 0.702    | 0.624     | -0.078 |
| Attention Fusion | 0.657    | 0.548     | -0.109 |

Different direction, different winner!
Late Fusion was best here (opposite of A).
```

### What This Taught Us
```
1. Cross-dataset transfer is HARD
2. No single model wins in both directions
3. Fusion models can overfit MORE than simple models
4. We need better features, not better models
```

---

## 🔬 PHASE 4: ADNI Level-2 - The Debugging Experiment (INTENTIONAL CHEATING)

### Why We Did This
```
After Level-1 failed (0.60 AUC), we had to ask:
"Is the MODEL broken? Or are the FEATURES weak?"

To answer this, we INTENTIONALLY added circular features:
  - MMSE (Mini-Mental State Exam)
  - CDR-SB (Clinical Dementia Rating)

These features DIRECTLY MEASURE cognitive impairment.
Using them to predict dementia is like using "is wet" to predict rain.
```

### Result
```
AUC: 0.988 (almost perfect!)

This PROVED:
  - The model architecture WORKS
  - The training pipeline is CORRECT
  - The problem in Level-1 was FEATURE QUALITY, not the model
```

### Why This Matters
```
Most published papers use MMSE/CDR and claim high AUC.
They're cheating without realizing it.
Our honest Level-1 (0.60) is the TRUE baseline.
Our Level-2 (0.99) shows what happens when you cheat.
```

---

## 🔬 PHASE 5: ADNI Level-MAX (THE BREAKTHROUGH)

### Why We Did This
```
Level-1 failed because Age+Sex are weak.
Level-2 succeeded because MMSE+CDR are circular (cheating).

Question: What if we use REAL biological features?
Features that are:
  - Clinically meaningful
  - NOT circular
  - Actually measure disease biology
```

### What Features We Used (14 total)
```
Demographics (3):
  1. Age
  2. Sex  
  3. Education (PTEDUCAT)

Genetics (1):
  4. APOE4 (0, 1, or 2 risk alleles)

Brain Volumes (7):
  5. Hippocampus (first region to shrink in AD!)
  6. Ventricles (expand as brain shrinks)
  7. Entorhinal cortex
  8. Fusiform gyrus
  9. Middle temporal
  10. Whole brain volume
  11. Intracranial volume (ICV)

CSF Biomarkers (3):
  12. ABETA (amyloid-beta 42)
  13. TAU
  14. PTAU (phosphorylated tau)
```

### Why These Features Are HONEST
```
- Hippocampus volume: Measurable BEFORE symptoms appear
- CSF proteins: Direct biological markers of disease process
- APOE4: Genetic risk, not a measurement of current cognition
- NONE of these are circular!
```

### Result
```
AUC: 0.808 (Late Fusion)
     0.643 (MRI-Only)

BREAKTHROUGH!
  - Same MRI features as Level-1
  - Same model architecture as Level-1
  - ONLY difference: clinical features
  
Level-1 (Age+Sex):     0.598 AUC
Level-MAX (Biomarkers): 0.808 AUC
                        ─────────
                        +0.210 AUC difference!

This is +21% improvement from features alone.
Model architecture difference was <3%.

PRIMARY FINDING: Feature Content >> Architecture
```

---

## 🔬 PHASE 6: Longitudinal with CNN (THE FAILURE)

### Why We Tried This
```
ADNI has multiple scans per patient over time.
If someone is developing Alzheimer's, their brain changes.

Hypothesis: If we track ResNet features across visits,
we can predict who will progress from MCI to dementia.
```

### What We Did
```
1. Selected MCI patients with 2+ visits (N=~400)
2. Extracted ResNet features at each visit
3. Created sequences: [visit1_512, visit2_512, visit3_512, ...]
4. Trained LSTM to predict: Will they convert to dementia?
```

### Result
```
AUC: 0.441 (WORSE than random!)

Random guessing = 0.50
We got 0.44
The model learned to predict WRONG.
```

### Why It Failed
```
ResNet features are SCALE-INVARIANT.
They capture "what patterns exist" not "how big things are."

Brain atrophy = things getting SMALLER
ResNet sees: "This is a hippocampus" at Visit 1
ResNet sees: "This is a hippocampus" at Visit 2
ResNet CANNOT see: "This hippocampus is 15% smaller"

The LSTM had nothing useful to learn from.
```

### What This Taught Us
```
Deep learning features ≠ disease-relevant features
CNN features are great for classification
CNN features are USELESS for tracking change over time
Need EXPLICIT volumetric measurements
```

---

## 🔬 PHASE 7: Longitudinal with Biomarkers (THE BEST RESULT)

### Why We Did This
```
Phase 6 failed because ResNet can't see atrophy.
What if we use actual volume measurements from ADNIMERGE?
These are EXPLICIT numbers: "Hippocampus = 3,456 mm³"
```

### What Features We Used (21 total)
```
BASELINE VOLUMES (6):
  - bl_hippocampus, bl_ventricles, bl_entorhinal
  - bl_midtemp, bl_fusiform, bl_wholebrain

FOLLOWUP VOLUMES (6):
  - Same 6 regions at last visit

CHANGE OVER TIME (6):
  - Delta = Followup - Baseline for each region
  - Hippocampus_delta = How much it shrank

DEMOGRAPHICS (3):
  - Age, Sex, APOE4
```

### The Key Feature
```
HIPPOCAMPAL ATROPHY RATE = hippocampus_delta / time_between_visits

This captures: "How fast is the hippocampus shrinking?"
Faster shrinkage → Higher conversion risk
```

### Population
```
MCI patients only (N=341)
  - 115 Converters (progressed to dementia)
  - 226 Stable (stayed MCI)
5-fold stratified cross-validation
```

### Model
```
Random Forest (100 trees)

Why NOT deep learning?
  - Only 341 subjects (too small for NN)
  - Only 21 features (tabular data)
  - Random Forest is perfect for this
  - Also interpretable (can see feature importance)
```

### Result
```
AUC: 0.848 ± 0.025 (BEST RESULT IN ENTIRE PROJECT!)

Most important feature: HIPPOCAMPAL ATROPHY RATE
Second: Baseline hippocampus volume
Third: APOE4 genetic risk
```

### What This Taught Us
```
1. Simple model (RF) + Right features (volumes) = BEST result
2. Complex model (LSTM) + Wrong features (ResNet) = WORST result
3. Change over time is more predictive than single snapshot
4. Hippocampal atrophy rate is the #1 predictor
```

---

### Step 1: Select MCI Cohort
```
File: 06_full_cohort_analysis.py → load_mci_cohort()
Source: ADNIMERGE.csv
Filter: Subjects with MCI at baseline (first visit)
Why MCI? These are the "at-risk" group we want to predict.
Subjects: 341 MCI patients with ≥2 visits
```

### Step 2: Determine Labels (Progression)
```
File: 06_full_cohort_analysis.py → determine_progression()
Logic:
  - Look at diagnosis at LAST visit
  - If changed to "Dementia" or "AD" → Converter (Label = 1)
  - If still "MCI" or "CN" → Stable (Label = 0)
Result: 115 converters, 226 stable
```

### Step 3: Extract Longitudinal Features
```
File: 06_full_cohort_analysis.py → extract_longitudinal_biomarkers()
What we extracted (21 features):

BASELINE VOLUMES (6):
  - bl_hippocampus
  - bl_ventricles
  - bl_entorhinal
  - bl_midtemp
  - bl_fusiform
  - bl_wholebrain

DEMOGRAPHICS (3):
  - age
  - sex (0/1)
  - apoe4 (0/1/2)

FOLLOWUP VOLUMES (6):
  - fu_hippocampus
  - fu_ventricles
  - fu_entorhinal
  - fu_midtemp
  - fu_fusiform
  - fu_wholebrain

CHANGE OVER TIME (6):
  - hippocampus_delta (followup - baseline)
  - ventricles_delta
  - entorhinal_delta
  - midtemp_delta
  - fusiform_delta
  - wholebrain_delta

Key feature: Hippocampal ATROPHY RATE (delta / time)
```

### Step 4: NO MRI CNN Features!
```
This experiment used ONLY tabular biomarkers.
No ResNet18, no deep learning for image features.
Why? Testing if clinical data alone can predict conversion.
```

### Step 5: Train Model
```
File: 06_full_cohort_analysis.py
Model: Random Forest (100 trees)
Why Random Forest?
  - Works great with small tabular data (341 subjects)
  - Deep learning needs thousands of samples
  - Interpretable (can see feature importance)
Cross-validation: 5-fold stratified
```

### Step 6: Result
```
AUC: 0.848 ± 0.025 (BEST RESULT!)
Most important feature: Hippocampal atrophy rate
Why it worked: Change over time is more predictive than single snapshots.
```

---

## 📊 Feature Summary Table

| Phase | Features Used | Why These Features | AUC |
|-------|---------------|-------------------|-----|
| 1 OASIS | MRI(512) + Age,nWBV,eTIV,ASF,Educ(5) | What was available | 0.794 |
| 2 Level-1 | MRI(512) + Age,Sex(2) | Minimal honest baseline | 0.598 |
| 3 Cross | MRI(512) + Age,Educ(2) | Intersection of both datasets | 0.55-0.62 |
| 4 Level-2 | MRI(512) + Age,Sex,MMSE,CDR(4) | Intentionally circular (debugging) | 0.988 |
| 5 Level-MAX | MRI(512) + 14 biomarkers | Real biology, no cheating | 0.808 |
| 6 Long-CNN | MRI(512) × N visits | Trying temporal deep learning | 0.441 |
| 7 Long-Bio | 21 volumetric features | Explicit atrophy measurements | **0.848** |

---

## 🗺️ The Decision Tree (Why Each Step Led to Next)

```
Q: Does multimodal fusion work?
├─ OASIS (0.79) → YES! Fusion beats MRI-only
│
├─ Q: Does it work on a bigger dataset?
│  ├─ ADNI Level-1 (0.60) → NO! Age+Sex are useless
│  │
│  ├─ Q: Is the model broken?
│  │  ├─ Level-2 (0.99) → NO! Model works with good features
│  │  │
│  │  ├─ Q: Can we find honest features that work?
│  │  │  └─ Level-MAX (0.81) → YES! Biomarkers work honestly
│  │  │
│  │  └─ Q: Do models generalize across datasets?
│  │     └─ Cross-dataset (0.55-0.62) → NO! Heavy overfitting
│  │
│  └─ Q: Can we track disease progression over time?
│     ├─ Longitudinal CNN (0.44) → NO! ResNet can't see atrophy
│     │
│     └─ Longitudinal Biomarkers (0.85) → YES! Volumetric deltas work!
│
└─ FINAL ANSWER: Feature content matters more than model architecture
```

---

## 🔑 The Key Lessons (In Order We Learned Them)

### Lesson 1: Fusion works... sometimes (Phase 1)
```
OASIS showed fusion can beat MRI-only.
But we had good clinical features (brain volumes, education).
```

### Lesson 2: Weak features = weak results (Phase 2)
```
ADNI Level-1 crashed to 0.60 with just Age+Sex.
Fusion doesn't magically create signal from nothing.
```

### Lesson 3: Circular features are cheating (Phase 4)
```
Adding MMSE got 0.99 AUC - but it's meaningless.
Most papers in literature do this without admitting it.
```

### Lesson 4: Real biomarkers work honestly (Phase 5)
```
Level-MAX proved you CAN get good results (0.81) without cheating.
Hippocampus volume and CSF proteins are the key.
```

### Lesson 5: Models overfit to dataset quirks (Phase 3)
```
Cross-dataset transfer collapsed by 15-30%.
Fancier models transferred WORSE than simple ones.
```

### Lesson 6: CNN features can't track change (Phase 6)
```
ResNet sees patterns, not sizes.
LSTM couldn't learn from scale-invariant features.
```

### Lesson 7: Volumetric deltas are the answer (Phase 7)
```
Explicit "how much did hippocampus shrink?" measurements work.
Simple Random Forest + right features = BEST result (0.848).
```

### THE MASTER LESSON
```
Feature Content (+21% AUC) >> Model Architecture (<3% AUC)

Same model + better features = massive improvement
Better model + same features = tiny improvement

FEATURES MATTER MORE THAN ARCHITECTURE
```

---

## 🛠️ Code Files (Where Each Phase Lives)

| Phase | Main Script | Location |
|-------|-------------|----------|
| 1 OASIS | `mri_feature_extraction.py`, `train_multimodal.py` | `project/scripts/` |
| 2 Level-1 | `train_level1.py` | `project_adni/src/` |
| 3 Cross | `cross_dataset_robustness.py` | `project_adni/src/` |
| 4 Level-2 | `train_level2.py` | `project_adni/src/` |
| 5 Level-MAX | `train_level_max.py` | `project_adni/src/` |
| 6 Long-CNN | Phase 1 scripts | `project_longitudinal_fusion/src/` |
| 7 Long-Bio | `06_full_cohort_analysis.py` | `project_longitudinal_fusion/scripts/` |

---

## 📁 Data Files

| File | Location | What's In It |
|------|----------|--------------|
| `oasis_all_features.npz` | `extracted_features/` | OASIS MRI(512) + Clinical(6) + Labels |
| `adni_baseline_features.npz` | `data/ADNI/extracted_features/` | ADNI MRI(512) + Subject IDs |
| `train_level1.csv` | `project_adni/data/features/` | Level-1 train set |
| `train_level_max.csv` | `project_adni/data/features/` | Level-MAX train set |
| `ADNIMERGE_23Dec2025.csv` | `data/ADNI/` | Master clinical data (all biomarkers) |

---

## 🎯 TL;DR - The Entire Journey in 60 Seconds

```
1. OASIS (0.79) → "Fusion works! Let's try bigger data"

2. ADNI Level-1 (0.60) → "WTF? This is terrible!"
   
3. Cross-dataset (0.55-0.62) → "Models don't generalize either!"

4. Level-2 circular (0.99) → "OK the model works, features are the problem"

5. Level-MAX biomarkers (0.81) → "BREAKTHROUGH! Real biomarkers work honestly"

6. Longitudinal CNN (0.44) → "DISASTER! ResNet can't track atrophy"

7. Longitudinal biomarkers (0.85) → "BEST RESULT! Hippocampal atrophy rate wins"

═══════════════════════════════════════════════════════════════
DISCOVERY: Feature content matters 10× more than model complexity
           +21% AUC from better features
           <3% AUC from better architecture
═══════════════════════════════════════════════════════════════
```

---

**End of Pipeline Documentation**

*This is the story of how we discovered that features beat architecture.*
