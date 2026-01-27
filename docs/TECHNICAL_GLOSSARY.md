# 🧠 Technical Glossary & Explanation Guide

**Purpose:** This document explains every technical term, concept, and technology used in the Deep Learning-Based Multimodal MRI Analysis project. Written in simple language to help you understand and present the work confidently.

**Last Updated:** January 27, 2026

---

## 📋 Table of Contents

1. [Project Flow (Simple Overview)](#project-flow-simple-overview)
2. [Performance Metrics](#performance-metrics)
3. [Deep Learning Concepts](#deep-learning-concepts)
4. [Model Architectures](#model-architectures)
5. [Data & Preprocessing](#data--preprocessing)
6. [Medical & Clinical Terms](#medical--clinical-terms)
7. [Technology Stack](#technology-stack)
8. [Design Decisions (Why We Used What)](#design-decisions-why-we-used-what)

---

## 🎯 Project Flow (Simple Overview)

### What We Did (In 5 Steps):

1. **Got MRI Brain Scans** from two public datasets (OASIS-1 and ADNI-1)
2. **Extracted Features** from the scans using a pre-trained deep learning model (ResNet18)
3. **Combined** MRI features with clinical data (age, sex, biomarkers)
4. **Trained Models** to predict who will develop Alzheimer's disease
5. **Evaluated** how well the models work using metrics like AUC

### The Journey:

```
Raw MRI Scans (3D Brain Images)
    ↓
ResNet18 Feature Extraction (converts images to 512 numbers)
    ↓
Combine with Clinical Data (age, sex, CSF biomarkers)
    ↓
Train Fusion Models (Late Fusion, Attention Fusion)
    ↓
Evaluate Performance (AUC, Accuracy, Sensitivity)
    ↓
Results: 0.848 AUC (Longitudinal) | 0.808 AUC (Level-MAX)
```

---

## 📊 Performance Metrics

### AUC (Area Under the Curve)

**What it is:**
- A number between 0.0 and 1.0 that tells you how good your model is at classification
- Specifically, it measures the area under the ROC curve

**What it means:**
- **AUC = 1.0:** Perfect model - correctly classifies everything
- **AUC = 0.5:** Random guessing - the model is useless
- **AUC = 0.848:** Our best result - very good performance!

**Where we used it:**
- Primary metric for ALL experiments (OASIS, ADNI, Longitudinal)
- Reported in every results table

**Why we chose it:**
- Standard metric in medical AI research
- Works well with imbalanced datasets (we have more healthy than sick people)
- Easy to compare with other published papers

**Alternatives we could have used:**
- Accuracy (but misleading with imbalanced data)
- F1-Score (good, but AUC is more standard in medical imaging)

---

### ROC Curve (Receiver Operating Characteristic)

**What it is:**
- A graph that shows the trade-off between True Positive Rate and False Positive Rate
- X-axis: False Positive Rate (how many healthy people we wrongly call sick)
- Y-axis: True Positive Rate (how many sick people we correctly identify)

**What it means:**
- The curve shows performance at different decision thresholds
- A curve closer to the top-left corner is better
- The area under this curve is the AUC

**Where we used it:**
- Generated ROC curves for all experiments
- Visualized in `figures/` folder (e.g., `A1_oasis_model_comparison.pdf`)

**Why we chose it:**
- Visual way to show model performance
- Shows the full picture, not just one threshold
- Standard in medical diagnostics

---

### Sensitivity (True Positive Rate / Recall)

**What it is:**
- Percentage of sick people that the model correctly identifies
- Formula: `Sensitivity = True Positives / (True Positives + False Negatives)`

**What it means:**
- High sensitivity = we catch most of the sick people
- Low sensitivity = we miss many sick people (dangerous in healthcare!)

**Where we used it:**
- Reported in confusion matrices
- Important for medical applications (we don't want to miss patients)

**Example:**
- If 100 people have Alzheimer's and we detect 85 of them: Sensitivity = 85%

---

### Specificity (True Negative Rate)

**What it is:**
- Percentage of healthy people that the model correctly identifies as healthy
- Formula: `Specificity = True Negatives / (True Negatives + False Positives)`

**What it means:**
- High specificity = we don't falsely alarm healthy people
- Low specificity = we wrongly scare many healthy people

**Where we used it:**
- Paired with sensitivity in evaluation
- Trade-off: increasing sensitivity often decreases specificity

---

### Confidence Interval (95% CI)

**What it is:**
- A range that tells you how uncertain your result is
- Example: AUC = 0.848 ± 0.025 means the true value is likely between 0.823 and 0.873

**What it means:**
- Narrow interval = we're confident in the result
- Wide interval = result is uncertain, need more data

**Where we used it:**
- Calculated using bootstrap method (see below)
- Reported for all AUC values

**Why we chose it:**
- Shows statistical rigor
- Helps determine if improvements are real or just luck

---

### Bootstrap

**What it is:**
- A statistical method to estimate confidence intervals
- We randomly resample our data 1,000 times and recalculate AUC each time

**How it works:**
1. Take original test set (e.g., 126 subjects)
2. Randomly sample 126 subjects WITH replacement (some appear multiple times)
3. Calculate AUC on this resampled set
4. Repeat 1,000 times
5. The range of AUCs gives us the confidence interval

**Where we used it:**
- File: `train_level1.py`, `train_level_max.py` (function: `bootstrap_ci()`)
- Used for all AUC confidence intervals

**Why we chose it:**
- Works with small datasets
- Doesn't assume normal distribution
- Standard method in machine learning

---

### Cross-Validation (K-Fold)

**What it is:**
- A method to test your model on different subsets of data
- We split data into K parts (folds), train on K-1 parts, test on the remaining part
- Repeat K times so each part gets to be the test set once

**What it means:**
- More reliable than a single train/test split
- Reduces the risk of getting lucky with one particular split

**Where we used it:**
- **5-fold cross-validation** in longitudinal experiments
- **Stratified** version (see below) to keep class balance

**Why we chose it:**
- Standard practice in machine learning
- Makes best use of limited data
- Gives more robust performance estimates

---

### Stratification

**What it is:**
- Ensuring each fold in cross-validation has the same proportion of classes
- Example: If 33% of subjects convert to dementia, each fold should have ~33% converters

**Why it matters:**
- Prevents one fold from being too easy or too hard
- Ensures fair comparison across folds

**Where we used it:**
- Longitudinal fusion experiments (`StratifiedKFold` in scikit-learn)
- File: `project_longitudinal_fusion/src/training/cross_validation.py`

---

## 🧠 Deep Learning Concepts

### ResNet18 (Residual Network with 18 layers)

**What it is:**
- A deep neural network architecture with 18 layers
- "Residual" means it uses skip connections (shortcuts that help training)

**What it does:**
- Takes an image as input
- Outputs a 512-dimensional feature vector (512 numbers that represent the image)

**Where we used it:**
- **MRI feature extraction** - converted 3D brain scans into 512 numbers
- Files: `mri_feature_extraction.py`, `adni_feature_extraction.py`

**Why we chose it:**
- Pre-trained on ImageNet (1.2 million natural images)
- Proven to work well for medical imaging
- Fast and efficient (doesn't need huge GPU)

**Alternatives we could have used:**
- ResNet50 (deeper, but slower and needs more data)
- VGG16 (older architecture, less efficient)
- Custom CNN (would need to train from scratch - requires massive data)

---

### 2.5D Processing

**What it is:**
- A compromise between 2D and 3D processing
- We extract multiple 2D slices from the 3D MRI scan (axial, coronal, sagittal views)
- Process each slice with 2D ResNet
- Average the results

**Why we used it:**
- 3D CNNs need too much memory and data
- Pure 2D loses spatial context
- 2.5D is the sweet spot: efficient + captures 3D information

**Where we used it:**
- Feature extraction pipeline
- Extracted slices from middle of brain (most informative region)

---

### Transfer Learning

**What it is:**
- Using a model pre-trained on one task (ImageNet classification) for a different task (brain MRI analysis)

**Why it works:**
- Early layers learn general features (edges, textures)
- These features are useful across different image types

**Where we used it:**
- ResNet18 pre-trained on ImageNet → fine-tuned for brain MRIs

**Why we chose it:**
- We don't have millions of brain scans to train from scratch
- Transfer learning works well with small medical datasets
- Standard practice in medical imaging

---

### Feature Vector (Embedding)

**What it is:**
- A list of numbers that represents an image
- Our ResNet18 outputs 512 numbers per MRI scan

**What it means:**
- Each number captures some aspect of the brain structure
- Similar brains have similar feature vectors
- The model learns which features are important for Alzheimer's prediction

**Where we used it:**
- Saved as `.npz` files (compressed NumPy arrays)
- Files: `oasis_all_features.npz`, `adni_baseline_features.npz`

---

## 🏗️ Model Architectures

### MRI-Only Model

**What it is:**
- A simple baseline that uses ONLY MRI features (512 numbers from ResNet)
- No clinical data (age, sex, etc.)

**Architecture:**
```
Input: 512 MRI features
  ↓
Dense Layer (512 → 32)
  ↓
ReLU Activation
  ↓
Dropout (50%)
  ↓
Dense Layer (32 → 16)
  ↓
ReLU Activation
  ↓
Dropout (50%)
  ↓
Output: 2 classes (Healthy vs Alzheimer's)
```

**Where we used it:**
- Baseline for all experiments
- OASIS: 0.770 AUC
- ADNI Level-1: 0.583 AUC

**Why we used it:**
- Shows what MRI alone can achieve
- Comparison point for fusion models

---

### Late Fusion Model

**What it is:**
- Combines MRI features and clinical features by concatenating them
- "Late" because we combine features AFTER extracting them separately

**Architecture:**
```
MRI Features (512) ──┐
                     ├─→ Concatenate (514) → MLP → Output
Clinical (2) ────────┘
```

**Where we used it:**
- Main fusion approach
- Best OASIS result: 0.794 AUC
- Best ADNI Level-MAX: 0.808 AUC

**Why we chose it:**
- Simple and interpretable
- Standard fusion method
- Works well when both modalities are informative

**Alternatives:**
- Early fusion (combine raw data - doesn't work for different modalities)
- Intermediate fusion (complex, needs more data)

---

### Attention-Gated Fusion

**What it is:**
- A smarter fusion that learns HOW MUCH to trust each modality
- Uses a "gate" (learned weights) to decide MRI vs clinical importance

**Architecture:**
```
MRI Encoder → MRI Features ──┐
                             ├─→ Attention Gate → Weighted Fusion → Output
Clinical Encoder → Clinical ─┘
```

**Where we used it:**
- Alternative to late fusion
- ADNI Level-MAX: 0.808 AUC (same as late fusion)

**Why we tried it:**
- Theoretically better than simple concatenation
- Can adapt to different patients (some rely more on MRI, others on biomarkers)

**What we found:**
- Performed similarly to late fusion
- More complex, no clear benefit with our data size

---

### Random Forest

**What it is:**
- A traditional machine learning model (not deep learning)
- Builds many decision trees and averages their predictions

**Where we used it:**
- **Longitudinal fusion** - achieved our BEST result: 0.848 AUC
- File: `project_longitudinal_fusion/scripts/06_full_cohort_analysis.py`

**Why it won:**
- Works better with small tabular data (341 subjects, 21 features)
- Deep learning needs more data to shine
- Interpretable (can see which features matter most)

**Key finding:**
- Hippocampal atrophy rate was the most important feature!

---

### LSTM (Long Short-Term Memory)

**What it is:**
- A type of recurrent neural network designed for sequences
- Can remember information over time

**Where we tried it:**
- Longitudinal experiment (Phase 1)
- Tried to model progression as a sequence of visits

**What happened:**
- Failed miserably: 0.441 AUC (worse than random!)

**Why it failed:**
- ResNet features are scale-invariant (can't detect volume changes)
- LSTM needs meaningful temporal signals
- Our sequences were too short (2-3 visits per patient)

**Lesson learned:**
- Deep learning isn't always the answer
- Need the RIGHT features, not just fancy models

---

## 📂 Data & Preprocessing

### OASIS-1 Dataset

**What it is:**
- Open Access Series of Imaging Studies
- Cross-sectional MRI dataset (one scan per person)

**Details:**
- **436 MRI scans** from **205 unique subjects**
- Age range: 18-96 years
- Includes healthy controls and Alzheimer's patients

**Where we used it:**
- Initial experiments and validation
- Best result: 0.794 AUC (Late Fusion)

**Why we chose it:**
- Publicly available (no approval needed)
- Well-documented and cleaned
- Good for initial testing

---

### ADNI-1 Dataset

**What it is:**
- Alzheimer's Disease Neuroimaging Initiative
- Longitudinal dataset (multiple scans per person over time)

**Details:**
- **1,825 MRI scans** from **629 unique subjects**
- Includes rich biomarker data (CSF, genetics, cognition)

**Where we used it:**
- Main experiments (Level-1, Level-MAX, Longitudinal)
- Best result: 0.848 AUC (Longitudinal Fusion)

**Why we chose it:**
- Gold standard in Alzheimer's research
- Rich clinical data for fusion
- Longitudinal data for progression modeling

**Access:**
- Requires application and approval
- Free for research purposes

---

### ADNIMERGE

**What it is:**
- A master CSV file that combines all ADNI clinical data
- 12.65 MB file with thousands of rows and hundreds of columns

**What it contains:**
- Demographics (age, sex, education)
- Diagnoses (CN, MCI, AD)
- Biomarkers (CSF Aβ42, Tau, pTau)
- Genetics (APOE4 status)
- Cognition (MMSE, ADAS13, CDR-SB)
- Brain volumes (hippocampus, ventricles, etc.)

**Where we used it:**
- Source of all clinical features
- File: `data/ADNIMERGE_23Dec2025.csv`
- Parser: `project_adni/src/adnimerge_utils.py`

---

### Baseline Selection

**What it is:**
- Choosing the FIRST scan for each subject (baseline visit)
- Ignoring follow-up scans (m06, m12, m24, etc.)

**Why we did it:**
- Prevents data leakage (using future information to predict present)
- Ensures fair comparison (one scan per person)
- Standard practice in cross-sectional studies

**Where we used it:**
- OASIS: naturally cross-sectional
- ADNI: selected baseline from longitudinal data
- File: `baseline_selection.py`

---

### Data Leakage

**What it is:**
- When information from the test set "leaks" into training
- Makes results look better than they really are
- Scientific misconduct if not caught!

**Types we prevented:**

1. **Subject Leakage:**
   - Same person in both train and test sets
   - **Prevention:** Subject-wise splitting

2. **Temporal Leakage:**
   - Using future data to predict past
   - **Prevention:** Baseline-only selection

3. **Feature Leakage:**
   - Using outcome-related features (MMSE predicts dementia because it measures dementia!)
   - **Prevention:** Level-1 excludes circular features

4. **Normalization Leakage:**
   - Computing mean/std on entire dataset before splitting
   - **Prevention:** Fit scaler on train, transform test

**Verification:**
- Ran integrity audits (see `project_longitudinal_fusion/scripts/audit_integrity.py`)
- **Result:** ZERO leakage detected across all experiments

---

### Subject-Wise Splitting

**What it is:**
- Splitting data by SUBJECTS, not by scans
- Ensures no subject appears in both train and test

**Example:**
- ❌ Wrong: Subject A's baseline in train, Subject A's m12 in test
- ✅ Correct: All of Subject A's scans in train OR all in test

**Where we used it:**
- All experiments (OASIS, ADNI, Longitudinal)
- Implementation: `data_split.py`, `cross_validation.py`

**Why it matters:**
- Scans from the same person are highly correlated
- Without subject-wise splitting, the model "cheats" by recognizing individuals

---

### Normalization (Standardization)

**What it is:**
- Scaling features to have mean=0 and standard deviation=1
- Formula: `z = (x - mean) / std`

**Why we need it:**
- Features have different scales (age: 60-90, hippocampus: 3000-8000)
- Models learn better when features are on similar scales

**Where we used it:**
- Clinical features (age, biomarkers)
- NOT on MRI features (already normalized by ResNet)

**Critical detail:**
- Fit scaler on TRAIN set only
- Apply same transformation to TEST set
- Prevents normalization leakage

---

## 🏥 Medical & Clinical Terms

### MCI (Mild Cognitive Impairment)

**What it is:**
- A stage between normal aging and dementia
- Memory/thinking problems noticeable but not severe enough to interfere with daily life

**Why it matters:**
- ~30-40% of MCI patients progress to Alzheimer's within 5 years
- Our goal: predict WHO will progress

**Where we used it:**
- Main target population in ADNI experiments
- "Converters" = MCI → Dementia
- "Stable" = MCI stays MCI

---

### AD (Alzheimer's Disease)

**What it is:**
- Most common form of dementia
- Progressive brain disorder affecting memory, thinking, and behavior

**Characteristics:**
- Brain atrophy (shrinkage), especially hippocampus
- Accumulation of amyloid plaques and tau tangles
- Irreversible and fatal

**Where we used it:**
- Target disease for prediction
- Binary classification: Healthy (CN) vs Alzheimer's (AD)

---

### CN (Cognitively Normal)

**What it is:**
- Healthy controls with no cognitive impairment
- Normal memory and thinking for their age

**Where we used it:**
- Negative class in binary classification
- OASIS: CN vs AD
- ADNI: CN vs MCI/AD

---

### Hippocampus

**What it is:**
- A seahorse-shaped brain region critical for memory
- Located in the temporal lobe

**Why it matters:**
- **First region to atrophy in Alzheimer's**
- Volume loss correlates with disease severity
- **Our finding:** Hippocampus alone achieved 0.725 AUC!

**Where we used it:**
- Key feature in Level-MAX experiments
- Most important feature in Random Forest model
- Measured in mm³ from MRI scans

---

### Ventricles

**What it is:**
- Fluid-filled cavities in the brain
- Expand as brain tissue shrinks

**Why it matters:**
- Indirect marker of brain atrophy
- Larger ventricles = more tissue loss

**Where we used it:**
- Volumetric feature in Level-MAX
- Complementary to hippocampus (inverse relationship)

---

### CSF Biomarkers (Cerebrospinal Fluid)

**What they are:**
- Proteins measured in spinal fluid via lumbar puncture

**Three key biomarkers:**

1. **Aβ42 (Amyloid-beta 42):**
   - Lower in Alzheimer's (gets stuck in brain plaques)
   
2. **Tau:**
   - Higher in Alzheimer's (released from dying neurons)
   
3. **pTau (Phosphorylated Tau):**
   - Higher in Alzheimer's (specific to AD pathology)

**Where we used them:**
- Level-MAX experiments (14-feature model)
- Strong predictors (clinical features that actually work!)

**Why they matter:**
- Direct biological markers of disease
- Much stronger than demographics (age/sex)

---

### APOE4

**What it is:**
- A genetic variant of the APOE gene
- Strongest genetic risk factor for late-onset Alzheimer's

**Impact:**
- 1 copy: 3x higher risk
- 2 copies: 12x higher risk

**Where we used it:**
- Level-MAX experiments
- Longitudinal analysis

**Our finding:**
- APOE4 carriers: 44-49% conversion rate
- Non-carriers: 23% conversion rate

---

### MMSE (Mini-Mental State Examination)

**What it is:**
- A 30-point cognitive test
- Measures memory, attention, language, etc.

**Scoring:**
- 24-30: Normal
- 18-23: Mild dementia
- 0-17: Severe dementia

**Why we EXCLUDED it (Level-1):**
- **Circular reasoning:** MMSE measures cognitive impairment, which IS the outcome we're predicting
- Using MMSE to predict dementia is like using a thermometer to predict fever

**Where we INCLUDED it (Level-2):**
- "Circular" experiments to prove model architecture works
- Achieved 0.988 AUC (proves model is correct, but result is meaningless)

---

### CDR-SB (Clinical Dementia Rating - Sum of Boxes)

**What it is:**
- A clinical assessment of dementia severity
- Scores 6 domains: memory, orientation, judgment, etc.

**Why we excluded it:**
- Same reason as MMSE - circular
- Directly measures the outcome

---

## 💻 Technology Stack

### Python (3.12+)

**What it is:**
- Programming language for data science and machine learning

**Where we used it:**
- ALL backend code (data processing, model training, evaluation)

**Why we chose it:**
- Standard in machine learning research
- Rich ecosystem (PyTorch, NumPy, pandas, scikit-learn)

---

### PyTorch (2.0+)

**What it is:**
- Deep learning framework developed by Meta (Facebook)

**Where we used it:**
- Neural network models (MRI-Only, Late Fusion, Attention Fusion)
- ResNet18 feature extraction
- Training loops and optimization

**Why we chose it:**
- More flexible than TensorFlow
- Better for research (easier to customize)
- Excellent documentation

**Alternatives:**
- TensorFlow/Keras (more production-oriented)
- JAX (newer, less mature ecosystem)

---

### NumPy

**What it is:**
- Library for numerical computing with arrays

**Where we used it:**
- Storing MRI features (512-dimensional arrays)
- Mathematical operations
- Data manipulation

**File format:**
- `.npz` (compressed NumPy arrays) for feature storage

---

### pandas

**What it is:**
- Library for data manipulation with DataFrames (like Excel in Python)

**Where we used it:**
- Loading ADNIMERGE.csv
- Data cleaning and filtering
- Creating train/test CSV files

---

### scikit-learn

**What it is:**
- Machine learning library for traditional ML algorithms

**Where we used it:**
- Random Forest (our best model!)
- Train/test splitting
- Metrics (AUC, accuracy, confusion matrix)
- StandardScaler (normalization)
- StratifiedKFold (cross-validation)

---

### Next.js (16)

**What it is:**
- React framework for building web applications
- Developed by Vercel

**Where we used it:**
- Frontend website (https://neuroscope-mri.vercel.app)
- Server-side rendering for SEO
- File-based routing

**Why we chose it:**
- Best React framework for production
- Excellent performance
- Easy deployment on Vercel

---

### React (19.2)

**What it is:**
- JavaScript library for building user interfaces

**Where we used it:**
- All frontend components
- Interactive visualizations

---

### Three.js (@react-three/fiber)

**What it is:**
- 3D graphics library for the web

**Where we used it:**
- 3D brain visualization on homepage
- Interactive brain explorer page

**Why we chose it:**
- Industry standard for web 3D
- Great React integration

---

### Framer Motion

**What it is:**
- Animation library for React

**Where we used it:**
- Page transitions
- Hover effects
- Smooth animations throughout the site

---

### Tailwind CSS (v4)

**What it is:**
- Utility-first CSS framework

**Where we used it:**
- All styling on the frontend
- Responsive design
- Dark mode support

---

## 🎯 Design Decisions (Why We Used What)

### Why ResNet18 instead of ResNet50?

**Decision:** ResNet18

**Reasons:**
1. **Efficiency:** Faster training and inference
2. **Data size:** We don't have millions of scans to justify ResNet50
3. **Literature:** ResNet18 is standard in medical imaging with small datasets
4. **Results:** Worked well (0.77 AUC on OASIS)

**Trade-off:** ResNet50 might capture more complex patterns, but risks overfitting

---

### Why 2.5D instead of 3D CNN?

**Decision:** 2.5D (multi-slice 2D)

**Reasons:**
1. **Memory:** 3D CNNs need 10x more GPU memory
2. **Data:** 3D CNNs need 10x more training data
3. **Speed:** 2.5D is much faster
4. **Performance:** 2.5D achieves 90% of 3D performance with 10% of the cost

**Trade-off:** Lose some 3D spatial context, but gain practicality

---

### Why Late Fusion instead of Early Fusion?

**Decision:** Late Fusion (concatenate features)

**Reasons:**
1. **Modality mismatch:** Can't directly combine images and numbers
2. **Interpretability:** Can analyze each modality separately
3. **Flexibility:** Easy to swap out components
4. **Literature:** Standard approach for multimodal fusion

**Alternative tried:** Attention Fusion (no significant improvement)

---

### Why Random Forest for Longitudinal?

**Decision:** Random Forest (not LSTM or Transformer)

**Reasons:**
1. **Data size:** 341 subjects too small for deep learning
2. **Feature type:** Tabular data (21 features) suits Random Forest
3. **Interpretability:** Can see feature importance
4. **Performance:** 0.848 AUC (best result!)

**What we learned:** Deep learning isn't always the answer

---

### Why Exclude MMSE/CDR-SB (Level-1)?

**Decision:** Exclude cognitive tests from "honest" models

**Reasons:**
1. **Circular reasoning:** These tests measure the outcome we're predicting
2. **Real-world applicability:** Early detection means BEFORE cognitive symptoms
3. **Scientific integrity:** Want to know if MRI + biomarkers alone can predict

**Result:** Level-1 (honest) = 0.60 AUC, Level-2 (circular) = 0.99 AUC

---

### Why Bootstrap for Confidence Intervals?

**Decision:** Bootstrap (1000 iterations)

**Reasons:**
1. **Small sample size:** Parametric methods assume large N
2. **No assumptions:** Doesn't require normal distribution
3. **Standard practice:** Used in medical ML papers
4. **Robust:** Works with any metric (AUC, accuracy, etc.)

---

### Why Subject-Wise Splitting?

**Decision:** Split by subjects, not scans

**Reasons:**
1. **Prevent leakage:** Scans from same person are correlated
2. **Real-world scenario:** Model will see NEW patients, not new scans of known patients
3. **Scientific rigor:** Standard practice in medical ML

**Impact:** Lower AUC than scan-wise splitting, but honest evaluation

---

### Why ADNI over OASIS?

**Decision:** Focus on ADNI for main experiments

**Reasons:**
1. **Biomarkers:** ADNI has CSF, genetics, cognition
2. **Longitudinal:** Can study progression over time
3. **Sample size:** 629 subjects vs 205 in OASIS
4. **Relevance:** ADNI is the gold standard in AD research

**Trade-off:** Requires approval, more complex data

---

### Why 5-Fold Cross-Validation?

**Decision:** 5 folds (not 10)

**Reasons:**
1. **Computational cost:** 10-fold takes 2x longer
2. **Bias-variance trade-off:** 5-fold is the sweet spot
3. **Sample size:** With 341 subjects, 5 folds gives ~68 test subjects per fold
4. **Standard:** Most papers use 5-fold

---

## 📚 Summary: Where Everything Was Used

| Concept | Files | Purpose |
|---------|-------|---------|
| **ResNet18** | `mri_feature_extraction.py`, `adni_feature_extraction.py` | Convert MRI scans to 512-dim features |
| **Late Fusion** | `train_level1.py`, `train_level_max.py` | Combine MRI + clinical features |
| **Random Forest** | `project_longitudinal_fusion/scripts/06_full_cohort_analysis.py` | Best longitudinal model (0.848 AUC) |
| **Bootstrap CI** | All training scripts | Calculate confidence intervals |
| **Subject-Wise Split** | `data_split.py`, `cross_validation.py` | Prevent data leakage |
| **ADNIMERGE** | `adnimerge_utils.py` | Source of clinical features |
| **Hippocampus** | Level-MAX experiments | Most important biomarker |
| **Next.js** | `project/frontend/` | Interactive research portal |

---

## 🎤 Presentation Tips

### Key Points to Emphasize:

1. **Honest Methodology:**
   - We excluded circular features (MMSE)
   - Zero data leakage (verified with audits)
   - Transparent about limitations

2. **Breakthrough Results:**
   - 0.848 AUC (Longitudinal) - publication-ready!
   - 0.808 AUC (Level-MAX) - proves fusion works with proper biomarkers

3. **Key Insight:**
   - Fusion failed with weak features (age/sex)
   - Fusion succeeded with strong features (hippocampus, CSF, APOE4)
   - **Lesson:** Feature quality > Model complexity

4. **Practical Impact:**
   - Hippocampal atrophy rate is the best predictor
   - Simple Random Forest beats complex LSTM
   - Real-world applicable (no circular features)

---

**End of Technical Glossary**

*For questions during presentation, refer back to this document!*
