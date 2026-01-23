# 🧠 Longitudinal Analysis: Source & Discovery Phase

**Status:** 📁 ARCHIVED / SOURCE  
**Role:** Data Origin & Initial Investigation  
**Best Result:** 0.83 AUC (Biomarkers) | 0.52 AUC (ResNet)

---

## 🔍 What This Project Is
This is the **foundation**. It performed the initial mass extraction of data from ADNI and conducted the experiments that revealed *why* standard Deep Learning failed and *why* Biomarkers succeeded.

**Do NOT use this for the final model.** Use `project_longitudinal_fusion` instead.

---

## 🧪 Key Discoveries Made Here

### 1. The ResNet Failure (0.52 AUC)
We extracted 2,294 MRI scans and passed them through a ResNet18 (ImageNet).
- **Result:** Random-chance performance.
- **Reason:** Standard CNNs are "texture-biased" and scale-invariant; they failed to detect the subtle, global shrinkage of the hippocampus critical for AD diagnosis.

### 2. The Biomarker Breakthrough (0.83 AUC)
We pivoted to using clinical biomarkers (Volumes per ROI) from `ADNIMERGE`.
- **Result:** **0.83 AUC** instantly.
- **Insight:** "Delta" features (rate of change) were far more predictive than static snapshots.

### 3. Data Quality Fixes
- **Found:** 136 subjects labeled "Stable" who actually had Dementia.
- **Fix:** Redefined cohort to MCI-only to ensure valid progression labels.

---

## 📂 Codebase Inventory

### Data Processing (`src/`)
- **`src/feature_extraction.py`**: The script that ran ResNet on 2,000+ files. (Heavy lifting).
- **`src/data_inventory.py`**: Indexed the massive ADNI raw directory.
- **`src/train_delta_model.py`**: The first successful attempt at modeling change over time.

### Documentation found in `docs/`
- **`INVESTIGATION_REPORT.md`**: Detailed forensic analysis of why the first attempts failed. (Critical reading).
- **`LEAKAGE_PREVENTION.md`**: Our initial protocol for safe cross-validation.

---
*Status: Archived. Refer to `project_longitudinal_fusion` for final results.*
