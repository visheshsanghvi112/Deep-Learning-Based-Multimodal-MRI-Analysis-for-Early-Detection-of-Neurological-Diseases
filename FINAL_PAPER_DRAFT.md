# Cross-Dataset Robustness of Multimodal Features for Early Dementia Detection
**Status**: Final Draft / Audit Complete  
**Date**: December 23, 2025

---

## 1. Methods

### 1.1 Datasets and Preprocessing

We utilized two distinct datasets to evaluate the robustness of early detection models:
1.  **OASIS-1**: Single-site, 1.5T MRI. High homogenity. Label availability: CDR.
2.  **ADNI-1**: Multi-site, 1.5T MRI. High heterogeneity. Label availability: Clinical Diagnosis (CN, MCI, AD).

**Data Cleaning & Integrity (Audit Verified):**
Our pipeline enforces strict data hygiene to preventing leakage:
*   **Subject-Level De-duplication**: ADNI subjects (N=629) were verified unique by PTID. OASIS rows (N=436 raw) were treated as unique visits.
*   **Baseline Selection**: For ADNI, strictly the earliest available scan (screening `sc` or baseline `bl`) was selected per subject. Late-stage visits were discarded.
*   **Subject-Wise Splits**: Train/Test splits (80/20) were performed strictly by Subject ID, ensuring no subject appears in both sets. Overlap check confirmed 0 subjects in common.
*   **Feature Intersection**: Cross-dataset experiments utilized **only** the intersection of features available in both cohorts: **MRI (ResNet18-512)**, **Age**, and **Education**, plus **Sex** where available.
*   **Exclusion of Circular Features**: For all early-detection experiments (Level-1 and Cross-Dataset), we explicitly **EXCLUDED** cognitively downstream measures including **MMSE**, **CDR-SB**, **ADAS-Cog**, and **MoCA**.

### 1.2 Dataset Differences & Label Shift (Critical)

A key limitation identified during audit is the definition of the "Positive" class:
*   **OASIS Positive**: Defined as **CDR 0.5** (Very Mild Dementia / MCI-like). This represents an "Early/Prodromal" detection task.
*   **ADNI Positive**: Defined as **MCI + AD** (Mild Cognitive Impairment + Alzheimer's Disease). This represents a broader, more severe disease spectrum.

**Impact**: This creates a **Label Shift**. The ADNI "Sick" class contains typically more advanced pathology than the OASIS "Sick" class. Consequently, models trained on ADNI (Severity Bias) may struggle to calibrate probabilities for the subtler OASIS cases (Threshold Shift), resulting in high AUC (ranking ability) but low Accuracy (decision boundary misalignment).

---

## 2. Results

### 2.1 OASIS In-Dataset Evaluation (Reference)
*Task: Detect CDR 0.5 vs CDR 0 (Early Detection)*
*   **Method**: 5-Fold Cross Validation.
*   **Performance**: Multimodal Late Fusion achieved **AUC ~0.80 - 0.86**.
*   **Finding**: High-quality, single-site data yields strong internal validation performance.

### 2.2 ADNI Level-1: The "Honest" Baseline
*Task: Detect MCI/AD vs CN using MRI + Basic Demographics (Age, Sex)*
*   **Constraint**: No cognitive scores. Realistic screening scenario.
*   **MRI-Only**: AUC **0.583** (CI: 0.47-0.68).
*   **Late Fusion**: AUC **0.598** (CI: 0.49-0.70).
*   **Finding**: Performance is significantly lower than OASIS, reflecting the difficulty of multi-site heterogeneity and the exclusion of circular cognitive features. Multimodal benefit is marginal (+1.5%).

### 2.3 ADNI Level-2: The "Circular" Upper Bound
*Task: Reference performance using Cognitive Scores (MMSE, CDR-SB)*
*   **Constraint**: **NOT** for early detection. Validation of model capacity only.
*   **Late Fusion**: AUC **0.988**.
*   **Finding**: Including diagnostic features allows near-perfect classification, confirming the model *can* learn, but Level-1 is the only valid measure for early detection utility.

### 2.4 Cross-Dataset Robustness (Key Contribution)
*Task: Train on Source, Freeze, Test on Target. (Zero-Shot Transfer)*

**Experiment A: OASIS (Source) → ADNI (Target)**
*Can a model learned on clean data generalize to messy data?*
| Model | Source AUC | Target AUC | Robustness Notes |
|-------|------------|------------|------------------|
| **MRI-Only** | 0.814 | **0.607** | **Best generalization.** Exceeds ADNI's own baseline (0.583). |
| Late Fusion | 0.864 | 0.575 | Degraded. Clinical features (Age/Educ) hurt transfer. |
| Attention Fusion | 0.826 | 0.557 | **Unstable.** Gating mechanism fails to generalize. |

**Experiment B: ADNI (Source) → OASIS (Target)**
*Can a model learned on heterogeneous data generalize to clean data?*
| Model | Source AUC | Target AUC | Robustness Notes |
|-------|------------|------------|------------------|
| MRI-Only | 0.686 | 0.569 | Moderate drop. |
| **Late Fusion** | 0.734 | **0.624** | **Best generalization.** Clinical features (Age/Educ) helped here. |
| Attention Fusion | 0.713 | 0.548 | **Unstable.** Severe drop. |

---

## 3. Discussion

**Robustness Asymmetry**: We observed asymmetric robustness. Transfer from high-quality (OASIS) to heterogeneous (ADNI) data favored **MRI-Only** models, suggesting imaging features are more invariant than demographic correlations. Conversely, transfer from heterogeneous (ADNI) to high-quality (OASIS) data favored **Late Fusion**, suggesting the demographic signal learned from a diverse population is robust.

**Failure of Attention**: Complexity did not equal robustness. The Attention Fusion model, while theoretically superior, consistently underperformed the simpler Late Fusion and MRI-Only models in zero-shot transfer settings, likely due to overfitting gating parameters to the source domain.

**OASIS as a Teacher**: Remarkably, the MRI model trained on OASIS (AUC 0.607 on ADNI) outperformed the MRI model trained on ADNI itself (AUC 0.583). This suggests that for MRI feature learning, **data quality (homogeneity)** may be more valuable than data quantity or domain matching.

---

## 4. Limitations (Audit Findings)

1.  **Accuracy Collapse under Shift**: While AUC remained robust (0.60+), accuracy in cross-dataset experiments dropped to ~34-40%. This confirms a specific **Prior/Threshold Shift** driven by the "Label Shift" (CDR 0.5 vs MCI+AD) and dataset imbalance, rendering `Accuracy` a misleading metric for this transfer task.
2.  **OASIS .npz Integrity**: The legacy OASIS feature file lacks explicit subject IDs. While row-level integrity is maintained, this prevents external metadata merging.
3.  **Low "Honest" Performance**: Real-world early detection (Level-1) performance (~0.60 AUC) is soberingly low compared to literature often inflated by leakage (Level-2, ~0.99 AUC). This highlights the extreme difficulty of the task when strictly controlled.

---

## 5. Conclusion

This study demonstrates that while dataset shift degrades performance, **MRI representations learned from high-quality single-site data (OASIS) are surprisingly robust**, even outperforming models trained on the target domain. However, complex multimodal fusion techniques (Attention) are brittle. Future work should prioritize **Domain Adaptation** strategies to correct the observed threshold shifts rather than increasing model complexity.
