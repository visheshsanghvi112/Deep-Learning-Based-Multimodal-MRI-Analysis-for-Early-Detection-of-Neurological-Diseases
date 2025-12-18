# 🧠 Early Dementia Detection from MRI

**Deep Learning-Based Multimodal MRI Analysis for Early Detection of Neurological Diseases**

[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()
[![Dataset](https://img.shields.io/badge/Dataset-OASIS--1-blue)]()
[![Subjects](https://img.shields.io/badge/Subjects-436-orange)]()
[![AUC](https://img.shields.io/badge/Best%20AUC-0.78-success)]()

---

## 🎯 Quick Summary

| Metric | Value |
|--------|-------|
| **Dataset** | OASIS-1 Cross-sectional |
| **Total Subjects** | 436 |
| **Classification Task** | CDR=0 (Normal) vs CDR=0.5 (Very Mild Dementia) |
| **Best AUC (Realistic)** | 0.78 |
| **MRI Features** | 512-dim (ResNet18) |
| **Clinical Features** | 6-dim (Age, MMSE, nWBV, eTIV, ASF, Educ) |

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** | 📚 **Complete project documentation** - Everything you need |
| [ADNI_COMPREHENSIVE_REPORT.md](ADNI_COMPREHENSIVE_REPORT.md) | ADNI dataset analysis (for future work) |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Extract Features (Already Done)
```bash
python mri_feature_extraction.py
# Output: extracted_features/oasis_all_features.npz (1.83 MB)
```

### 3. Load Features for Classification
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load features
data = np.load('extracted_features/oasis_all_features.npz', allow_pickle=True)
mri = data['mri_features']         # (436, 512)
clinical = data['clinical_features']  # (436, 6)
labels = data['labels']            # (436,) CDR values

# Filter to CDR=0 vs CDR=0.5 only
mask = [(l == 0 or l == 0.5) for l in labels]
X = mri[mask]  # 205 subjects
y = np.array([0 if l == 0 else 1 for l in labels[mask]])

# Train and evaluate
clf = LogisticRegression(max_iter=1000, C=0.1)
scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
print(f"MRI-only AUC: {scores.mean():.3f} ± {scores.std():.3f}")
# Output: MRI-only AUC: 0.775 ± 0.074
```

---

## 📊 Classification Results

### Realistic Early Detection (Without MMSE)
| Feature Set | AUC | Notes |
|-------------|-----|-------|
| MRI only (512d) | **0.78** | Pure imaging biomarker |
| Clinical w/o MMSE (5d) | 0.74 | Demographics + brain volumes |
| Combined (517d) | **0.78** | Best realistic combination |
| nWBV only (baseline) | 0.75 | Brain volume reference |

### With MMSE (⚠️ Data Leakage Concern)
| Feature Set | AUC | Notes |
|-------------|-----|-------|
| Clinical + MMSE (6d) | 0.87 | MMSE highly correlated with CDR |
| Combined (518d) | 0.82 | MMSE dominates |

---

## 📁 Project Structure

```
D:/discs/
├── PROJECT_DOCUMENTATION.md    ← 📚 Complete documentation (READ THIS)
├── README.md                   ← Quick start (you are here)
├── requirements.txt            ← Python dependencies
│
├── mri_feature_extraction.py   ← Main CNN extraction pipeline
├── extracted_features/
│   ├── oasis_all_features.npz  ← 436 subjects, all features
│   └── oasis_all_features.pt   ← PyTorch format
│
├── disc1/ ... disc12/          ← OASIS-1 MRI data
├── ADNI/                       ← ADNI dataset (future work)
└── project/                    ← Deep learning model code
```

---

## 🔬 Methodology

1. **MRI Processing**: 2.5D multi-slice approach (axial, coronal, sagittal)
2. **Feature Extraction**: ResNet18 pretrained on ImageNet → 512-dim
3. **Clinical Features**: Age, MMSE, brain volumes (z-score normalized)
4. **Classification**: Logistic Regression, Random Forest, Gradient Boosting

See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for full details.

---

## 📝 Key Findings

1. **MRI provides meaningful signal** (AUC 0.78 > nWBV baseline 0.75)
2. **MMSE dominates clinical features** but has data leakage concern
3. **ResNet18 transfer learning works** for dementia detection
4. **205 subjects usable** for binary classification

---

## 📚 Citation

```bibtex
@article{marcus2007oasis,
  title={Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data},
  author={Marcus, Daniel S and Wang, Tracy H and others},
  journal={Journal of Cognitive Neuroscience},
  volume={19},
  number={9},
  pages={1498--1507},
  year={2007}
}
```

---

*Last Updated: December 18, 2025*
