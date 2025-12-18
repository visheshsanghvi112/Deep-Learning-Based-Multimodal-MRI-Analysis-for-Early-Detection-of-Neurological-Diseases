# 🧠 Deep Learning-Based Multimodal MRI Analysis for Early Detection of Neurological Diseases

> **National Research Conference Paper Implementation**

[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()
[![Dataset](https://img.shields.io/badge/Dataset-OASIS--1-blue)]()
[![Subjects](https://img.shields.io/badge/Subjects-436-orange)]()
[![Best AUC](https://img.shields.io/badge/Late%20Fusion%20AUC-0.80-success)]()
[![Python](https://img.shields.io/badge/Python-3.8+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)]()

---

## 👤 Author

**Vishesh Sanghvi**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Vishesh%20Sanghvi-0077B5?logo=linkedin)](https://linkedin.com/in/vishesh-sanghvi-96b16a237/)
[![Portfolio](https://img.shields.io/badge/Portfolio-visheshsanghvi.me-000000?logo=vercel)](https://www.visheshsanghvi.me/)

---

## 📄 About This Research

This repository contains the complete implementation and research code for my paper:

**"Deep Learning-Based Multimodal MRI Analysis for Early Detection of Neurological Diseases"**

*Submitted to National Research Conference*

The research explores multimodal fusion strategies combining:
- **MRI imaging features** extracted via CNN (ResNet18)
- **Clinical/demographic data** (Age, Brain Volumes, Education, etc.)

To detect early-stage dementia (CDR 0.5: Very Mild Dementia) from the OASIS-1 dataset.

---

## 🎯 Quick Summary

| Metric | Value |
|--------|-------|
| **Dataset** | OASIS-1 Cross-sectional |
| **Total Subjects** | 436 (205 for classification) |
| **Classification Task** | CDR=0 (Normal) vs CDR=0.5 (Very Mild Dementia) |
| **Best AUC** | **0.80** (Late Fusion) |
| **MRI Features** | 512-dim (ResNet18 CNN) |
| **Clinical Features** | 6-dim (Age, MMSE, nWBV, eTIV, ASF, Educ) |
| **Fusion Strategies** | Late Fusion, Attention-Gated Fusion |

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

### Deep Learning Fusion Models
| Model | AUC | Description |
|-------|-----|-------------|
| **Late Fusion** | **0.80** | Concatenate MRI + Clinical → MLP |
| Attention-Gated Fusion | 0.79 | Learnable attention weights |
| MRI-Only (CNN) | 0.78 | Pure imaging biomarker |

### Traditional ML Baselines (Without MMSE)
| Feature Set | AUC | Notes |
|-------------|-----|-------|
| MRI only (512d) | 0.78 | ResNet18 transfer learning |
| Clinical w/o MMSE (5d) | 0.74 | Demographics + brain volumes |
| Combined (517d) | 0.78 | Feature concatenation |
| nWBV only (baseline) | 0.75 | Brain volume reference |

### With MMSE (⚠️ Data Leakage Concern)
| Feature Set | AUC | Notes |
|-------------|-----|-------|
| Clinical + MMSE (6d) | 0.87 | MMSE highly correlated with CDR |
| Combined (518d) | 0.82 | MMSE dominates |

### Key Research Finding
> **Attention fusion underperforms late fusion on small datasets (N=205)**. Multi-seed analysis showed attention has 22% higher variance than late fusion. This is a valid research finding—attention mechanisms require larger datasets to learn meaningful cross-modal interactions.

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
2. **Feature Extraction**: ResNet18 pretrained on ImageNet → 512-dim embeddings
3. **Clinical Features**: Age, MMSE, brain volumes (z-score normalized)
4. **Fusion Strategies**:
   - **Late Fusion**: Concatenate feature vectors → MLP classifier
   - **Attention-Gated Fusion**: Learnable cross-modal attention weights
5. **Evaluation**: 5-fold cross-validation with multiple random seeds

See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for full details.

---

## 📝 Key Findings

1. **Late fusion achieves best performance** (AUC 0.80) on small medical datasets
2. **MRI provides meaningful signal** beyond brain volume baselines (0.78 vs 0.75)
3. **Attention mechanisms require more data** - higher variance with N=205 subjects
4. **MMSE dominates clinical features** but has data leakage concern with CDR labels
5. **ResNet18 transfer learning works** for dementia detection from structural MRI

---

## 📚 References

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

## 📜 License

This project is for academic and research purposes. The OASIS dataset is publicly available for research use.

---

## 🙏 Acknowledgments

- **OASIS Project** for providing the open-access MRI dataset
- **PyTorch** and **scikit-learn** communities for excellent ML libraries

---

<p align="center">
  <b>Vishesh Sanghvi</b><br>
  <a href="https://linkedin.com/in/vishesh-sanghvi-96b16a237/">LinkedIn</a> • 
  <a href="https://www.visheshsanghvi.me/">Portfolio</a>
</p>

*Last Updated: December 18, 2025*
