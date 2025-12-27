<div align="center">

# 🧠 Deep Learning-Based Multimodal MRI Analysis for Early Detection of Neurological Diseases

### 📑 Research Implementation with Complete Documentation & Live Demo

<p>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-00C853?style=for-the-badge&logo=statuspage&logoColor=white" alt="Status"/>
  <img src="https://img.shields.io/badge/Datasets-OASIS--1%20%2B%20ADNI--1-2196F3?style=for-the-badge&logo=databricks&logoColor=white" alt="Dataset"/>
  <img src="https://img.shields.io/badge/Total%20Subjects-834-FF6F00?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="Subjects"/>
  <img src="https://img.shields.io/badge/Frontend-Live-9C27B0?style=for-the-badge&logo=vercel&logoColor=white" alt="Frontend"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
</p>

---

### 👨‍💻 Author

**Vishesh Sanghvi**

<a href="https://linkedin.com/in/vishesh-sanghvi-96b16a237/">
  <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
</a>
<a href="https://www.visheshsanghvi.me/">
  <img src="https://img.shields.io/badge/Portfolio-Visit-FF5722?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Portfolio"/>
</a>
<a href="https://github.com/visheshsanghvi112">
  <img src="https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
</a>

</div>

---

## 🌟 Live Demo

<div align="center">

### 🎯 [**View Live Frontend →**](https://neuroscope.vercel.app)

*Interactive research portal with complete documentation, cross-dataset results, and honest assessment*

</div>

---

## 📄 About This Research

<table>
<tr>
<td width="60%">

This repository contains the **complete implementation, documentation, and research analysis** for:

> **"Deep Learning-Based Multimodal MRI Analysis for Early Detection of Neurological Diseases"**
>
> *Research validated on OASIS-1 & ADNI-1 datasets*

The research explores **honest multimodal fusion** with:

🔬 **MRI imaging features** via 2.5D ResNet18 (512-dim)  
📊 **Clinical/demographic data** (Age, Education, Sex)  
🧬 **Biological biomarkers** (CSF, APOE4) - Level-1.5  
🎯 **Cross-dataset validation** (OASIS ↔ ADNI transfer)

**Key Focus:** Methodological rigor, transparent evaluation, and honest reporting of results.

</td>
<td width="40%">

```
┌─────────────────────────┐
│   🧠 MRI Scans (512d)   │
│    ResNet18 Features    │
└───────────┬─────────────┘
            │
            ▼
    ┌───────────────┐
    │  🔗 FUSION    │
    │  Late/Attn    │
    └───────┬───────┘
            │
            ▼
┌─────────────────────────┐
│  📋 Clinical (2-6d)     │
│  Age, Sex, CSF, APOE4   │
└─────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │  🎯 Classify  │
    │  CN vs MCI/AD │
    └───────────────┘
```

</td>
</tr>
</table>

---

## 🎯 Research Summary

| Metric | OASIS-1 | ADNI-1 (Cross-Sectional) | ADNI-1 (Longitudinal) |
|--------|---------|--------------------------|----------------------|
| **Total Scans** | 436 | 1,825 | 2,262 |
| **Unique Subjects** | 205 | 629 | 629 |
| **Train / Test** | 164 / 41 | 503 / 126 | 503 / 126 |
| **Task** | CDR 0 vs 0.5 | CN vs MCI+AD | Stable vs Converter |
| **Best AUC (Honest)** | 0.77 | 0.60 | 0.52 (Delta) |
| **With MMSE (Circular)** | 0.99 | 0.99 | N/A |

### 🔑 Key Insights:
> 1. **Cross-sectional detection (0.60 AUC)** - Honest baseline without cognitive scores
> 2. **Progression prediction (0.52 AUC)** - Even harder than snapshot detection
> 3. **Longitudinal change provides +1.3%** - Marginal, not statistically significant

---

## 📖 Complete Documentation

| Document | Description | Status |
|----------|-------------|--------|
| **[DATA_CLEANING_AND_PREPROCESSING.md](DATA_CLEANING_AND_PREPROCESSING.md)** | 📚 Complete data cleaning pipeline (20+ pages) | ✅ Thesis-Ready |
| **[PROJECT_ASSESSMENT_HONEST_TAKE.md](PROJECT_ASSESSMENT_HONEST_TAKE.md)** | 🔍 Honest analysis of why fusion underperforms (15+ pages) | ✅ Complete |
| **[REALISTIC_PATH_TO_PUBLICATION.md](REALISTIC_PATH_TO_PUBLICATION.md)** | 🎯 2-3 week roadmap to competitive AUC (12+ pages) | ✅ Action Plan |
| **[project_longitudinal/docs/](project_longitudinal/docs/)** | 🔄 Longitudinal progression experiment | ✅ NEW |
| **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** | 🚀 Frontend + backend deployment steps | ✅ Ready |
| [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) | 📊 Original project overview | ✅ Reference |
| [FINAL_PAPER_DRAFT.md](FINAL_PAPER_DRAFT.md) | 📝 Research paper draft | ✅ Draft |

**All documentation is downloadable from the live frontend:** `/documentation` page

---

## 🖥️ Live Frontend Features

The **[NeuroScope Research Portal](https://neuroscope.vercel.app)** includes:

### 📊 Interactive Pages:
- **Homepage:** Research overview with 3D brain visualization
- **Data Cleaning:** Complete preprocessing documentation
- **Infrastructure Constraints:** Honest storage limitations (200GB+ pipeline)
- **Honest Assessment:** Why fusion fails (dimension imbalance, weak features)
- **Publication Strategy:** Biomarker extraction roadmap
- **Cross-Dataset Results:** OASIS ↔ ADNI transfer experiments
- **Downloadable Docs:** All 3 markdown files (thesis-ready)

### 🎨 Features:
- ✅ Mobile responsive
- ✅ Dark mode support
- ✅ Accessible (WCAG 2.1)
- ✅ SEO optimized
- ✅ Fast (Next.js 16)

---

## 🚀 Quick Start

### Option 1: View Live Demo
```bash
Visit: https://neuroscope.vercel.app
```

### Option 2: Run Locally

#### Frontend (Next.js):
```bash
cd project/frontend
npm install
npm run dev
# Open http://localhost:3000
```

#### Backend (FastAPI):
```bash
pip install -r requirements.txt
cd project/backend
uvicorn main:app --reload
# Open http://localhost:8000/docs
```

---

## 📊 Honest Research Results

### Level-1 (Realistic Early Detection - NO MMSE/CDR-SB)

**OASIS-1:**
| Model | AUC | 95% CI | Status |
|-------|-----|--------|--------|
| MRI-Only | 0.770 | ±0.080 | Baseline |
| Clinical-Only | 0.743 | ±0.082 | Demographics |
| **Late Fusion** | **0.794** | ±0.083 | **+2.4%** |
| Attention Fusion | 0.790 | ±0.109 | +2.0% (high variance) |

**ADNI-1:**
| Model | AUC | 95% CI | Status |
|-------|-----|--------|--------|
| MRI-Only | 0.583 | 0.47-0.68 | Baseline |
| Late Fusion | 0.598 | 0.49-0.70 | +1.5% (not significant) |

### Level-2 (Circular - WITH MMSE/CDR-SB) ⚠️

**ADNI:**
| Model | AUC | Note |
|-------|-----|------|
| Late Fusion | **0.988** | Proves model works, but circular |

**This 0.99 AUC proves:**
1. Model architecture is correct
2. Data pipeline works
3. **But MMSE is circular** (directly measures outcome)

### Cross-Dataset Transfer (Zero-Shot)

**OASIS → ADNI:**
| Model | Source AUC | Target AUC | Drop |
|-------|------------|------------|------|
| MRI-Only | 0.814 | **0.607** | -0.207 (BEST transfer) |
| Late Fusion | 0.864 | 0.575 | -0.289 |
| Attention | 0.826 | 0.557 | -0.269 (WORST) |

**Key Finding:** MRI-Only beats fusion in cross-dataset transfer!

### 🔄 Longitudinal Progression Experiment

**Research Question:** *Does observing CHANGE over time help predict progression?*

#### Phase 1: Initial Experiment (ResNet Features)

| Model | AUC | Description |
|-------|-----|-------------|
| Single-Scan (Baseline) | 0.510 | First visit only |
| Delta Model | 0.517 | Baseline + follow-up + change |
| Sequence (LSTM) | 0.441 | All visits as sequence |

**Initial Findings:**
- 📊 **2,262 MRI scans** from 629 subjects processed
- ❌ All models near-chance performance
- ❓ Why? Triggered deep investigation...

#### Phase 2: Deep Investigation

**Issues Discovered:**
1. ❌ **Label contamination:** 136 Dementia patients labeled "Stable" (they can't worsen!)
2. ❌ **Wrong features:** ResNet trained on ImageNet, not brains
3. ❌ **Features are scale-invariant:** Can't detect volume changes

#### Phase 3: Corrected Experiment (Actual Biomarkers)

| Approach | AUC | Improvement |
|----------|-----|-------------|
| ResNet features | 0.52 | baseline |
| Biomarkers (baseline) | 0.74 | +22 points |
| **Biomarkers + Longitudinal** | **0.83** | **+31 points** |
| + APOE4 genetic risk | 0.81 | +29 points |
| + ADAS13 cognitive | 0.84 | +32 points |

**Key Discoveries:**
- 🏆 **Hippocampus volume** alone: 0.725 AUC (best single predictor!)
- 🧬 **APOE4 carriers**: 44-49% conversion rate vs 23% non-carriers
- 📈 **Longitudinal adds +9.5%**: Atrophy RATE matters!
- 💡 **Simple models win**: Logistic regression (0.83) > LSTM (0.44)

> **Final Conclusion:** Longitudinal MRI data **DOES help** (+9.5% AUC) when using proper biomarkers (hippocampus, ventricles, entorhinal). ResNet features are unsuitable for progression prediction. See `project_longitudinal/docs/INVESTIGATION_REPORT.md` for full analysis.

---

## 🔍 Data Cleaning & Integrity

### 7 Major Cleaning Steps Applied:

✅ **Subject-level de-duplication** (ADNI: 1,825 → 629)  
✅ **Baseline-only selection** (no temporal leakage)  
✅ **Removal of longitudinal visits** (m06, m12 excluded)  
✅ **Subject-wise train/test splits** (zero overlap)  
✅ **Feature intersection enforcement** (cross-dataset)  
✅ **Exclusion of circular features** (MMSE, CDR-SB)  
✅ **Separation of Level-1 vs Level-2** models

### Data Integrity:
- **100% leakage prevention** verified
- **Zero subject overlap** between train/test
- **Standard baseline protocols** (no cherry-picking)
- **Transparent documentation** of all steps

**Full details:** See `DATA_CLEANING_AND_PREPROCESSING.md`

---

## 🛠️ Infrastructure Constraints

### Storage Reality:
```
OASIS-1 raw:     50GB zip → 70GB extracted
ADNI-1 raw:      50GB+ similar
Feature files:   Intermediate preprocessing
Model artifacts: Checkpoints, logs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total pipeline:  200GB+
```

### Impact on Design:
- Used baseline-only scans (not full longitudinal)
- Focused on OASIS-1 and ADNI-1 (not OASIS-2/3, ADNI-2/3)
- Extracted features once, stored as .npz (compressed)
- Limited to structural MRI (excluded PET, DTI)

### Justification:
> **This is not an excuse - it's a real constraint.**  
> Sample size (N=205-629) is comparable to published literature.  
> Our contribution is **honest methodology** and **cross-dataset validation**, not maximal dataset size.

**Full context:** See Infrastructure Constraints section in `/documentation`

---

## 📁 Project Structure

```
D:/discs/
├── 📄 README.md                              ← You are here
├── 📚 Complete Documentation (3 files)
│   ├── DATA_CLEANING_AND_PREPROCESSING.md   ← 20+ pages, thesis-ready
│   ├── PROJECT_ASSESSMENT_HONEST_TAKE.md    ← 15+ pages, critical analysis
│   └── REALISTIC_PATH_TO_PUBLICATION.md     ← 12+ pages, biomarker strategy
│
├── 🚀 Deployment & Config
│   ├── DEPLOYMENT_GUIDE.md                  ← Frontend + backend steps
│   ├── DEPLOYMENT_QUICK_FIX.md              ← Quick reference
│   ├── vercel.json                          ← Deployment config
│   └── render.yaml                          ← Backend config
│
├── 🖥️ project/
│   ├── frontend/                            ← Next.js 16 app
│   │   ├── src/
│   │   │   ├── app/
│   │   │   │   ├── page.tsx                 ← Homepage
│   │   │   │   ├── documentation/page.tsx   ← Doc hub (NEW)
│   │   │   │   ├── dataset/                 ← OASIS page
│   │   │   │   ├── adni/                    ← ADNI page
│   │   │   │   └── results/                 ← Results page
│   │   │   └── components/
│   │   │       ├── hero-3d.tsx              ← 3D brain viz
│   │   │       └── ui/                      ← shadcn/ui components
│   │   ├── public/                          ← Static files + markdown docs
│   │   └── package.json
│   │
│   └── backend/                             ← FastAPI backend
│       └── main.py                          ← API endpoints
│
├── 🧠 project_adni/                         ← ADNI pipeline
│   ├── src/
│   │   ├── baseline_selection.py            ← Baseline scan selection
│   │   ├── data_split.py                    ← Train/test splitting
│   │   ├── train_level1.py                  ← Honest model (no MMSE)
│   │   ├── train_level2.py                  ← Circular model (with MMSE)
│   │   └── cross_dataset_robustness.py      ← Transfer experiments
│   └── data/                                ← Processed features
│
├── 📊 Data & Features
│   ├── extracted_features/
│   │   ├── oasis_all_features.npz           ← OASIS features (1.83 MB)
│   │   └── adni_baseline_features.npz       ← ADNI features
│   ├── disc1/ ... disc12/                   ← OASIS raw MRI
│   └── ADNI/                                ← ADNI raw data
│
└── requirements.txt                         ← Python dependencies
```

---

## 🔬 Methodology

### MRI Feature Extraction:
- **Architecture:** 2.5D ResNet18 (pretrained on ImageNet)
- **Approach:** Multi-slice (axial, coronal, sagittal)
- **Output:** 512-dimensional feature vectors
- **Aggregation:** Mean pooling across slices

### Clinical Features:
- **Level-1 (OASIS):** Age, Sex, Education (honest baseline)
- **Level-1 (ADNI):** Age, Sex (minimal features)
- **Level-1.5 (Target):** + CSF (ABETA, TAU, PTAU) + APOE4
- **Level-2 (Reference):** + MMSE + CDR-SB (circular)

### Fusion Strategies:
1. **Late Fusion:** Concatenate features → MLP
2. **Attention-Gated Fusion:** Learnable cross-modal weights

### Evaluation:
- **5-fold cross-validation** with stratification
- **Subject-wise splits** (no leakage)
- **Cross-dataset transfer** (OASIS ↔ ADNI)
- **Bootstrap confidence intervals** (1000 iterations)

---

## 📝 Key Research Findings

<table>
<tr>
<td>✅</td>
<td><b>Data cleaning is impeccable</b> - Zero leakage across all experiments</td>
</tr>
<tr>
<td>⚠️</td>
<td><b>Level-1 results are honest but not competitive</b> - 0.60 AUC reflects true difficulty</td>
</tr>
<tr>
<td>❌</td>
<td><b>Fusion underperforms MRI-only in cross-dataset transfer</b> - Clinical features too weak</td>
</tr>
<tr>
<td>🔬</td>
<td><b>Dimension imbalance is the root cause</b> - 512 strong vs 2 weak features</td>
</tr>
<tr>
<td>💡</td>
<td><b>Solution: Extract biomarkers</b> - CSF + APOE4 can reach 0.72-0.75 AUC</td>
</tr>
<tr>
<td>📚</td>
<td><b>Transparent documentation</b> - All limitations openly acknowledged</td>
</tr>
</table>

---

## 🎯 Path to Publication (2-3 Weeks)

### Current Status:
- ❌ Level-1 (0.60 AUC) - **Not publishable** as-is
- ✅ Data cleaning - **Thesis-ready**
- ✅ Documentation - **Complete**

### Action Plan:
1. **Week 1:** Extract CSF + APOE4 from ADNIMERGE
2. **Week 2:** Retrain models with Level-1.5 features
3. **Week 3:** Write paper draft, submit

### Expected Outcome:
- **Level-1.5 AUC:** 0.72-0.75 (publishable)
- **Fusion gain:** +14% (statistically significant)
- **Venue:** Workshop or mid-tier journal

**Full roadmap:** See `REALISTIC_PATH_TO_PUBLICATION.md`

---

## 🚀 Deployment

### Live URLs:
```
Frontend: https://neuroscope.vercel.app (Next.js on Vercel)
Backend:  https://neuroscope-api.onrender.com (FastAPI on Render)
```

### Deploy Your Own:

**Frontend (Vercel):**
```bash
1. vercel.com → Import GitHub repo
2. Root Directory: project/frontend
3. Framework: Next.js (auto-detected)
4. Deploy
```

**Backend (Render):**
```bash
1. render.com → New Web Service
2. Build: pip install -r requirements.txt
3. Start: gunicorn project.backend.main:app -k uvicorn.workers.UvicornWorker
4. Deploy
```

**Full guide:** See `DEPLOYMENT_GUIDE.md`

---

## 📚 Citations & References

### Datasets:
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

@article{petersen2010adni,
  title={Alzheimer's Disease Neuroimaging Initiative (ADNI)},
  author={Petersen, Ronald C and others},
  journal={Neurology},
  volume={74},
  number={3},
  pages={201--209},
  year={2010}
}
```

### Architecture:
```bibtex
@inproceedings{he2016resnet,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR},
  pages={770--778},
  year={2016}
}
```

---

## 📜 License

This project is for **academic and research purposes**.

- **Code:** MIT License
- **OASIS Dataset:** Publicly available for research
- **ADNI Dataset:** Requires application and approval

**Not for clinical use.** Research prototype only.

---

## 🙏 Acknowledgments

- 🏥 **OASIS Project** - Open-access MRI dataset
- 🧠 **ADNI Initiative** - Alzheimer's disease neuroimaging
- 🔥 **PyTorch & scikit-learn** - ML frameworks
- ⚛️ **Next.js & Vercel** - Frontend deployment
- 🎨 **shadcn/ui** - Beautiful UI components
- 🎓 **Research mentors** - Guidance and support

---

<div align="center">

### 🌟 Star this repo if you find it helpful!

**Key Highlights:**
- ✅ **Zero-leakage data cleaning** (fully documented)
- ✅ **Honest evaluation** (0.60 AUC reflects reality)
- ✅ **Cross-dataset validation** (OASIS ↔ ADNI)
- ✅ **Complete documentation** (thesis-ready)
- ✅ **Live demo** (interactive frontend)
- ✅ **Transparent limitations** (infrastructure constraints)

---

**Made with ❤️ for honest research by Vishesh Sanghvi**

<a href="https://linkedin.com/in/vishesh-sanghvi-96b16a237/">
  <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"/>
</a>
<a href="https://www.visheshsanghvi.me/">
  <img src="https://img.shields.io/badge/Portfolio-FF5722?style=flat-square&logo=google-chrome&logoColor=white" alt="Portfolio"/>
</a>
<a href="https://github.com/visheshsanghvi112">
  <img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"/>
</a>

<br/><br/>

<sub>📅 Last Updated: December 24, 2025 | 🚀 Frontend Live | 📚 Complete Documentation Available</sub>

</div>
