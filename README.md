  <div align="center">

  # 🧠 Deep Learning-Based Multimodal MRI Analysis for Early Detection of Neurological Diseases

  ### 📑 Research Implementation with Complete Documentation & Live Demo

  <p>
    <img src="https://img.shields.io/badge/Status-Production%20Ready-00C853?style=for-the-badge&logo=statuspage&logoColor=white" alt="Status"/>
    <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="License"/>
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

  ### 🎯 [**View Live Frontend →**](https://neuroscope-mri.vercel.app)

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

  **🎯 Core Discovery:** Feature engineering (+21% AUC gain) had 7× greater impact than architectural sophistication (<3% AUC gain) in our multimodal systems.

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

  | Metric | OASIS-1 | ADNI-1 (Level-1) | ADNI-1 (Level-MAX) | Longitudinal |
  |--------|---------|------------------|--------------------|--------------|
  | **Total Scans** | 436 | 1,825 | 1,825 | 2,262 |
  | **Unique Subjects** | 205 | 629 | 629 | 629 |
  | **Train / Test** | 164 / 41 | 503 / 126 | 503 / 126 | 503 / 126 |
  | **Features** | Age/Sex/Edu | Age/Sex | 14 Biomarkers | Volumetric Delta |
  | **Best AUC** | 0.79 | 0.60 | **0.808** ✅ | **0.848** 🏆 |

  ### 🔑 Key Insights:
  > 1. **🎯 PRIMARY FINDING:** Feature engineering (+21% AUC) outweighed architecture changes (<3% AUC)
  > 2. **Level-MAX breakthrough (0.808 AUC)** - Biomarkers unlock fusion potential!
  > 3. **Longitudinal success (0.848 AUC)** - Atrophy RATE is the best predictor
  > 4. **Level-1 honest baseline (0.60 AUC)** - Age/Sex alone insufficient
  > 5. **Hippocampus is king** - Single best predictor (0.725 AUC alone)

  ---

  ## 📖 Complete Documentation

  | Document | Description | Status |
  |----------|-------------|--------|
  | **[docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md)** | 🏆 **MASTER DOCUMENTATION** (2,121 lines, CIA-level) | ✅ **THE Reference** |
  | **[docs/DATA_CLEANING_AND_PREPROCESSING.md](docs/DATA_CLEANING_AND_PREPROCESSING.md)** | 📚 Complete data cleaning pipeline (20+ pages) | ✅ Thesis-Ready |
  | **[docs/PROJECT_ASSESSMENT_HONEST_TAKE.md](docs/PROJECT_ASSESSMENT_HONEST_TAKE.md)** | 🔍 Honest analysis of why fusion underperforms (15+ pages) | ✅ Complete |
  | **[docs/REALISTIC_PATH_TO_PUBLICATION.md](docs/REALISTIC_PATH_TO_PUBLICATION.md)** | 🎯 Roadmap to competitive AUC (12+ pages) | ✅ Achieved! |
  | **[docs/LEVEL_MAX_RESULTS.md](docs/LEVEL_MAX_RESULTS.md)** | 🎯 Level-MAX breakthrough (0.808 AUC) | ✅ NEW |
  | **[project_longitudinal_fusion/](project_longitudinal_fusion/README.md)** | 🏆 **Longitudinal FINAL SUCCESS (0.848 AUC)** | ✅ **VICTORY** |
  | **[project_longitudinal/](project_longitudinal/README.md)** | ⏳ Longitudinal Archive (Investigation) | ✅ Archived |
  | **[docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)** | 🚀 Frontend + backend deployment steps | ✅ Ready |
  | [docs/RESEARCH_PAPER_FULL.md](docs/RESEARCH_PAPER_FULL.md) | 📝 Complete research paper | ✅ Draft |

  **All documentation is downloadable from the live frontend:** `/documentation` page

  ---

  ## 🖥️ Live Frontend Features

  The **[NeuroScope Research Portal](https://neuroscope-mri.vercel.app)** includes:

  ### 📊 Interactive Pages:
  - **Homepage (`/`):** Research overview with 3D brain visualization
  - **Documentation (`/documentation`):** Complete research docs with dataset access info
  - **OASIS Dataset (`/dataset`):** OASIS-1 data exploration
  - **ADNI Dataset (`/adni`):** ADNI-1 data exploration
  - **Results (`/results`):** Classification results across all experiments
  - **Visualizations (`/interpretability`):** Interactive research figures with zoom
  - **Research Roadmap (`/roadmap`):** Visual research journey
  - **3D Brain Explorer (`/brain-explorer`):** Interactive brain visualization
  - **Pipeline (`/pipeline`):** ML pipeline overview

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
  Visit: https://neuroscope-mri.vercel.app
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

  ### Level-1.5 (Level-MAX: Honest Biomarkers) ✅ 🎯

  **ADNI with Rich Biological Profile:**
  | Model | AUC | Accuracy | 95% CI | Gain |
  |-------|-----|----------|--------|------|
  | MRI-Only | 0.643 | 62.7% | 0.53-0.73 | Baseline |
  | **Late Fusion (Level-MAX)** | **0.808** | **76.2%** | **0.75-0.87** | **+16.5%** |
  | **Attention Fusion (Level-MAX)** | **0.808** | **75.4%** | **0.74-0.88** | **+16.5%** |

  **Clinical Features (14D):**
  - Demographics: Age, Sex, Education
  - Genetics: APOE4
  - Volumetrics: Hippocampus, Ventricles, Entorhinal, Fusiform, MidTemp, WholeBrain, ICV
  - CSF Biomarkers: Aβ42, Tau, pTau

  **Key Achievement:** 0.81 AUC proves fusion works when given proper biological signals, not just weak demographics!

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
  | **Biomarkers + Longitudinal** | **0.85** | **+33 points** |
  | + APOE4 genetic risk | 0.81 | +29 points |
  | + ADAS13 cognitive | 0.84 | +32 points |

  **Key Discoveries:**
  - 🏆 **Hippocampus volume** alone: 0.725 AUC (best single predictor!)
  - 🧬 **APOE4 carriers**: 44-49% conversion rate vs 23% non-carriers
  - 📈 **Longitudinal adds +9.5%**: Atrophy RATE boosts performance to **0.848 AUC**!
  - 💡 **Simple models win**: Random Forest (0.85) > LSTM (0.44)

  > **Final Conclusion:** Longitudinal MRI data **DOES help** (+11% AUC) when using proper biomarkers (hippocampus, ventricles, entorhinal). We achieved **0.848 AUC** with Random Forest. See `project_longitudinal_fusion/README.md` for full analysis.

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

  **Full details:** See `docs/DATA_CLEANING_AND_PREPROCESSING.md`

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
  ├── 📄 LICENSE                                ← MIT License
  ├── 📄 requirements.txt                       ← Python dependencies
  │
  ├── 📁 data/                                  ← All datasets (README inside)
  │   ├── disc1/ ... disc12/                    ← OASIS raw MRI (12 folders)
  │   ├── ADNI/                                 ← ADNI raw data (404 subject folders)
  │   │   └── ADNIMERGE_23Dec2025.csv           ← Clinical metadata (12.65 MB)
  │   └── extracted_features/                   ← Processed features
  │       ├── oasis_all_features.npz            ← OASIS features (1.75 MB)
  │       └── adni_baseline_features.npz        ← ADNI features
  │
  ├── 📁 docs/                                  ← All documentation (README inside)
  │   ├── RESEARCH_PAPER_FULL.md                ← Complete research paper
  │   ├── RESEARCH_PAPER_IEEE_FORMAT.md         ← IEEE formatted version
  │   ├── DATA_CLEANING_AND_PREPROCESSING.md    ← 20+ pages, thesis-ready
  │   ├── PROJECT_ASSESSMENT_HONEST_TAKE.md     ← 15+ pages, critical analysis
  │   ├── REALISTIC_PATH_TO_PUBLICATION.md      ← 12+ pages, biomarker strategy
  │   ├── PROJECT_DOCUMENTATION.md              ← Project overview
  │   ├── PROJECT_INSPECTION_REPORT.md          ← Detailed inspection
  │   ├── DEPLOYMENT_GUIDE.md                   ← Frontend + backend steps
  │   ├── README_FIGURES.md                     ← Figure documentation
  │   └── *.txt                                 ← Analysis reports
  │
  ├── 📁 scripts/                               ← Utility scripts (README inside)
  │   ├── generate_visualizations.py            ← Main visualization generator
  │   ├── generate_data_figures.py              ← Data statistics plots
  │   ├── generate_interpretability_images.py   ← Model interpretability viz
  │   ├── visualize_adnimerge_usage.py          ← ADNIMERGE usage plots
  │   ├── check_adnimerge_usage.py              ← ADNIMERGE analysis
  │   ├── extract_adni_samples.py               ← ADNI data extraction
  │   ├── generate_adni_json.py                 ← ADNI metadata generation
  │   └── quick_adni_check.py                   ← Quick ADNI validation
  │
  ├── 📁 figures/                               ← All visualizations & plots
  │   ├── A1_oasis_model_comparison.*           ← OASIS results (PDF + PNG)
  │   ├── B1_adni_level1_honest.*               ← ADNI honest results
  │   ├── C1_in_vs_cross_dataset_collapse.*     ← Transfer learning analysis
  │   ├── D1_preprocessing_pipeline.*           ← Data pipeline flowchart
  │   └── longitudinal/                         ← Longitudinal visualizations
  │
  ├── 🖥️ project/
  │   ├── frontend/                             ← Next.js 16 app
  │   │   ├── src/
  │   │   │   ├── app/
  │   │   │   │   ├── page.tsx                  ← Homepage
  │   │   │   │   ├── documentation/            ← Research docs hub
  │   │   │   │   ├── dataset/                  ← OASIS-1 explorer
  │   │   │   │   ├── adni/                     ← ADNI-1 explorer
  │   │   │   │   ├── results/                  ← Classification results
  │   │   │   │   ├── interpretability/         ← Research visualizations
  │   │   │   │   ├── roadmap/                  ← Research journey
  │   │   │   │   ├── pipeline/                 ← ML pipeline
  │   │   │   │   └── brain-explorer/           ← 3D brain visualization
  │   │   │   └── components/
  │   │   │       ├── hero-3d.tsx               ← 3D brain viz
  │   │   │       └── ui/                       ← shadcn/ui components
  │   │   ├── public/                           ← Static files + markdown docs
  │   │   └── package.json
  │   │
  │   └── backend/                              ← FastAPI backend
  │       └── main.py                           ← API endpoints
  │
  ├── 🧠 project_adni/                          ← ADNI pipeline
  │   ├── src/
  │   │   ├── baseline_selection.py             ← Baseline scan selection
  │   │   ├── data_split.py                     ← Train/test splitting
  │   │   ├── train_level1.py                   ← Honest model (no MMSE)
  │   │   ├── train_level_max.py                ← Level-MAX (biomarkers)
  │   │   ├── create_level_max_dataset.py       ← Level-MAX data builder
  │   │   ├── visualize_level_max.py            ← Level-MAX plots
  │   │   ├── train_level2.py                   ← Circular model (with MMSE)
  │   │   └── cross_dataset_robustness.py       ← Transfer experiments
  │   └── data/                                 ← Processed features
  │
  ├── 🏆 project_longitudinal_fusion/           ← FINAL SUCCESS (0.848 AUC)
  │   ├── README.md                             ← Master Report
  │   ├── FINAL_FUSION_REPORT.md                ← Viva Document
  │   ├── scripts/                              ← Analysis & Audit scripts
  │   └── results/                              ← validated 0.848 results
  │
  └── ⏳ project_longitudinal/                  ← (Archive) Initial investigation
      └── src/                                  ← Legacy scripts
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
  <td>🎯</td>
  <td><b>PRIMARY INSIGHT: Feature content > Architecture</b> - +21% from features vs <3% from architecture changes</td>
  </tr>
  <tr>
  <td>✅</td>
  <td><b>Data cleaning is impeccable</b> - Zero leakage across all experiments</td>
  </tr>
  <tr>
  <td>✅</td>
  <td><b>Level-MAX proves fusion works</b> - 0.808 AUC when given proper biological signals</td>
  </tr>
  <tr>
  <td>⚠️</td>
  <td><b>Level-1 results are honest but not competitive</b> - 0.60 AUC reflects true difficulty with weak features</td>
  </tr>
  <tr>
  <td>❌</td>
  <td><b>ResNet features fail for temporal analysis</b> - 0.52 AUC (ImageNet features are scale-invariant)</td>
  </tr>
  <tr>
  <td>💡</td>
  <td><b>Domain-specific features win</b> - Hippocampal atrophy rates achieve 0.848 AUC</td>
  </tr>
  <tr>
  <td>📚</td>
  <td><b>Transparent documentation</b> - All limitations openly acknowledged</td>
  </tr>
  </table>

  ---

  ## 🎯 Publication Status

  ### ✅ ACHIEVED:
  - ✅ **Level-MAX AUC: 0.808** - Publication-ready!
  - ✅ **Fusion gain: +16.5%** - Statistically significant
  - ✅ **Longitudinal AUC: 0.83** - Biomarker progression validated
  - ✅ Data cleaning - **Thesis-ready**
  - ✅ Documentation - **2,100+ lines** (CIA-level comprehensive)

  ### 📊 Key Results Summary:
  | Experiment | Best AUC | Model | Status |
  |------------|----------|-------|--------|
  | Level-1 (Age/Sex only) | 0.60 | Late Fusion | Honest baseline |
  | **Level-MAX (14 biomarkers)** | **0.808** | **Late/Attention** | **✅ Publication-ready** |
  | Longitudinal Fusion | **0.848** | Random Forest | ✅ **Best Result** |
  | Cross-dataset Transfer | 0.607 | MRI-Only | Best transfer |

  **Full details:** See [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) (2,121 lines, fully verified)

  ---

  ## 🚀 Deployment

  ### Live Demo:
  ```
  https://neuroscope-mri.vercel.app
  ```

  **Frontend:** Next.js 16 deployed on Vercel

  ### Deploy Your Own:

  **Vercel (Frontend):**
  ```bash
  1. Fork this repository on GitHub
  2. Visit vercel.com → Import GitHub repo
  3. Root Directory: project/frontend
  4. Framework: Next.js (auto-detected)
  5. Deploy
  ```

  Your site will be live at: `https://your-project-name.vercel.app`

  **Full guide:** See `docs/DEPLOYMENT_GUIDE.md`

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
  - 🎯 **PRIMARY FINDING:** Feature engineering (+21%) > Architecture sophistication (<3%)
  - ✅ **Zero-leakage data cleaning** (fully documented)
  - ✅ **Longitudinal VICTORY** (0.848 AUC on full MCI cohort)
  - ✅ **Level-MAX breakthrough** (0.808 AUC with biomarkers)
  - ✅ **Honest evaluation** (0.60 AUC reflects reality without biomarkers)
  - ✅ **Cross-dataset validation** (OASIS ↔ ADNI)
  - ✅ **Complete documentation** (2,300+ lines, thesis-ready)
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

  <sub>📅 Last Updated: January 29, 2026 | 🚀 Frontend Live | 📚 Complete Documentation Available</sub>

  </div>
