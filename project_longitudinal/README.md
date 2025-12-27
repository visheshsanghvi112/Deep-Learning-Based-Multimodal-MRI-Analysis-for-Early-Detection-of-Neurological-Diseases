# Longitudinal ADNI Experiment

**Status:** ✅ Complete | 📊 Results Analyzed | 🔬 Key Discoveries Made

## Overview
This is a **comprehensive longitudinal experiment** investigating whether temporal MRI changes improve dementia progression prediction.

**Research Question:**  
> Does observing CHANGE over time (multiple MRIs of the same person) help detect or predict dementia progression more reliably?

**Answer:** **YES!** But only with proper biomarkers. See findings below.

---

## Research Journey

### Phase 1: Initial Experiment (ResNet Features)

| Model | AUC | Description |
|-------|-----|-------------|
| Single-Scan | 0.510 | Baseline only |
| Delta Model | 0.517 | +0.7% improvement |
| LSTM Sequence | 0.441 | Underperformed |

**Initial Result:** Near-chance performance. Prompted deep investigation.

### Phase 2: Deep Investigation

**Issues Discovered:**
1. ❌ **Label contamination:** 136 Dementia patients labeled "Stable"
2. ❌ **Wrong features:** ResNet trained on ImageNet, scale-invariant
3. ❌ **Feature stability:** Within-subject change too small to detect

### Phase 3: Corrected Experiment (Biomarkers)

| Approach | AUC | Improvement |
|----------|-----|-------------|
| ResNet features | 0.52 | baseline |
| Biomarkers (baseline) | 0.74 | +22 points |
| **Biomarkers + Longitudinal** | **0.83** | **+31 points** |
| + APOE4 | 0.81 | +29 points |
| + ADAS13 | 0.84 | +32 points |

---

## Key Discoveries

1. 🏆 **Hippocampus is best single predictor:** 0.725 AUC alone
2. 🧬 **APOE4 doubles risk:** 23% → 44-49% conversion rate
3. 📈 **Longitudinal adds +9.5%:** 0.74 → 0.83 with temporal change
4. 💡 **Simple models win:** Logistic regression (0.83) > LSTM (0.44)

---

## Data Source

| Metric | Value |
|--------|-------|
| **Path** | `C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T\ADNI` |
| **Subjects** | 639 unique |
| **Total Scans** | 2,294 NIfTI files |
| **Avg Scans/Subject** | 3.6 |
| **MCI Subjects** | 298 (focus cohort) |
| **Clinical Data** | ADNIMERGE CSV |

---

## Project Structure

```
project_longitudinal/
├── src/                        # Python scripts
│   ├── data_inventory.py       # Scan 2,294 NIfTI files
│   ├── data_preparation.py     # Create progression labels
│   ├── feature_extraction.py   # Extract ResNet features
│   ├── train_single_scan.py    # Baseline model
│   ├── train_delta_model.py    # Change-based model
│   ├── train_sequence_model.py # LSTM sequence model
│   └── evaluate.py             # Generate comparison report
├── data/
│   ├── processed/              # CSVs (inventory, splits)
│   └── features/               # longitudinal_features.npz
├── results/
│   ├── single_scan/            # Phase 1 results
│   ├── delta_model/            # Phase 1 results
│   ├── sequence_model/         # Phase 1 results
│   ├── biomarker_analysis/     # Phase 3 results (NEW)
│   ├── comparison_report.md
│   └── metrics_comparison.json
└── docs/
    ├── TASK_DEFINITION.md      # Task specification
    ├── LEAKAGE_PREVENTION.md   # Safety measures
    ├── RESULTS_SUMMARY.md      # Initial results
    └── INVESTIGATION_REPORT.md # Complete analysis (15+ findings)
```

---

## Running the Pipeline

```bash
cd d:\discs\project_longitudinal

# Phase 1: Initial ResNet experiment
python src/data_inventory.py
python src/data_preparation.py
python src/feature_extraction.py  # Takes 2-4 hours
python src/train_single_scan.py
python src/train_delta_model.py
python src/train_sequence_model.py
python src/evaluate.py

# Phase 3: Biomarker analysis (in Python)
# See results/biomarker_analysis/
```

---

## Key Files

| File | Purpose |
|------|---------|
| `docs/INVESTIGATION_REPORT.md` | **Complete 15+ finding analysis** |
| `results/biomarker_analysis/final_findings.json` | Best model results |
| `results/comparison_report.md` | Phase 1 ResNet results |

---

## Important Notes

- ⚠️ **ResNet features don't work** for progression prediction
- ✅ **Biomarkers work:** Hippocampus, ventricles, entorhinal
- ✅ **APOE4 is powerful:** Include in all models
- ⚠️ **Subject-level splits:** No leakage
- ⚠️ **Separate from** `project_adni/` baseline work
