# 🚀 QUICK START GUIDE

## What is This?

This is a **NEW experiment** to test if combining MRI + Biomarkers beats biomarker-only models.

**Goal:** Achieve >0.85 AUC by fusing ResNet features + Hippocampus/Ventricles/APOE4

## ⚠️ CRITICAL: No Existing Work is Modified!

✅ `project/` - UNTOUCHED  
✅ `project_adni/` - UNTOUCHED  
✅ `project_longitudinal/` - UNTOUCHED  

This is a completely separate experiment in `project_biomarker_fusion/`

---

## How to Run

### Option 1: Run Full Pipeline (Recommended)

```powershell
cd D:\discs\project_biomarker_fusion
.\run_pipeline.ps1
```

This will:
1. Extract biomarkers from ADNIMERGE
2. Combine with ResNet features
3. Train PyTorch fusion model
4. Generate comparison report

**Time:** ~30-60 minutes (depending on GPU)

### Option 2: Run Step-by-Step

```powershell
# Step 1: Extract biomarkers
python src\01_extract_biomarkers.py

# Step 2: Prepare dataset
python src\02_prepare_fusion_data.py

# Step 3: Train model
python src\03_train_fusion.py

# Step 4: Evaluate
python src\04_evaluate.py
```

---

## Expected Results

| Model | AUC | Status |
|-------|-----|--------|
| ResNet-only | 0.52 | Baseline (from project_longitudinal/) |
| Biomarker-only | 0.83 | Previous best (logistic regression) |
| **Fusion (THIS)** | **???** | **To be determined!** |

### Possible Outcomes:

**Scenario A: Fusion > 0.85**
- 🎯 **Publication-ready!**
- Deep learning fusion adds clear value
- Paper angle: "Multimodal fusion beats simple models"

**Scenario B: Fusion ≈ 0.83**
- 🤝 **Still publishable!**
- Validates biomarker findings
- Paper angle: "Simple models competitive, but fusion validates approach"

**Scenario C: Fusion < 0.83**
- 📊 **Honest reporting**
- Simple models preferred (Occam's Razor)
- Paper angle: "When deep learning doesn't help (and why)"

---

## Files Created

```
project_biomarker_fusion/
├── README.md                          ← Overview
├── QUICKSTART.md                      ← This file
├── run_pipeline.ps1                   ← Master script
│
├── data/
│   ├── biomarker_longitudinal.npz     ← Extracted biomarkers
│   ├── biomarker_longitudinal.csv     ← (for inspection)
│   └── fusion_dataset.npz             ← Combined dataset
│
├── src/
│   ├── 01_extract_biomarkers.py       ← Extract from ADNIMERGE
│   ├── 02_prepare_fusion_data.py      ← Combine ResNet + biomarkers
│   ├── 03_train_fusion.py             ← Train PyTorch model
│   └── 04_evaluate.py                 ← Compare all models
│
└── results/
    ├── checkpoints/best_model.pt      ← Trained weights
    ├── metrics.json                   ← Performance metrics
    ├── comparison.json                ← vs baselines
    └── model_comparison.png           ← Visualization
```

---

## Safety Checklist

Before running, verify:

- [x] `project/` exists and is untouched
- [x] `project_adni/` exists and is untouched
- [x] `project_longitudinal/` exists and is untouched
- [x] `data/ADNI/ADNIMERGE_23Dec2025.csv` exists
- [x] `project_longitudinal/data/features/longitudinal_features.npz` exists

All scripts have read-only access to existing data. New files only go to `project_biomarker_fusion/`.

---

## Troubleshooting

### Error: "FileNotFoundError: ADNIMERGE"
```powershell
# Check if ADNIMERGE exists
ls D:\discs\data\ADNI\ADNIMERGE_23Dec2025.csv
```

### Error: "Missing longitudinal_features.npz"
```powershell
# Check if longitudinal features exist
ls D:\discs\project_longitudinal\data\features\longitudinal_features.npz
```

### CUDA Out of Memory
Reduce batch size in `src/03_train_fusion.py`:
```python
BATCH_SIZE = 16  # was 32
```

### Low AUC (<0.70)
Check:
1. Data balance (converters vs stable)
2. Feature standardization
3. Try different hyperparameters

---

## What Happens Next?

### If Results are Strong (>0.85):
1. Update main README.md
2. Update PROJECT_DOCUMENTATION.md
3. Add to research paper
4. Celebrate! 🎉

### If Results are Moderate (≈0.83):
1. Document findings honestly
2. Compare with simpler models
3. Paper: "When is deep learning worth it?"

### If Results are Weak (<0.80):
1. Investigate why (ablation studies)
2. Try different architectures
3. Paper: "Lessons learned from fusion experiments"

---

## Questions?

Check the main README.md for detailed methodology.

**Remember:** This is a NEW experiment. Your existing work is 100% safe! ✅
