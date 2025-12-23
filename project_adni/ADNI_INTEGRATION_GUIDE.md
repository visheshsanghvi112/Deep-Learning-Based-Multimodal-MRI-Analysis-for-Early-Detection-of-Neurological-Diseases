# 🔬 ADNI + OASIS Integration Strategy

## Current State

### What You Have
| Dataset | Status | Details |
|---------|--------|---------|
| **OASIS-1** | ✅ Fully Integrated | 436 subjects, features extracted, frontend working |
| **ADNI** | 📊 Analyzed Only | 203 subjects, 230 NIfTI scans (2005-2008) |

### Your Current Pipeline (OASIS-1 Only)
```
OASIS MRI (.gif) → ResNet18 Features (512-dim) → Classification → Frontend
     ↓
Clinical Data (age, MMSE, CDR, nWBV) → Fusion Models → Results Display
```

---

## 🎯 Integration Options

### Option 1: **ADNI as External Validation Set** (Recommended First)
**Best for:** Testing if your OASIS-trained model generalizes to new data

```
┌─────────────────────────────────────────────────────────────┐
│ TRAIN on OASIS-1 (436 subjects)                             │
│   - Extract features with existing pipeline                 │
│   - Train your 3 models (MRI-Only, Late Fusion, Attention)  │
│   - Get your AUC ~0.79                                      │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ TEST on ADNI (203 subjects)                                 │
│   - Extract ADNI features with SAME ResNet18                │
│   - Apply trained OASIS models                              │
│   - Report external validation AUC                          │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**
1. Run feature extraction on ADNI (adapt `mri_feature_extraction.py`)
2. Create `adni_features.csv` matching OASIS format
3. Load OASIS-trained models and predict on ADNI
4. Report generalization performance

**Value:** Proves your model isn't overfitting to OASIS quirks

---

### Option 2: **ADNI + OASIS Combined Training** 
**Best for:** Larger dataset, more robust models

```
┌──────────────────────────────────────────┐
│ Combined Dataset (639 subjects)         │
│   - OASIS-1: 436                        │
│   - ADNI:    203                        │
│   - Total:   639                        │
└──────────────────────────────────────────┘
            ↓
     5-Fold Cross-Validation
     (mixing both datasets)
            ↓
  Better generalization, higher power
```

**Implementation:**
1. Extract ADNI features
2. Merge `oasis_features.csv` + `adni_features.csv`
3. Add dataset source column (`dataset: 'oasis' or 'adni'`)
4. Retrain models on combined data
5. Stratify CV by both CDR AND dataset

**Challenges:**
- ADNI uses different acquisition protocols (MPRAGE variants)
- Need to harmonize clinical variables
- Domain shift between datasets

---

### Option 3: **Multi-Dataset Learning with Domain Adaptation**
**Best for:** Advanced research, publication-quality

```
┌─────────────────────────────────────────────────┐
│ Domain-Aware Model                              │
│   - Dataset Indicator as feature                │
│   - Or: Train domain classifier adversarially   │
│   - Learn dataset-invariant features            │
└─────────────────────────────────────────────────┘
```

**Implementation:**
1. Add `dataset_id` as one-hot encoded feature
2. Or use adversarial domain adaptation
3. Train model to predict CDR while being dataset-agnostic
4. Report performance stratified by dataset

---

## 🔧 Step-by-Step: ADNI Feature Extraction

### Step 1: Adapt Feature Extraction Script

Create `adni_feature_extraction.py` (self-contained and runnable):

```python
"""
Extract ResNet18 features from ADNI NIfTI files
Matches OASIS pipeline for compatibility
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ResNet18 backbone identical to OASIS pipeline
_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
RESNET = torch.nn.Sequential(*list(_resnet.children())[:-1]).to(DEVICE).eval()

TRANSFORM = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_nifti_slices(nii_path: Path, num_slices: int = 5):
  """Load middle sagittal neighborhood slices from a NIfTI volume.
  Returns list of 2D uint8 arrays scaled to [0,255]."""
  img = nib.load(str(nii_path))
  data = img.get_fdata()
  mid = data.shape[0] // 2
  slices = []
  for i in range(-num_slices // 2, num_slices // 2 + 1):
    idx = mid + i
    if 0 <= idx < data.shape[0]:
      sl = data[idx, :, :]
      sl = sl.astype(np.float32)
      rng = sl.max() - sl.min()
      if rng > 0:
        sl = (sl - sl.min()) / rng
      sl = (sl * 255.0).clip(0, 255).astype(np.uint8)
      slices.append(sl)
  return slices

def extract_resnet_features(slices: list[np.ndarray]) -> np.ndarray:
  """Generate a single 512-d feature by averaging per-slice embeddings."""
  feats = []
  with torch.no_grad():
    for sl in slices:
      img = Image.fromarray(sl).convert("RGB")
      x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
      f = RESNET(x).squeeze().cpu().numpy()  # (512,)
      feats.append(f)
  if len(feats) == 0:
    return np.zeros(512, dtype=np.float32)
  return np.mean(np.stack(feats, axis=0), axis=0)

def extract_adni_features() -> pd.DataFrame:
  base = Path("D:/discs/ADNI")
  rows = []
  for nii in base.rglob("*.nii"):
    subject_id = nii.parts[-4]  # e.g., 002_S_0295
    slices = load_nifti_slices(nii)
    features = extract_resnet_features(slices)
    rows.append({
      "subject_id": subject_id,
      "scan_path": str(nii),
      **{f"f{i}": float(features[i]) for i in range(512)},
    })
  return pd.DataFrame(rows)

if __name__ == "__main__":
  df = extract_adni_features()
  out = Path("adni_mri_features.csv")
  df.to_csv(out, index=False)
  print(f"Saved {len(df)} rows to {out}")
```

Prerequisites (Windows):

```bash
pip install nibabel numpy pillow torch torchvision
```

### Step 2: Get ADNI Clinical Data

**Problem:** Your ADNI folder only has MRI scans, no clinical data!

**Solutions:**
1. **Download from ADNI website**: https://adni.loni.usc.edu/
   - Need to register (free for researchers)
   - Get `ADNIMERGE.csv` with CDR, MMSE, demographics
   - Match subject IDs (e.g., `002_S_0295`)

2. **Use scan dates as proxy**:
   - Early scans (2005-2006) → likely baseline
   - Late scans (2007-2008) → likely follow-up
   - Estimate progression

3. **Extract from filename metadata**:
   - Some ADNI filenames contain diagnostic info
   - Parse acquisition parameters

### Step 3: Match Clinical to MRI Data

```python
# Load ADNI clinical data
adni_clinical = pd.read_csv("ADNIMERGE.csv")

# Match by subject ID and date
adni_clinical['subject_id'] = adni_clinical['PTID']  # e.g., "002_S_0295"
adni_clinical['scan_date'] = pd.to_datetime(adni_clinical['EXAMDATE'])

# Merge with extracted features
adni_features = extract_adni_features()
adni_merged = adni_features.merge(adni_clinical, on='subject_id')

# Save in OASIS format
adni_merged.to_csv("adni_features.csv")
```

Note: Without `ADNIMERGE.csv`, you will not have reliable CDR/MMSE for ADNI. Use MRI-only models or treat clinical fields as missing.

---

## 📊 Frontend Integration

### Add ADNI to Data Explorer

Update `OasisDataExplorer.tsx` to support both datasets:

```tsx
interface Subject {
  id: string
  dataset: 'oasis' | 'adni'  // NEW
  gender: string
  age: number
  cdr: number | null
  mmse: number | null
  nwbv: number
}

// Fetch both datasets
useEffect(() => {
  (async () => {
    const [oasisRes, adniRes] = await Promise.all([
      fetch("/oasis-data.json"),
      fetch("/adni-data.json"), // NEW
    ])
    const [oasis, adni] = await Promise.all([
      oasisRes.json(),
      adniRes.json(),
    ])
    setData([...(oasis ?? []), ...(adni ?? [])])
  })()
}, [])

// Add dataset filter
<RingFilter 
  label="Dataset"
  segments={(() => {
    const oasisCount = data.filter(d => d.dataset === 'oasis').length
    const adniCount = data.filter(d => d.dataset === 'adni').length
    return [
      { id: 'oasis', label: 'OASIS-1', value: oasisCount, color: '#3b82f6' },
      { id: 'adni', label: 'ADNI', value: adniCount, color: '#f59e0b' },
    ]
  })()}
  selectedIds={filters.dataset}
  onToggle={(id) => toggleFilter('dataset', id)}
/>
```

### Comparative Visualization

```tsx
// Show side-by-side comparison
<div className="grid grid-cols-2 gap-4">
  <Card>
    <CardHeader>OASIS-1 Cohort</CardHeader>
    <CardContent>
      <p>N = {oasisData.length}</p>
      <p>Age: {avgAge(oasisData)}</p>
      <p>CDR 0.5: {countCDR(oasisData, 0.5)}</p>
    </CardContent>
  </Card>
  
  <Card>
    <CardHeader>ADNI Cohort</CardHeader>
    <CardContent>
      <p>N = {adniData.length}</p>
      <p>Age: {avgAge(adniData)}</p>
      <p>CDR 0.5: {countCDR(adniData, 0.5)}</p>
    </CardContent>
  </Card>
</div>
```

---

## 🚀 Recommended Implementation Order

### Phase 1: Feature Extraction (Week 1)
1. ✅ Create `adni_feature_extraction.py` (adapt from OASIS)
2. ✅ Extract ResNet18 features for all 230 scans
3. ✅ Save to `adni_mri_features.csv`

### Phase 2: Clinical Data (Week 1-2)
1. 🔄 Download ADNIMERGE.csv from ADNI portal
2. 🔄 Match subjects by ID and scan date
3. 🔄 Create `adni_clinical_matched.csv`

### Phase 3: External Validation (Week 2)
1. 🔄 Load your trained OASIS models
2. 🔄 Predict on ADNI features
3. 🔄 Report external validation AUC
4. 🔄 Compare OASIS vs ADNI performance

### Phase 4: Combined Training (Week 3)
1. ⏳ Merge OASIS + ADNI datasets
2. ⏳ Retrain models on combined data
3. ⏳ Stratified CV by dataset
4. ⏳ Report improved performance

### Phase 5: Frontend Update (Week 3-4)
1. ⏳ Create `adni-data.json`
2. ⏳ Update data explorer to support both
3. ⏳ Add dataset comparison views
4. ⏳ Show combined statistics

---

## 📈 Expected Outcomes

### If ADNI Validation Works Well (AUC > 0.75)
✅ Your model generalizes across datasets  
✅ Learned features are robust  
✅ Ready for publication  

### If ADNI Validation is Poor (AUC < 0.60)
⚠️ Domain shift issue  
⚠️ Need harmonization or domain adaptation  
⚠️ OASIS-specific overfitting  

### With Combined Training
📈 Likely AUC improvement: +0.03 to +0.05  
📈 More robust to protocol variations  
📈 Better generalization to new data  

---

## 🔍 Key Differences Between OASIS-1 and ADNI

| Aspect | OASIS-1 | ADNI |
|--------|---------|------|
| **Format** | .gif (2D slices) | .nii (3D volumes) |
| **Subjects** | 436 | 203 (in your folder) |
| **Acquisition** | T1-MPRAGE | Multiple MPRAGE variants |
| **Preprocessing** | Unknown | GradWarp, B1, N3 correction |
| **Age Range** | 18-96 years | Typically 55-90 (AD focus) |
| **CDR** | 0, 0.5, 1, 2 | Need clinical file |
| **Scans/Subject** | 1-4 | 1-2 (in your folder) |

---

## 💡 Quick Win: Start Here

**Today:** Run this to see what you have:
```bash
cd D:/discs
python adni_comprehensive_analysis.py  # You already ran this
```

**This Week:** Extract ADNI features
```bash
# Copy and adapt mri_feature_extraction.py
python adni_feature_extraction.py
# Output: adni_mri_features.csv (MRI features only, no clinical yet)
```

**Next Week:** Get clinical data
```bash
# Register at https://adni.loni.usc.edu
# Download ADNIMERGE.csv
# Match and merge
```

**In 2 Weeks:** External validation
```python
# Load OASIS-trained model
# Predict on ADNI
# Report results
```

---

## 🔐 ADNI Data Usage & Access

- ADNI data requires registration and agreement to usage policies at the ADNI/LONI portal.
- Clinical files (e.g., `ADNIMERGE.csv`) must be downloaded from the portal; they are not in your `ADNI/` folder.
- Ensure subject ID (`PTID`, e.g., `002_S_0295`) and exam date alignment when merging MRI with clinical records.

---

## ❓ Questions to Decide

1. **Do you have ADNI clinical data?**
   - If NO → Download from ADNI portal first
   - If YES → Where is it?

2. **What's your goal?**
   - **Validate existing model** → Option 1 (External Validation)
   - **Improve model** → Option 2 (Combined Training)
   - **Publish research** → Option 3 (Domain Adaptation)

3. **Timeline?**
   - **Quick (1 week)** → Just extract features, show in frontend
   - **Thorough (1 month)** → Full validation + combined training

---

Need help with any specific part? I can create the extraction scripts for you!
