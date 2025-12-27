# Longitudinal Task Definition

## Research Question
> Does observing CHANGE over time (multiple MRIs of the same person) help detect or predict dementia progression more reliably?

## Task Type: Progression Prediction

### Label Definition
- **Stable (Label=0)**: Subject's diagnosis remained the same from baseline to last visit
- **Converter (Label=1)**: Subject's diagnosis worsened (CN→MCI, CN→AD, MCI→AD)

### Example Progression Paths
```
Stable:    CN → CN → CN → CN      (no change)
Stable:    MCI → MCI → MCI        (no change)
Converter: CN → CN → MCI → MCI    (worsened)
Converter: MCI → MCI → AD         (worsened)
```

## Input Features

### Per-Scan Features
- 512-dimensional ResNet18 CNN embeddings
- Extracted from 9 slices (3 axial, 3 coronal, 3 sagittal)
- Mean-pooled across slices

### Temporal Structure
- Multiple scans per subject (1-6+ visits)
- Visits ordered by scan date
- Typical intervals: 6 months, 12 months, 24 months

## Model Approaches

### 1. Single-Scan (Baseline Comparison)
- Uses ONLY first visit per subject
- Matches cross-sectional baseline experiment
- No temporal information used

### 2. Delta Model (Change-Based)
- Uses first + last visit per subject
- Explicit delta computation: `delta = last_features - first_features`
- Input: (baseline, followup, delta) concatenated

### 3. Sequence Model (Full Temporal)
- Uses ALL available visits as a sequence
- LSTM/GRU to model temporal dynamics
- Input: Variable-length sequence [visit1, visit2, ..., visitN]

## What This Experiment Tests

1. **Does observing temporal change help?**
   - Compare delta model vs single-scan
   
2. **Does full trajectory help more than simple delta?**
   - Compare sequence model vs delta model
   
3. **Is complexity worth it?**
   - Compare sequence model vs single-scan

## Important Distinctions from Baseline Experiment

| Aspect | Baseline (project_adni) | This Experiment |
|--------|------------------------|-----------------|
| Scans used | 1 per subject | ALL per subject |
| Task | Cross-sectional detection | Progression prediction |
| Label source | Current diagnosis | Future diagnosis change |
| Temporal info | None | Full trajectory |
