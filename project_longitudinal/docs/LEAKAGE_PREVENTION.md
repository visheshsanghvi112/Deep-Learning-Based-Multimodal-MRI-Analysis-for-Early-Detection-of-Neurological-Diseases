# Leakage Prevention Measures

This document enumerates all measures taken to prevent data leakage in the longitudinal experiment.

## 1. Subject-Level Train/Test Split

### Implementation
```python
train_subjects, test_subjects = train_test_split(
    unique_subjects, 
    test_size=0.2, 
    stratify=labels,
    random_state=42
)
```

### Verification
```python
train_set = set(train_subjects)
test_set = set(test_subjects)
overlap = train_set & test_set
assert len(overlap) == 0, "LEAKAGE DETECTED!"
```

### Why It Matters
- Same subject must NEVER appear in both train and test
- All visits from a subject are in the SAME split
- Prevents model from memorizing subject-specific patterns

## 2. Temporal Label Construction

### Implementation
- Labels derived from LAST diagnosis, not baseline
- Progression = compare(baseline_diagnosis, final_diagnosis)
- Future information used ONLY for labeling, not as input

### Example
```
Subject 001:
  Visit 1 (2006): CN    ← This is INPUT
  Visit 2 (2007): CN    ← This is INPUT
  Visit 3 (2008): MCI   ← This determines LABEL (Converter)
```

### Why It Matters
- Prevents using future diagnosis as input feature
- Models must predict future from past, not recognize present

## 3. Normalization Protocol

### Implementation
```python
scaler = StandardScaler()
scaler.fit(train_features)  # FIT on train ONLY
train_normalized = scaler.transform(train_features)
test_normalized = scaler.transform(test_features)  # TRANSFORM only
```

### Why It Matters
- Test set statistics must not influence training
- Prevents information from test set leaking into normalization

## 4. Temporal Ordering

### Implementation
- Visits sorted by scan date before processing
- Sequence models receive visits in chronological order
- No future visits can influence prediction of past

### Code
```python
order = np.argsort(visit_dates)
features_sorted = features[order]
```

## 5. Isolated Experiment

### Separation from Baseline Work
- Completely new folder: `project_longitudinal/`
- No shared code with `project_adni/`
- Different data splits (generated independently)
- Different labels (progression vs cross-sectional)

### Why It Matters
- Prevents contamination from baseline experiment
- Results are independently valid

## Verification Checklist

- [x] No subject overlap between train and test
- [x] Labels derived from future information only
- [x] Normalization fit on train, applied to test
- [x] Temporal ordering preserved
- [x] Completely isolated from baseline experiment
- [x] Random seed fixed for reproducibility
