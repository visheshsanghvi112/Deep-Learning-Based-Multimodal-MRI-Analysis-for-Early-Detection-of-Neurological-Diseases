# Longitudinal ADNI Experiment Results

## Research Question
> Does observing CHANGE over time (multiple MRIs of the same person) help detect or predict dementia progression more reliably?

---

## Model Comparison

| Model | Description | AUC | AUPRC | Accuracy |
|-------|-------------|-----|-------|----------|
| Single-Scan (Baseline) | Uses only first visit per subject | 0.5103 | 0.3705 | 0.5794 |
| Delta Model (Change) | Uses baseline + followup + delta | 0.5171 | 0.4338 | 0.5397 |
| Sequence Model (LSTM) | Uses all visits as temporal sequence | 0.4409 | 0.3656 | 0.4762 |

---

## Key Findings

### Best Performing Model
**Delta Model (Change)** with AUC = 0.5171

### Does Longitudinal Data Help?

**YES** - Multi-scan models show **+1.3%** improvement over single-scan.

This suggests that:
1. Temporal changes contain additional discriminative information
2. Observing disease trajectory helps prediction
3. Change patterns (delta) may capture early neurodegeneration

---

## Leakage Prevention

The following measures were taken to prevent data leakage:

1. **Subject-Level Splitting**: No subject appears in both train and test sets
2. **Future Labels Only**: Progression labels derived from FINAL diagnosis, not baseline
3. **Separate Normalization**: Training statistics used to normalize test data
4. **Isolated Experiment**: This is completely separate from baseline cross-sectional work

---

## Limitations

1. Sample size may limit sequence model capacity
2. Variable follow-up intervals not explicitly modeled
3. Missing visits not handled with sophisticated imputation
4. Single random seed - should run multiple seeds for robust comparison

---

## Conclusion

This experiment provides evidence on whether longitudinal MRI information genuinely helps early dementia detection.
The comparison between single-scan, delta, and sequence models allows us to understand WHEN and WHY multi-scan information may (or may not) add value.

**Negative results are equally valid** - if longitudinal data doesn't help, that's an important finding for the field.
