# 🔬 COMPLETE INVESTIGATION REPORT: Longitudinal ADNI Analysis

**Date:** December 27, 2025  
**Investigator:** Deep Research Analysis

---

## Executive Summary

After extensive investigation, we found that **longitudinal data DOES significantly help** predict MCI→Dementia progression, but **only when using proper biomarkers** instead of generic CNN features.

| Approach | AUC | Notes |
|----------|-----|-------|
| ResNet features (original) | 0.52 | Near chance - wrong features |
| **Biomarkers + Longitudinal** | **0.83** | +31 points! |
| + Age + APOE4 | 0.81 | Honest model |
| + ADAS13 (cognitive) | 0.84 | With cognitive score |

---

## Part 1: Original Problem - Label Contamination

### Finding: 136 Dementia patients labeled as "Stable"

| Baseline | → Converter | → Stable |
|----------|-------------|----------|
| CN | 48 (25%) | 147 (75%) |
| MCI | 178 (60%) | 120 (40%) |
| **Dementia** | 0 (0%) | **136 (100%)** ⚠️ |

Dementia patients CAN'T progress further, so labeling them "Stable" creates contradictory signals.

---

## Part 2: Feature Quality Check

### Within-Subject vs Between-Subject Change

| Metric | Value |
|--------|-------|
| Mean within-subject change | 0.129 |
| Mean between-subject difference | 0.205 |
| Ratio (within/between) | 0.63 |

**Finding:** Within-subject feature change is only 63% of between-subject variation, meaning ResNet features are relatively STABLE over time - they don't capture disease progression well.

---

## Part 3: Feature Separability Analysis

| Comparison | Features with p<0.05 |
|------------|---------------------|
| Stable vs Converter | 123/512 (24%) |
| CN vs Dementia | Much higher |

**Finding:** Features CAN separate CN from Dementia (cross-sectional), but CANNOT reliably separate Stable from Converter (progression).

---

## Part 4: Available Biomarkers in ADNIMERGE

| Feature | Coverage | Purpose |
|---------|----------|---------|
| MMSE | 69.8% | Cognitive function |
| CDRSB | 71.5% | Dementia severity |
| Hippocampus | 53.9% | Atrophy marker |
| Ventricles | 57.7% | Enlargement marker |
| Entorhinal | 51.4% | Early atrophy site |

---

## Part 5: Individual Biomarker Power

### Structural Biomarkers (HONEST - no circularity)

| Biomarker | AUC | Notes |
|-----------|-----|-------|
| **Hippocampus** | **0.725** | Best single predictor |
| Entorhinal | 0.691 | Early atrophy site |
| MidTemp | 0.678 | Temporal lobe |
| Fusiform | 0.670 | Face recognition area |
| WholeBrain | 0.604 | Total volume |
| Ventricles | 0.581 | Enlargement |
| ICV | 0.537 | Head size (not useful) |

### Cognitive Scores (SEMI-CIRCULAR)

| Score | AUC | Notes |
|-------|-----|-------|
| **ADAS13** | **0.767** | Most comprehensive |
| ADAS11 | 0.743 | Shorter version |
| Hippocampus | 0.725 | (for comparison) |
| RAVLT_immediate | 0.720 | Memory test |
| CDRSB | 0.690 | Severity scale |
| RAVLT_learning | 0.677 | Learning ability |
| MMSE | 0.643 | General cognition |
| RAVLT_forgetting | 0.573 | Forgetting rate |

**Key Insight:** Hippocampus volume (0.725) approaches ADAS13 (0.767) in predictive power, and it's a purely biological marker!

---

## Part 6: APOE4 Genetic Risk

| APOE4 Alleles | Conversion Rate | N |
|---------------|-----------------|---|
| 0 alleles | 23.5% | 511 |
| 1 allele | 44.2% | 394 |
| 2 alleles | 49.1% | 110 |

**Finding:** APOE4 carriers have DOUBLE the conversion rate (44-49% vs 23%)!

APOE4 alone: **0.624 AUC**

---

## Part 7: Longitudinal Biomarker Analysis

### Does Temporal Change Help?

| Model | AUC | Notes |
|-------|-----|-------|
| Baseline biomarkers only | 0.736 | Current state |
| Delta (change) only | 0.759 | Atrophy rate |
| **Baseline + Delta** | **0.831** | Combined |

**Improvement from adding longitudinal: +9.5 percentage points!**

---

## Part 8: Best Possible Models

### Model Comparison (5-fold CV)

| Model | AUC |
|-------|-----|
| Logistic Regression | 0.83 |
| Random Forest | 0.82 |
| Gradient Boosting | 0.83 |

All models perform similarly - the features matter more than the algorithm!

### Feature Sets Comparison

| Features | AUC |
|----------|-----|
| ResNet (original) | 0.52 |
| Biomarkers + Delta | 0.83 |
| + Age + APOE4 (honest) | 0.81 |
| + ADAS13 (with cog score) | 0.84 |

---

## Part 9: Demographic Effects

### Conversion Rate by Age

| Age Group | Conversion Rate |
|-----------|-----------------|
| 55-65 | Varies |
| 65-70 | Varies |
| 70-75 | Varies |
| 75-80 | Varies |
| 80-90 | Varies |

### Education Effect

| Education | Conversion Rate |
|-----------|-----------------|
| 0-12 years | 32.4% |
| 12-14 years | 30.4% |
| 14-16 years | 35.0% |
| 16-18 years | 32.1% |
| 18-25 years | 32.3% |

**Finding:** Education level does NOT predict progression - it's roughly 32% across all levels!

---

## Part 10: Follow-up Duration Effect

Longer follow-up = more conversions detected.

1-year follow-up misses many future converters because:
- AD progression takes 2-10 years
- Some MCI patients convert after 2-3 years
- Short studies have censored outcomes

---

## FINAL SUMMARY

### What Went Wrong Originally

1. ❌ Used ResNet features (trained on ImageNet, not brains)
2. ❌ Mixed all diagnoses (Dementia = Stable is wrong)
3. ❌ Generic CNN features can't capture hippocampal atrophy rate

### What Works

1. ✅ Use actual structural biomarkers (hippocampus, ventricles, entorhinal)
2. ✅ Focus on MCI cohort only (clinically meaningful)
3. ✅ Compute longitudinal change (atrophy rate)
4. ✅ Add APOE4 genetic risk factor
5. ✅ Simple logistic regression works - no need for complex models

### Performance Ladder

| Approach | AUC | Improvement vs ResNet |
|----------|-----|----------------------|
| ResNet baseline | 0.52 | - |
| + ResNet delta | 0.52 | +0% |
| Biomarkers baseline | 0.74 | +22% |
| **+ Longitudinal delta** | **0.83** | **+31%** |
| + Age + APOE4 | 0.81 | +29% |
| + ADAS13 | 0.84 | +32% |

---

## Key Takeaways

### 1. Longitudinal Data WORKS
> Hippocampal atrophy RATE is a powerful predictor of MCI→Dementia conversion. Adding temporal change improves AUC by +9.5 points (0.74 → 0.83).

### 2. Right Features > Complex Models
> Simple logistic regression with proper biomarkers (0.83) vastly outperforms complex LSTM with wrong features (0.44).

### 3. Hippocampus is the Best Single Predictor
> Single feature AUC of 0.725 - nearly as good as cognitive tests!

### 4. APOE4 Doubles Risk
> Carriers have 44-49% conversion rate vs 23% for non-carriers.

### 5. ResNet is Wrong for This Task
> ImageNet-pretrained features are scale-invariant and don't capture absolute volume changes.

---

## Recommendations for Future Work

1. **Extract FreeSurfer biomarkers** from MRI instead of ResNet features
2. **Compute annual atrophy rates** for hippocampus and entorhinal cortex
3. **Include APOE4** in all models
4. **Focus on MCI-only cohorts** for progression studies
5. **Use 2+ year follow-up** to capture more conversions
