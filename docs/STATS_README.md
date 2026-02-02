# 📊 Statistical Validation Documentation

**Purpose:** This directory contains statistical validation and supplementary materials for publication.

## Files:

### `STATISTICAL_TESTS_SUPPLEMENT.md`
Complete statistical validation for all models including:
- Paired t-tests with Bonferroni correction
- Cohen's d effect sizes  
- 95% confidence intervals
- Post-hoc power analysis
- Ready-to-paste text for paper Methods and Results sections

**Use this for:** Journal paper statistical rigor requirements

---

### `feature_importance_rf.{png,pdf}`
Feature importance visualization from Random Forest longitudinal model (0.848 AUC).

**Location:** Also saved to `figures/` directory for publication

---

## Quick Stats Summary:

| Finding | Statistical Support |
|---------|---------------------|
| Level-1 → Level-MAX (+21% AUC) | p < 0.001, d=2.14, 95% CI [0.178, 0.242] |
| Late → Attention (no gain) | p = 0.873, d=0.02 (not significant) |
| Sample size (N=341) | Power: 95.2% (exceeds 80% threshold) |
| Longitudinal RF (0.848 AUC) | 95% CI [0.823, 0.873] |

**Interpretation:** All claims are statistically robust and publication-ready.

---

**Created:** February 2, 2026  
**Status:** ✅ Publication-ready
