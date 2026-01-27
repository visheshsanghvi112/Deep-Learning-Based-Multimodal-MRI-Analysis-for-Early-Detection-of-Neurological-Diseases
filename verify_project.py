"""
COMPREHENSIVE PROJECT VERIFICATION
===================================
Deep verification of all claims in the project against actual data files.
"""

import json
import os
from pathlib import Path

print("="*80)
print("COMPREHENSIVE PROJECT VERIFICATION")
print("="*80)

# Track all issues
issues = []
verifications = []

def verify(claim, actual, expected, tolerance=0.001):
    """Verify a claim against actual data."""
    if abs(actual - expected) <= tolerance:
        verifications.append(f"✅ {claim}: {actual:.3f} (expected {expected:.3f})")
        return True
    else:
        issues.append(f"❌ {claim}: {actual:.3f} ≠ {expected:.3f}")
        return False

# ============================================================================
# 1. LEVEL-MAX VERIFICATION (0.808 AUC claim)
# ============================================================================
print("\n" + "="*80)
print("1. LEVEL-MAX RESULTS VERIFICATION")
print("="*80)

level_max_path = r"d:\discs\project_adni\results\level_max\results.json"
with open(level_max_path) as f:
    level_max = json.load(f)

late_auc = level_max['Late_Fusion']['AUC']
attn_auc = level_max['Attention_Fusion']['AUC']
mri_auc = level_max['MRI_Only']['AUC']

verify("Level-MAX Late Fusion AUC", late_auc, 0.808, tolerance=0.001)
verify("Level-MAX Attention Fusion AUC", attn_auc, 0.808, tolerance=0.001)
verify("Level-MAX MRI-Only AUC", mri_auc, 0.643, tolerance=0.001)

# ============================================================================
# 2. LEVEL-1 VERIFICATION (0.60 AUC claim)
# ============================================================================
print("\n" + "="*80)
print("2. LEVEL-1 RESULTS VERIFICATION")
print("="*80)

level1_path = r"d:\discs\project_adni\results\level1\metrics.json"
with open(level1_path) as f:
    level1 = json.load(f)

l1_mri = level1['results']['MRI-Only']['auc']
l1_late = level1['results']['Late Fusion']['auc']
l1_attn = level1['results']['Attention Fusion']['auc']

verify("Level-1 MRI-Only AUC", l1_mri, 0.583, tolerance=0.001)
verify("Level-1 Late Fusion AUC", l1_late, 0.598, tolerance=0.001)
verify("Level-1 Attention Fusion AUC", l1_attn, 0.590, tolerance=0.001)

# ============================================================================
# 3. LEVEL-2 VERIFICATION (0.988 AUC claim - circular)
# ============================================================================
print("\n" + "="*80)
print("3. LEVEL-2 RESULTS VERIFICATION (Circular)")
print("="*80)

level2_path = r"d:\discs\project_adni\results\level2\metrics.json"
with open(level2_path) as f:
    level2 = json.load(f)

l2_late = level2['results']['Late Fusion']['auc']
l2_attn = level2['results']['Attention Fusion']['auc']

verify("Level-2 Late Fusion AUC (circular)", l2_late, 0.988, tolerance=0.001)
verify("Level-2 Attention Fusion AUC (circular)", l2_attn, 0.985, tolerance=0.001)

# ============================================================================
# 4. LONGITUDINAL VERIFICATION (0.848 AUC claim)
# ============================================================================
print("\n" + "="*80)
print("4. LONGITUDINAL RESULTS VERIFICATION")
print("="*80)

long_path = r"d:\discs\project_longitudinal_fusion\results\full_cohort\full_cohort_results.json"
with open(long_path) as f:
    long_data = json.load(f)

rf_auc = long_data['model_results']['RandomForest']['mean_auc']
lr_auc = long_data['model_results']['LogisticRegression']['mean_auc']
gb_auc = long_data['model_results']['GradientBoosting']['mean_auc']

verify("Longitudinal Random Forest AUC", rf_auc, 0.848, tolerance=0.001)
verify("Longitudinal Logistic Regression AUC", lr_auc, 0.836, tolerance=0.001)
verify("Longitudinal Gradient Boosting AUC", gb_auc, 0.832, tolerance=0.001)

# Check sample size
n_subjects = long_data['n_subjects']
n_converters = long_data['n_converters']
n_stable = long_data['n_stable']

print(f"\n📊 Longitudinal Cohort:")
print(f"   Total subjects: {n_subjects}")
print(f"   Converters: {n_converters} ({n_converters/n_subjects*100:.1f}%)")
print(f"   Stable: {n_stable} ({n_stable/n_subjects*100:.1f}%)")

if n_subjects == 341:
    verifications.append(f"✅ Longitudinal sample size: {n_subjects} (expected 341)")
else:
    issues.append(f"❌ Longitudinal sample size: {n_subjects} ≠ 341")

# ============================================================================
# 5. CONSISTENCY CHECKS
# ============================================================================
print("\n" + "="*80)
print("5. CONSISTENCY CHECKS")
print("="*80)

# Check that Level-MAX is better than Level-1
if late_auc > l1_late:
    verifications.append(f"✅ Level-MAX ({late_auc:.3f}) > Level-1 ({l1_late:.3f})")
else:
    issues.append(f"❌ Level-MAX should be > Level-1")

# Check that Level-2 is better than Level-MAX (circular should be best)
if l2_late > late_auc:
    verifications.append(f"✅ Level-2 ({l2_late:.3f}) > Level-MAX ({late_auc:.3f}) - circular is best")
else:
    issues.append(f"❌ Level-2 should be > Level-MAX")

# Check that Longitudinal is best honest result
if rf_auc > late_auc:
    verifications.append(f"✅ Longitudinal ({rf_auc:.3f}) > Level-MAX ({late_auc:.3f}) - best honest result")
else:
    issues.append(f"❌ Longitudinal should be best honest result")

# ============================================================================
# 6. IMPROVEMENT CALCULATIONS
# ============================================================================
print("\n" + "="*80)
print("6. IMPROVEMENT CALCULATIONS")
print("="*80)

# Level-MAX improvement over MRI-Only
lmax_improvement = ((late_auc - mri_auc) / mri_auc) * 100
print(f"Level-MAX improvement: {lmax_improvement:.1f}% (claimed: +16.5%)")
if abs(lmax_improvement - 16.5) < 1.0:
    verifications.append(f"✅ Level-MAX improvement: {lmax_improvement:.1f}%")
else:
    issues.append(f"❌ Level-MAX improvement: {lmax_improvement:.1f}% ≠ 16.5%")

# Longitudinal improvement over baseline biomarkers
baseline_bio = long_data['ablation_results']['baseline_only']['mean_auc']
long_improvement = ((rf_auc - baseline_bio) / baseline_bio) * 100
print(f"Longitudinal improvement: {long_improvement:.1f}% (baseline: {baseline_bio:.3f})")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

print(f"\n✅ PASSED: {len(verifications)}")
for v in verifications:
    print(f"   {v}")

if issues:
    print(f"\n❌ ISSUES FOUND: {len(issues)}")
    for issue in issues:
        print(f"   {issue}")
else:
    print(f"\n🎉 NO ISSUES FOUND - ALL CLAIMS VERIFIED!")

print("\n" + "="*80)
if len(issues) == 0:
    print("✅ PROJECT INTEGRITY: 100% VERIFIED")
else:
    print(f"⚠️ PROJECT INTEGRITY: {len(verifications)}/{len(verifications)+len(issues)} checks passed")
print("="*80)
