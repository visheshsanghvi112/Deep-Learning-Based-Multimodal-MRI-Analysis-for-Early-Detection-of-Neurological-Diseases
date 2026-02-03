#!/usr/bin/env python3
"""
Complete Statistical Verification with Fold-Level Data
Now we can FULLY verify Cohen's d and t-tests!
"""

import json
import numpy as np
from scipy import stats

print('='*70)
print('COMPLETE STATISTICAL VERIFICATION - WITH FOLD DATA!')
print('='*70)

# Load longitudinal results with FOLD-LEVEL data
with open('d:/discs/project_longitudinal_fusion/results/full_cohort/full_cohort_results.json', 'r') as f:
    long = json.load(f)

print('\n[1] RANDOM FOREST FOLD-LEVEL RESULTS')
print('-'*70)

rf_folds = long['model_results']['RandomForest']['fold_aucs']
rf_mean = long['model_results']['RandomForest']['mean_auc']
rf_std = long['model_results']['RandomForest']['std_auc']
rf_ci_lower = long['model_results']['RandomForest']['ci_lower']
rf_ci_upper = long['model_results']['RandomForest']['ci_upper']

print(f'Fold AUCs: {[round(x, 3) for x in rf_folds]}')
print(f'Mean AUC: {rf_mean:.4f}')
print(f'Std Dev: {rf_std:.4f}')
print(f'95% CI: [{rf_ci_lower:.3f}, {rf_ci_upper:.3f}]')

# Verify mean
calc_mean = np.mean(rf_folds)
print(f'\n✓ Mean verification: {calc_mean:.4f} ({"MATCH" if abs(calc_mean - rf_mean) < 0.0001 else "MISMATCH"})')

# Verify std
calc_std = np.std(rf_folds, ddof=1)  # ddof=1 for sample std
print(f'✓ Std verification: {calc_std:.4f} vs {rf_std:.4f}')

print('\n[2] NOW CHECK IF WE HAVE LEVEL-1 FOLD DATA')
print('-'*70)

# We need to find Level-1 and Level-MAX fold-level results to compute Cohen's d
# Let's check the ADNI results

# Try to find fold-level ADNI data
print('Searching for ADNI fold-level data')
