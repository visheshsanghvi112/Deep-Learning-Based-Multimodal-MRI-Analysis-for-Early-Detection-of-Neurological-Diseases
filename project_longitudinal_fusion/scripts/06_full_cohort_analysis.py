"""
Full MCI Cohort Biomarker Analysis
==================================
Uses ALL available MCI subjects with longitudinal biomarker data.
Target: Confirm 0.83 AUC with proper 5-fold CV.

This is the DEFINITIVE biomarker-only experiment.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
ADNIMERGE_PATH = r"D:\discs\data\ADNI\ADNIMERGE_23Dec2025.csv"
OUTPUT_DIR = Path(r"D:\discs\project_longitudinal_fusion\results\full_cohort")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Biomarkers to use
VOLUMETRIC = ['Hippocampus', 'Ventricles', 'Entorhinal', 'MidTemp', 'Fusiform', 'WholeBrain']
DEMOGRAPHIC = ['AGE', 'PTGENDER', 'APOE4']

def load_mci_cohort():
    """Load all MCI subjects with longitudinal data."""
    print("="*60)
    print("LOADING FULL MCI COHORT")
    print("="*60)
    
    df = pd.read_csv(ADNIMERGE_PATH, low_memory=False)
    print(f"Total ADNIMERGE rows: {len(df)}")
    print(f"Total subjects: {df['PTID'].nunique()}")
    
    # Get subjects with MCI at baseline
    baseline = df[df['VISCODE'] == 'bl']
    mci_ptids = baseline[baseline['DX_bl'].str.contains('MCI', na=False)]['PTID'].unique()
    print(f"\nMCI subjects at baseline: {len(mci_ptids)}")
    
    # Filter to MCI subjects only
    mci_df = df[df['PTID'].isin(mci_ptids)].copy()
    
    # Map visit codes to numeric
    visit_map = {'bl': 0, 'm06': 6, 'm12': 12, 'm18': 18, 'm24': 24, 
                 'm36': 36, 'm48': 48, 'm60': 60, 'm72': 72}
    mci_df['visit_num'] = mci_df['VISCODE'].map(visit_map)
    mci_df = mci_df[mci_df['visit_num'].notna()]
    
    return mci_df, mci_ptids

def determine_progression(subj_df):
    """Determine if subject progressed to dementia."""
    # Sort by visit
    subj_df = subj_df.sort_values('visit_num')
    
    # Get diagnosis at each visit
    dx_col = 'DX' if 'DX' in subj_df.columns else 'DX_bl'
    
    diagnoses = subj_df[dx_col].dropna().values
    if len(diagnoses) < 2:
        return None
    
    first_dx = str(diagnoses[0]).upper()
    last_dx = str(diagnoses[-1]).upper()
    
    # Check for progression to dementia
    if 'MCI' in first_dx:
        if 'DEMENTIA' in last_dx or 'AD' in last_dx:
            return 1  # Converter
        elif 'MCI' in last_dx or 'CN' in last_dx:
            return 0  # Stable
    
    return None  # Unclear

def extract_longitudinal_biomarkers(subj_df):
    """Extract baseline, followup, and delta biomarkers."""
    subj_df = subj_df.sort_values('visit_num')
    
    # Need at least 2 visits
    if len(subj_df) < 2:
        return None
    
    baseline = subj_df.iloc[0]
    followup = subj_df.iloc[-1]
    
    # Check for required biomarkers at baseline
    baseline_values = {}
    for vol in VOLUMETRIC:
        val = baseline.get(vol)
        if pd.isna(val):
            return None
        baseline_values[vol] = val
    
    # Check for required biomarkers at followup
    followup_values = {}
    for vol in VOLUMETRIC:
        val = followup.get(vol)
        if pd.isna(val):
            return None
        followup_values[vol] = val
    
    # Demographics (from baseline)
    age = baseline.get('AGE')
    if pd.isna(age):
        return None
    
    sex = 1.0 if baseline.get('PTGENDER') == 'Male' else 0.0
    apoe4 = baseline.get('APOE4', 0)
    if pd.isna(apoe4):
        apoe4 = 0
    
    # Calculate deltas
    delta_values = {f"{vol}_delta": followup_values[vol] - baseline_values[vol] 
                    for vol in VOLUMETRIC}
    
    # Time between visits (for rate normalization)
    time_diff = followup['visit_num'] - baseline['visit_num']
    
    # Combine all features
    features = {
        # Baseline volumetric (6)
        'bl_hippocampus': baseline_values['Hippocampus'],
        'bl_ventricles': baseline_values['Ventricles'],
        'bl_entorhinal': baseline_values['Entorhinal'],
        'bl_midtemp': baseline_values['MidTemp'],
        'bl_fusiform': baseline_values['Fusiform'],
        'bl_wholebrain': baseline_values['WholeBrain'],
        
        # Demographics (3)
        'age': age,
        'sex': sex,
        'apoe4': apoe4,
        
        # Followup volumetric (6)
        'fu_hippocampus': followup_values['Hippocampus'],
        'fu_ventricles': followup_values['Ventricles'],
        'fu_entorhinal': followup_values['Entorhinal'],
        'fu_midtemp': followup_values['MidTemp'],
        'fu_fusiform': followup_values['Fusiform'],
        'fu_wholebrain': followup_values['WholeBrain'],
        
        # Delta (6)
        'delta_hippocampus': delta_values['Hippocampus_delta'],
        'delta_ventricles': delta_values['Ventricles_delta'],
        'delta_entorhinal': delta_values['Entorhinal_delta'],
        'delta_midtemp': delta_values['MidTemp_delta'],
        'delta_fusiform': delta_values['Fusiform_delta'],
        'delta_wholebrain': delta_values['WholeBrain_delta'],
        
        'time_diff_months': time_diff
    }
    
    return features

def prepare_dataset(mci_df, mci_ptids):
    """Prepare full dataset."""
    print("\nExtracting longitudinal biomarkers...")
    
    samples = []
    
    for ptid in mci_ptids:
        subj_df = mci_df[mci_df['PTID'] == ptid]
        
        # Get progression label
        label = determine_progression(subj_df)
        if label is None:
            continue
        
        # Extract biomarkers
        features = extract_longitudinal_biomarkers(subj_df)
        if features is None:
            continue
        
        features['subject_id'] = ptid
        features['label'] = label
        samples.append(features)
    
    df = pd.DataFrame(samples)
    
    print(f"\nTotal subjects with complete data: {len(df)}")
    print(f"  Stable (0): {sum(df['label'] == 0)}")
    print(f"  Converter (1): {sum(df['label'] == 1)}")
    print(f"  Conversion rate: {sum(df['label'] == 1) / len(df) * 100:.1f}%")
    
    return df

def run_cross_validation(X, y, model, n_folds=5):
    """Run stratified k-fold cross-validation."""
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_aucs = []
    fold_accs = []
    all_preds = []
    all_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Train
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)
        
        # Predict
        y_prob = model_clone.predict_proba(X_val)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        fold_aucs.append(roc_auc_score(y_val, y_prob))
        fold_accs.append(accuracy_score(y_val, y_pred))
        
        all_preds.extend(y_prob)
        all_labels.extend(y_val)
    
    # Statistics
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    
    # 95% CI
    ci = stats.t.interval(0.95, len(fold_aucs)-1, loc=mean_auc, scale=stats.sem(fold_aucs))
    
    return {
        'fold_aucs': fold_aucs,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'mean_accuracy': np.mean(fold_accs),
        'overall_auc': roc_auc_score(all_labels, all_preds)
    }

def main():
    print("\n" + "="*70)
    print("  FULL MCI COHORT BIOMARKER ANALYSIS")
    print("  Goal: Confirm 0.83 AUC with proper 5-fold CV")
    print("="*70 + "\n")
    
    # Load data
    mci_df, mci_ptids = load_mci_cohort()
    
    # Prepare dataset
    data_df = prepare_dataset(mci_df, mci_ptids)
    
    # Feature columns (21 total)
    feature_cols = [
        'bl_hippocampus', 'bl_ventricles', 'bl_entorhinal', 
        'bl_midtemp', 'bl_fusiform', 'bl_wholebrain',
        'age', 'sex', 'apoe4',
        'fu_hippocampus', 'fu_ventricles', 'fu_entorhinal',
        'fu_midtemp', 'fu_fusiform', 'fu_wholebrain',
        'delta_hippocampus', 'delta_ventricles', 'delta_entorhinal',
        'delta_midtemp', 'delta_fusiform', 'delta_wholebrain'
    ]
    
    X = data_df[feature_cols].values
    y = data_df['label'].values
    
    print("\n" + "="*60)
    print("RUNNING 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    # Models to test
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, 
            class_weight='balanced', n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        result = run_cross_validation(X, y, model, n_folds=5)
        results[name] = result
        
        print(f"  Fold AUCs: {[f'{x:.3f}' for x in result['fold_aucs']]}")
        print(f"  Mean AUC: {result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
        print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        print(f"  Mean Accuracy: {result['mean_accuracy']:.4f}")
    
    # Feature ablation - test with different feature sets
    print("\n" + "="*60)
    print("FEATURE ABLATION STUDY")
    print("="*60)
    
    feature_sets = {
        'baseline_only': ['bl_hippocampus', 'bl_ventricles', 'bl_entorhinal', 
                          'bl_midtemp', 'bl_fusiform', 'bl_wholebrain',
                          'age', 'sex', 'apoe4'],
        'delta_only': ['delta_hippocampus', 'delta_ventricles', 'delta_entorhinal',
                       'delta_midtemp', 'delta_fusiform', 'delta_wholebrain'],
        'baseline_plus_delta': ['bl_hippocampus', 'bl_ventricles', 'bl_entorhinal', 
                                'bl_midtemp', 'bl_fusiform', 'bl_wholebrain',
                                'age', 'sex', 'apoe4',
                                'delta_hippocampus', 'delta_ventricles', 'delta_entorhinal',
                                'delta_midtemp', 'delta_fusiform', 'delta_wholebrain'],
        'all_21_features': feature_cols
    }
    
    ablation_results = {}
    best_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    
    for set_name, cols in feature_sets.items():
        X_subset = data_df[cols].values
        result = run_cross_validation(X_subset, y, best_model, n_folds=5)
        ablation_results[set_name] = result
        
        print(f"\n{set_name} ({len(cols)} features):")
        print(f"  Mean AUC: {result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
        print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    
    # Summary
    print("\n" + "="*70)
    print("  FINAL RESULTS SUMMARY")
    print("="*70)
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['mean_auc'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   AUC: {best_result['mean_auc']:.4f} ± {best_result['std_auc']:.4f}")
    print(f"   95% CI: [{best_result['ci_lower']:.4f}, {best_result['ci_upper']:.4f}]")
    print(f"   Sample size: {len(data_df)} subjects")
    
    # Compare with previous
    print(f"\n📊 COMPARISON:")
    print(f"   Previous (301 subjects): 0.71 AUC")
    print(f"   Current ({len(data_df)} subjects): {best_result['mean_auc']:.4f} AUC")
    
    if best_result['mean_auc'] > 0.83:
        print(f"\n   ✅ Beat target 0.83 AUC!")
    else:
        print(f"\n   Target was 0.83 AUC (difference: {0.83 - best_result['mean_auc']:.4f})")
    
    # Save results
    output = {
        'n_subjects': len(data_df),
        'n_converters': int(sum(y == 1)),
        'n_stable': int(sum(y == 0)),
        'conversion_rate': float(sum(y == 1) / len(y)),
        'model_results': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else 
                              [float(x) for x in vv] if isinstance(vv, list) else vv
                              for kk, vv in v.items()} 
                          for k, v in results.items()},
        'ablation_results': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else 
                                  [float(x) for x in vv] if isinstance(vv, list) else vv
                                  for kk, vv in v.items()} 
                              for k, v in ablation_results.items()}
    }
    
    with open(OUTPUT_DIR / 'full_cohort_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to: {OUTPUT_DIR / 'full_cohort_results.json'}")
    
    # Save data for later use
    data_df.to_csv(OUTPUT_DIR / 'full_cohort_data.csv', index=False)
    print(f"✅ Data saved to: {OUTPUT_DIR / 'full_cohort_data.csv'}")

if __name__ == "__main__":
    main()
