"""
Step 1: Extract Biomarkers from ADNIMERGE
==========================================
This script extracts longitudinal biomarker data for the same subjects
used in project_longitudinal/ experiment.

SAFETY: This does NOT modify any existing files!

Output: data/biomarker_longitudinal.npz
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
ADNIMERGE_PATH = r"D:\discs\data\ADNI\ADNIMERGE_23Dec2025.csv"
LONGITUDINAL_SPLITS = r"D:\discs\project_longitudinal\data\processed\train_test_split.csv"
LONGITUDINAL_FEATURES = r"D:\discs\project_longitudinal\data\features\longitudinal_features.npz"
OUTPUT_PATH = r"D:\discs\project_biomarker_fusion\data\biomarker_longitudinal.npz"

# Biomarkers to extract
BIOMARKERS = [
    'Hippocampus',    # Best single predictor (0.725 AUC)
    'Ventricles',     # Enlargement marker
    'Entorhinal',     # Early atrophy (0.691 AUC)
    'MidTemp',        # Temporal lobe (0.678 AUC)
    'Fusiform',       # Face recognition (0.670 AUC)
    'WholeBrain',     # Total volume (0.604 AUC)
    'APOE4',          # Genetic risk
    'AGE',            # Demographics
    'PTGENDER'        # Sex
]

def load_splits():
    """Load train/test splits from longitudinal experiment."""
    print("Loading train/test splits...")
    splits_df = pd.read_csv(LONGITUDINAL_SPLITS)
    print(f"Loaded {len(splits_df)} subjects")
    return splits_df

def load_adnimerge():
    """Load ADNIMERGE clinical data."""
    print("\nLoading ADNIMERGE...")
    df = pd.read_csv(ADNIMERGE_PATH)
    print(f"Loaded {len(df)} rows")
    
    # Convert PTID to match format (002_S_0295)
    if 'PTID' in df.columns:
        df['subject_id'] = df['PTID']
    
    # Map VISCODE to visit numbers
    visit_map = {
        'bl': 0,
        'm06': 6,
        'm12': 12,
        'm18': 18,
        'm24': 24,
        'm36': 36,
        'm48': 48,
        'm60': 60,
        'm72': 72
    }
    
    df['visit_num'] = df['VISCODE'].map(visit_map)
    df = df[df['visit_num'].notna()]  # Keep only mapped visits
    
    return df

def extract_biomarkers_for_subject(subject_id, adni_df):
    """Extract baseline and followup biomarkers for one subject."""
    subj_df = adni_df[adni_df['subject_id'] == subject_id].copy()
    
    if len(subj_df) < 2:
        return None  # Need at least 2 visits
    
    # Sort by visit
    subj_df = subj_df.sort_values('visit_num')
    
    # Get baseline (first visit)
    baseline = subj_df.iloc[0]
    
    # Get follow-up (last visit)
    followup = subj_df.iloc[-1]
    
    # Extract features
    def get_features(row):
        features = {}
        features['hippocampus'] = row.get('Hippocampus', np.nan)
        features['ventricles'] = row.get('Ventricles', np.nan)
        features['entorhinal'] = row.get('Entorhinal', np.nan)
        features['midtemp'] = row.get('MidTemp', np.nan)
        features['fusiform'] = row.get('Fusiform', np.nan)
        features['wholebrain'] = row.get('WholeBrain', np.nan)
        features['apoe4'] = row.get('APOE4', np.nan)
        features['age'] = row.get('AGE', np.nan)
        features['sex'] = 1 if row.get('PTGENDER') == 'Male' else 0
        return features
    
    baseline_features = get_features(baseline)
    followup_features = get_features(followup)
    
    # Calculate deltas
    delta_features = {}
    delta_features['hippocampus_delta'] = followup_features['hippocampus'] - baseline_features['hippocampus']
    delta_features['ventricles_delta'] = followup_features['ventricles'] - baseline_features['ventricles']
    delta_features['entorhinal_delta'] = followup_features['entorhinal'] - baseline_features['entorhinal']
    delta_features['midtemp_delta'] = followup_features['midtemp'] - baseline_features['midtemp']
    delta_features['fusiform_delta'] = followup_features['fusiform'] - baseline_features['fusiform']
    delta_features['wholebrain_delta'] = followup_features['wholebrain'] - baseline_features['wholebrain']
    
    return {
        'baseline': baseline_features,
        'followup': followup_features,
        'delta': delta_features,
        'time_diff_months': followup['visit_num'] - baseline['visit_num']
    }

def main():
    """Main extraction pipeline."""
    print("="*60)
    print("BIOMARKER EXTRACTION - SAFE MODE")
    print("This will NOT modify any existing files!")
    print("="*60)
    
    # Load data
    splits_df = load_splits()
    adni_df = load_adnimerge()
    
    # Extract biomarkers for each subject
    print("\nExtracting biomarkers...")
    results = []
    
    for idx, row in splits_df.iterrows():
        subject_id = row['subject_id']
        split = row['split']
        label = row['label']
        
        biomarkers = extract_biomarkers_for_subject(subject_id, adni_df)
        
        if biomarkers is None:
            print(f"Warning: {subject_id} - insufficient data")
            continue
        
        # Check for missing values
        baseline = biomarkers['baseline']
        followup = biomarkers['followup']
        delta = biomarkers['delta']
        
        if any(np.isnan(list(baseline.values())[:6])):  # Missing key biomarkers (6 volumes)
            print(f"Warning: {subject_id} - missing baseline biomarkers")
            continue
        
        if any(np.isnan(list(followup.values())[:6])):
            print(f"Warning: {subject_id} - missing followup biomarkers")
            continue
        
        results.append({
            'subject_id': subject_id,
            'split': split,
            'label': label,
            **{f'baseline_{k}': v for k, v in baseline.items()},
            **{f'followup_{k}': v for k, v in followup.items()},
            **{f'delta_{k}': v for k, v in delta.items()},
            'time_diff_months': biomarkers['time_diff_months']
        })
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(splits_df)} subjects...")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    print(f"\nSuccessfully extracted biomarkers for {len(results_df)} subjects")
    print(f"Train: {sum(results_df['split'] == 'train')}")
    print(f"Test: {sum(results_df['split'] == 'test')}")
    
    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Convert to arrays for npz
    save_dict = {
        'subject_ids': results_df['subject_id'].values,
        'splits': results_df['split'].values,
        'labels': results_df['label'].values,
        
        # Baseline (9 features)
        'baseline_hippocampus': results_df['baseline_hippocampus'].values,
        'baseline_ventricles': results_df['baseline_ventricles'].values,
        'baseline_entorhinal': results_df['baseline_entorhinal'].values,
        'baseline_midtemp': results_df['baseline_midtemp'].values,
        'baseline_fusiform': results_df['baseline_fusiform'].values,
        'baseline_wholebrain': results_df['baseline_wholebrain'].values,
        'baseline_apoe4': results_df['baseline_apoe4'].values,
        'baseline_age': results_df['baseline_age'].values,
        'baseline_sex': results_df['baseline_sex'].values,
        
        # Followup (6 features)
        'followup_hippocampus': results_df['followup_hippocampus'].values,
        'followup_ventricles': results_df['followup_ventricles'].values,
        'followup_entorhinal': results_df['followup_entorhinal'].values,
        'followup_midtemp': results_df['followup_midtemp'].values,
        'followup_fusiform': results_df['followup_fusiform'].values,
        'followup_wholebrain': results_df['followup_wholebrain'].values,
        
        # Deltas (6 features)
        'delta_hippocampus': results_df['delta_hippocampus'].values,
        'delta_ventricles': results_df['delta_ventricles'].values,
        'delta_entorhinal': results_df['delta_entorhinal'].values,
        'delta_midtemp': results_df['delta_midtemp'].values,
        'delta_fusiform': results_df['delta_fusiform'].values,
        'delta_wholebrain': results_df['delta_wholebrain'].values,
        
        'time_diff_months': results_df['time_diff_months'].values
    }
    
    np.savez_compressed(OUTPUT_PATH, **save_dict)
    
    # Also save CSV for inspection
    csv_path = OUTPUT_PATH.replace('.npz', '.csv')
    results_df.to_csv(csv_path, index=False)
    
    print(f"✅ Saved: {OUTPUT_PATH}")
    print(f"✅ Saved: {csv_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print("\nBaseline Biomarkers (mean ± std):")
    print(f"  Hippocampus: {results_df['baseline_hippocampus'].mean():.2f} ± {results_df['baseline_hippocampus'].std():.2f}")
    print(f"  Ventricles:  {results_df['baseline_ventricles'].mean():.2f} ± {results_df['baseline_ventricles'].std():.2f}")
    print(f"  Entorhinal:  {results_df['baseline_entorhinal'].mean():.2f} ± {results_df['baseline_entorhinal'].std():.2f}")
    print(f"  APOE4:       {results_df['baseline_apoe4'].mean():.2f} (alleles)")
    print(f"  Age:         {results_df['baseline_age'].mean():.1f} ± {results_df['baseline_age'].std():.1f}")
    
    print("\nDelta (Change over time):")
    print(f"  Hippocampus: {results_df['delta_hippocampus'].mean():.3f} ± {results_df['delta_hippocampus'].std():.3f}")
    print(f"  Ventricles:  {results_df['delta_ventricles'].mean():.3f} ± {results_df['delta_ventricles'].std():.3f}")
    print(f"  Entorhinal:  {results_df['delta_entorhinal'].mean():.3f} ± {results_df['delta_entorhinal'].std():.3f}")
    
    print(f"\nMean follow-up time: {results_df['time_diff_months'].mean():.1f} months")
    
    print("\n✅ COMPLETE - No existing files were modified!")

if __name__ == '__main__':
    main()
