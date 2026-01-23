"""
Data Preprocessing Module
=========================
Load and preprocess data from ADNI dataset.
Combines ResNet features with clinical biomarkers.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ADNIMERGE_PATH, ADNIMERGE_BACKUP,
    LONGITUDINAL_FEATURES_PATH, LONGITUDINAL_SPLITS_PATH,
    DATA_DIR, VOLUMETRIC_BIOMARKERS, DEMOGRAPHIC_FEATURES,
    RANDOM_SEED
)


def load_adnimerge() -> pd.DataFrame:
    """Load ADNIMERGE clinical data."""
    # Try primary path first
    if os.path.exists(ADNIMERGE_PATH):
        df = pd.read_csv(ADNIMERGE_PATH)
    elif os.path.exists(ADNIMERGE_BACKUP):
        df = pd.read_csv(ADNIMERGE_BACKUP)
    else:
        raise FileNotFoundError(
            f"ADNIMERGE not found at:\n"
            f"  - {ADNIMERGE_PATH}\n"
            f"  - {ADNIMERGE_BACKUP}"
        )
    
    # Standardize subject ID column
    if 'PTID' in df.columns:
        df['subject_id'] = df['PTID']
    
    # Map VISCODE to visit numbers
    visit_map = {
        'bl': 0, 'm06': 6, 'm12': 12, 'm18': 18,
        'm24': 24, 'm36': 36, 'm48': 48, 'm60': 60, 'm72': 72
    }
    df['visit_num'] = df['VISCODE'].map(visit_map)
    df = df[df['visit_num'].notna()]
    
    return df


def extract_biomarkers_for_subject(
    subject_id: str, 
    adni_df: pd.DataFrame
) -> Optional[Dict]:
    """
    Extract longitudinal biomarkers for a single subject.
    
    Returns:
        Dictionary with baseline, followup, delta biomarkers
        or None if insufficient data.
    """
    subj_df = adni_df[adni_df['subject_id'] == subject_id].copy()
    
    if len(subj_df) < 2:
        return None
    
    subj_df = subj_df.sort_values('visit_num')
    baseline = subj_df.iloc[0]
    followup = subj_df.iloc[-1]
    
    def get_volumetric(row):
        return {
            'hippocampus': row.get('Hippocampus', np.nan),
            'ventricles': row.get('Ventricles', np.nan),
            'entorhinal': row.get('Entorhinal', np.nan),
            'midtemp': row.get('MidTemp', np.nan),
            'fusiform': row.get('Fusiform', np.nan),
            'wholebrain': row.get('WholeBrain', np.nan)
        }
    
    baseline_vol = get_volumetric(baseline)
    followup_vol = get_volumetric(followup)
    
    # Check for missing volumetric data
    if any(np.isnan(list(baseline_vol.values()))):
        return None
    if any(np.isnan(list(followup_vol.values()))):
        return None
    
    # Demographics (from baseline only)
    age = baseline.get('AGE', np.nan)
    sex = 1.0 if baseline.get('PTGENDER') == 'Male' else 0.0
    apoe4 = baseline.get('APOE4', 0)
    
    if np.isnan(age):
        return None
    
    # Calculate deltas
    delta = {
        f'{k}_delta': followup_vol[k] - baseline_vol[k]
        for k in baseline_vol.keys()
    }
    
    # Time difference for rate normalization
    time_diff = followup['visit_num'] - baseline['visit_num']
    
    return {
        'baseline_volumetric': baseline_vol,
        'baseline_demographics': {'age': age, 'sex': sex, 'apoe4': apoe4},
        'followup_volumetric': followup_vol,
        'delta': delta,
        'time_diff_months': time_diff
    }


def load_resnet_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load precomputed ResNet features from longitudinal project.
    
    Returns:
        features, subject_ids, labels, splits
    """
    data = np.load(LONGITUDINAL_FEATURES_PATH, allow_pickle=True)
    return (
        data['features'],
        data['subject_ids'],
        data['labels'],
        data['splits']
    )


def aggregate_resnet_by_subject(
    features: np.ndarray,
    subject_ids: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray
) -> Dict[str, Dict]:
    """
    Aggregate per-scan ResNet features to per-subject features.
    
    For each subject:
    - baseline_resnet: First scan features (512-dim)
    - followup_resnet: Last scan features (512-dim)  
    - delta_resnet: Last - First features (512-dim)
    """
    unique_subjects = np.unique(subject_ids)
    aggregated = {}
    
    for subj in unique_subjects:
        mask = subject_ids == subj
        subj_features = features[mask]
        subj_labels = labels[mask]
        subj_splits = splits[mask]
        
        if len(subj_features) < 2:
            continue
        
        # Baseline = first, Followup = last
        baseline_resnet = subj_features[0]
        followup_resnet = subj_features[-1]
        delta_resnet = followup_resnet - baseline_resnet
        
        aggregated[subj] = {
            'baseline_resnet': baseline_resnet,
            'followup_resnet': followup_resnet,
            'delta_resnet': delta_resnet,
            'label': int(subj_labels[0]),
            'split': str(subj_splits[0])
        }
    
    return aggregated


def load_and_prepare_data(
    force_reload: bool = False
) -> Tuple[Dict, Dict, Dict]:
    """
    Main data loading function. Combines ResNet and biomarker features.
    
    Returns:
        train_data, test_data, scalers
    """
    cache_path = DATA_DIR / "fusion_dataset_v2.npz"
    
    if cache_path.exists() and not force_reload:
        print(f"Loading cached data from {cache_path}")
        return _load_cached_data(cache_path)
    
    print("="*60)
    print("PREPARING FUSION DATASET")
    print("="*60)
    
    # Load components
    print("\n[1/4] Loading ADNIMERGE clinical data...")
    adni_df = load_adnimerge()
    print(f"    Loaded {len(adni_df)} rows")
    
    print("\n[2/4] Loading ResNet features...")
    features, subject_ids, labels, splits = load_resnet_features()
    print(f"    Loaded {len(features)} scans from {len(np.unique(subject_ids))} subjects")
    
    print("\n[3/4] Aggregating ResNet features by subject...")
    resnet_data = aggregate_resnet_by_subject(features, subject_ids, labels, splits)
    print(f"    Aggregated to {len(resnet_data)} subjects with ≥2 scans")
    
    print("\n[4/4] Extracting biomarkers and merging...")
    
    # Prepare final dataset
    train_samples = []
    test_samples = []
    
    for subject_id, resnet_info in resnet_data.items():
        biomarkers = extract_biomarkers_for_subject(subject_id, adni_df)
        
        if biomarkers is None:
            continue
        
        # Assemble feature vectors
        sample = {
            'subject_id': subject_id,
            'label': resnet_info['label'],
            
            # ResNet features (512 each)
            'baseline_resnet': resnet_info['baseline_resnet'],
            'followup_resnet': resnet_info['followup_resnet'],
            'delta_resnet': resnet_info['delta_resnet'],
            
            # Baseline biomarkers (9 features)
            'baseline_bio': np.array([
                biomarkers['baseline_volumetric']['hippocampus'],
                biomarkers['baseline_volumetric']['ventricles'],
                biomarkers['baseline_volumetric']['entorhinal'],
                biomarkers['baseline_volumetric']['midtemp'],
                biomarkers['baseline_volumetric']['fusiform'],
                biomarkers['baseline_volumetric']['wholebrain'],
                biomarkers['baseline_demographics']['age'],
                biomarkers['baseline_demographics']['sex'],
                biomarkers['baseline_demographics']['apoe4']
            ]),
            
            # Followup biomarkers (6 features)
            'followup_bio': np.array([
                biomarkers['followup_volumetric']['hippocampus'],
                biomarkers['followup_volumetric']['ventricles'],
                biomarkers['followup_volumetric']['entorhinal'],
                biomarkers['followup_volumetric']['midtemp'],
                biomarkers['followup_volumetric']['fusiform'],
                biomarkers['followup_volumetric']['wholebrain']
            ]),
            
            # Delta biomarkers (6 features)
            'delta_bio': np.array([
                biomarkers['delta']['hippocampus_delta'],
                biomarkers['delta']['ventricles_delta'],
                biomarkers['delta']['entorhinal_delta'],
                biomarkers['delta']['midtemp_delta'],
                biomarkers['delta']['fusiform_delta'],
                biomarkers['delta']['wholebrain_delta']
            ]),
            
            'time_diff': biomarkers['time_diff_months']
        }
        
        if resnet_info['split'] == 'train':
            train_samples.append(sample)
        else:
            test_samples.append(sample)
    
    print(f"\n    Final dataset:")
    print(f"    - Train: {len(train_samples)} subjects")
    print(f"    - Test:  {len(test_samples)} subjects")
    print(f"    - Total: {len(train_samples) + len(test_samples)} subjects")
    
    # Convert to arrays
    train_data = _samples_to_arrays(train_samples)
    test_data = _samples_to_arrays(test_samples)
    
    # Standardize features
    print("\n    Standardizing features...")
    scalers = {}
    for key in ['baseline_resnet', 'followup_resnet', 'delta_resnet',
                'baseline_bio', 'followup_bio', 'delta_bio']:
        scaler = StandardScaler()
        train_data[key] = scaler.fit_transform(train_data[key])
        test_data[key] = scaler.transform(test_data[key])
        scalers[key] = scaler
    
    # Cache
    print(f"\n    Caching to {cache_path}...")
    _save_cached_data(cache_path, train_data, test_data)
    
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE")
    print("="*60)
    
    return train_data, test_data, scalers


def _samples_to_arrays(samples: List[Dict]) -> Dict:
    """Convert list of samples to dictionary of arrays."""
    return {
        'subject_ids': np.array([s['subject_id'] for s in samples]),
        'labels': np.array([s['label'] for s in samples]),
        'baseline_resnet': np.stack([s['baseline_resnet'] for s in samples]),
        'followup_resnet': np.stack([s['followup_resnet'] for s in samples]),
        'delta_resnet': np.stack([s['delta_resnet'] for s in samples]),
        'baseline_bio': np.stack([s['baseline_bio'] for s in samples]),
        'followup_bio': np.stack([s['followup_bio'] for s in samples]),
        'delta_bio': np.stack([s['delta_bio'] for s in samples]),
        'time_diff': np.array([s['time_diff'] for s in samples])
    }


def _save_cached_data(path: Path, train_data: Dict, test_data: Dict):
    """Save processed data to cache."""
    np.savez_compressed(
        path,
        # Train
        train_subject_ids=train_data['subject_ids'],
        train_labels=train_data['labels'],
        train_baseline_resnet=train_data['baseline_resnet'],
        train_followup_resnet=train_data['followup_resnet'],
        train_delta_resnet=train_data['delta_resnet'],
        train_baseline_bio=train_data['baseline_bio'],
        train_followup_bio=train_data['followup_bio'],
        train_delta_bio=train_data['delta_bio'],
        train_time_diff=train_data['time_diff'],
        # Test
        test_subject_ids=test_data['subject_ids'],
        test_labels=test_data['labels'],
        test_baseline_resnet=test_data['baseline_resnet'],
        test_followup_resnet=test_data['followup_resnet'],
        test_delta_resnet=test_data['delta_resnet'],
        test_baseline_bio=test_data['baseline_bio'],
        test_followup_bio=test_data['followup_bio'],
        test_delta_bio=test_data['delta_bio'],
        test_time_diff=test_data['time_diff']
    )


def _load_cached_data(path: Path) -> Tuple[Dict, Dict, Dict]:
    """Load processed data from cache."""
    data = np.load(path, allow_pickle=True)
    
    train_data = {
        'subject_ids': data['train_subject_ids'],
        'labels': data['train_labels'],
        'baseline_resnet': data['train_baseline_resnet'],
        'followup_resnet': data['train_followup_resnet'],
        'delta_resnet': data['train_delta_resnet'],
        'baseline_bio': data['train_baseline_bio'],
        'followup_bio': data['train_followup_bio'],
        'delta_bio': data['train_delta_bio'],
        'time_diff': data['train_time_diff']
    }
    
    test_data = {
        'subject_ids': data['test_subject_ids'],
        'labels': data['test_labels'],
        'baseline_resnet': data['test_baseline_resnet'],
        'followup_resnet': data['test_followup_resnet'],
        'delta_resnet': data['test_delta_resnet'],
        'baseline_bio': data['test_baseline_bio'],
        'followup_bio': data['test_followup_bio'],
        'delta_bio': data['test_delta_bio'],
        'time_diff': data['test_time_diff']
    }
    
    # Recompute scalers from train data
    scalers = {}
    for key in ['baseline_resnet', 'followup_resnet', 'delta_resnet',
                'baseline_bio', 'followup_bio', 'delta_bio']:
        scaler = StandardScaler()
        scaler.fit(train_data[key])
        scalers[key] = scaler
    
    return train_data, test_data, scalers


if __name__ == "__main__":
    # Test
    train_data, test_data, scalers = load_and_prepare_data(force_reload=True)
    
    print("\nData shapes:")
    print(f"  Train labels: {train_data['labels'].shape}")
    print(f"  Train baseline_resnet: {train_data['baseline_resnet'].shape}")
    print(f"  Train baseline_bio: {train_data['baseline_bio'].shape}")
