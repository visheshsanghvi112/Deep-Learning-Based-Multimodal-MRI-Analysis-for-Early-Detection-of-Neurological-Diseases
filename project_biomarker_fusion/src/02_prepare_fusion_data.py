"""
Step 2: Prepare Fusion Dataset
================================
Combines ResNet MRI features + Biomarkers for each subject.

Input:
  - project_longitudinal/data/features/longitudinal_features.npz (ResNet)
  - project_biomarker_fusion/data/biomarker_longitudinal.npz (Biomarkers)

Output:
  - data/fusion_dataset.npz

SAFETY: Only reads from existing files, creates new fusion dataset
"""

import os
import numpy as np
import pandas as pd

# Paths
RESNET_FEATURES = r"D:\discs\project_longitudinal\data\features\longitudinal_features.npz"
BIOMARKER_FEATURES = r"D:\discs\project_biomarker_fusion\data\biomarker_longitudinal.npz"
OUTPUT_PATH = r"D:\discs\project_biomarker_fusion\data\fusion_dataset.npz"

def load_resnet_features():
    """Load ResNet features from longitudinal experiment."""
    print("Loading ResNet features...")
    data = np.load(RESNET_FEATURES, allow_pickle=True)
    
    features = data['features']
    subject_ids = data['subject_ids']
    visit_nums = data['visit_nums']
    labels = data['labels']
    splits = data['splits']
    
    print(f"Loaded {len(features)} scans")
    
    # Group by subject to get baseline and followup
    pairs = {}
    
    unique_subjects = np.unique(subject_ids)
    for subj in unique_subjects:
        mask = subject_ids == subj
        subj_features = features[mask]
        subj_visits = visit_nums[mask]
        subj_labels = labels[mask]
        subj_splits = splits[mask]
        
        if len(subj_features) < 2:
            continue
        
        # Sort by visit
        order = np.argsort(subj_visits)
        subj_features = subj_features[order]
        
        # Baseline and followup
        baseline_resnet = subj_features[0]
        followup_resnet = subj_features[-1]
        delta_resnet = followup_resnet - baseline_resnet
        
        pairs[subj] = {
            'baseline_resnet': baseline_resnet,
            'followup_resnet': followup_resnet,
            'delta_resnet': delta_resnet,
            'label': subj_labels[0],
            'split': subj_splits[0]
        }
    
    print(f"Paired {len(pairs)} subjects")
    return pairs

def load_biomarker_features():
    """Load biomarker features."""
    print("\nLoading biomarker features...")
    data = np.load(BIOMARKER_FEATURES, allow_pickle=True)
    
    subject_ids = data['subject_ids']
    n = len(subject_ids)
    
    biomarkers = {}
    for i, subj in enumerate(subject_ids):
        biomarkers[subj] = {
            'split': str(data['splits'][i]),
            'label': int(data['labels'][i]),
            
            # Baseline
            'baseline_hippocampus': data['baseline_hippocampus'][i],
            'baseline_ventricles': data['baseline_ventricles'][i],
            'baseline_entorhinal': data['baseline_entorhinal'][i],
            'baseline_apoe4': data['baseline_apoe4'][i],
            'baseline_age': data['baseline_age'][i],
            'baseline_sex': data['baseline_sex'][i],
            
            # Followup
            'followup_hippocampus': data['followup_hippocampus'][i],
            'followup_ventricles': data['followup_ventricles'][i],
            'followup_entorhinal': data['followup_entorhinal'][i],
            
            # Deltas
            'delta_hippocampus': data['delta_hippocampus'][i],
            'delta_ventricles': data['delta_ventricles'][i],
            'delta_entorhinal': data['delta_entorhinal'][i],
            
            'time_diff_months': data['time_diff_months'][i]
        }
    
    print(f"Loaded {len(biomarkers)} subjects")
    return biomarkers

def merge_datasets(resnet_pairs, biomarkers):
    """Merge ResNet + Biomarker features."""
    print("\nMerging datasets...")
    
    # Find common subjects
    resnet_subjects = set(resnet_pairs.keys())
    biomarker_subjects = set(biomarkers.keys())
    common_subjects = resnet_subjects & biomarker_subjects
    
    print(f"ResNet subjects: {len(resnet_subjects)}")
    print(f"Biomarker subjects: {len(biomarker_subjects)}")
    print(f"Common subjects: {len(common_subjects)}")
    
    # Merge
    merged = []
    for subj in common_subjects:
        resnet_data = resnet_pairs[subj]
        bio_data = biomarkers[subj]
        
        # Verify labels match
        if resnet_data['label'] != bio_data['label']:
            print(f"Warning: {subj} - label mismatch!")
            continue
        
        merged.append({
            'subject_id': subj,
            'split': resnet_data['split'],
            'label': resnet_data['label'],
            
            # ResNet features (512 each)
            'baseline_resnet': resnet_data['baseline_resnet'],
            'followup_resnet': resnet_data['followup_resnet'],
            'delta_resnet': resnet_data['delta_resnet'],
            
            # Biomarkers
            **bio_data
        })
    
    print(f"Merged {len(merged)} subjects successfully")
    return merged

def prepare_arrays(merged_data):
    """Convert to numpy arrays for training."""
    print("\nPreparing arrays...")
    
    n = len(merged_data)
    
    # Initialize arrays
    subject_ids = []
    splits = []
    labels = []
    
    # ResNet features
    baseline_resnet = np.zeros((n, 512))
    followup_resnet = np.zeros((n, 512))
    delta_resnet = np.zeros((n, 512))
    
    # Biomarkers (6 baseline, 3 followup, 3 delta)
    baseline_bio = np.zeros((n, 6))
    followup_bio = np.zeros((n, 3))
    delta_bio = np.zeros((n, 3))
    
    time_diff = np.zeros(n)
    
    for i, data in enumerate(merged_data):
        subject_ids.append(data['subject_id'])
        splits.append(data['split'])
        labels.append(data['label'])
        
        # ResNet
        baseline_resnet[i] = data['baseline_resnet']
        followup_resnet[i] = data['followup_resnet']
        delta_resnet[i] = data['delta_resnet']
        
        # Baseline biomarkers
        baseline_bio[i] = [
            data['baseline_hippocampus'],
            data['baseline_ventricles'],
            data['baseline_entorhinal'],
            data['baseline_apoe4'],
            data['baseline_age'],
            data['baseline_sex']
        ]
        
        # Followup biomarkers
        followup_bio[i] = [
            data['followup_hippocampus'],
            data['followup_ventricles'],
            data['followup_entorhinal']
        ]
        
        # Delta biomarkers
        delta_bio[i] = [
            data['delta_hippocampus'],
            data['delta_ventricles'],
            data['delta_entorhinal']
        ]
        
        time_diff[i] = data['time_diff_months']
    
    return {
        'subject_ids': np.array(subject_ids),
        'splits': np.array(splits),
        'labels': np.array(labels),
        
        'baseline_resnet': baseline_resnet,
        'followup_resnet': followup_resnet,
        'delta_resnet': delta_resnet,
        
        'baseline_bio': baseline_bio,
        'followup_bio': followup_bio,
        'delta_bio': delta_bio,
        
        'time_diff_months': time_diff
    }

def main():
    """Main pipeline."""
    print("="*60)
    print("FUSION DATASET PREPARATION")
    print("="*60)
    
    # Load both datasets
    resnet_pairs = load_resnet_features()
    biomarkers = load_biomarker_features()
    
    # Merge
    merged = merge_datasets(resnet_pairs, biomarkers)
    
    # Prepare arrays
    arrays = prepare_arrays(merged)
    
    # Split statistics
    train_mask = arrays['splits'] == 'train'
    test_mask = arrays['splits'] == 'test'
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print(f"\nTotal subjects: {len(arrays['labels'])}")
    print(f"  Train: {train_mask.sum()}")
    print(f"  Test: {test_mask.sum()}")
    
    print(f"\nClass distribution:")
    print(f"  Stable (0): {(arrays['labels'] == 0).sum()}")
    print(f"  Converter (1): {(arrays['labels'] == 1).sum()}")
    
    print(f"\nFeature dimensions:")
    print(f"  Baseline ResNet: {arrays['baseline_resnet'].shape}")
    print(f"  Followup ResNet: {arrays['followup_resnet'].shape}")
    print(f"  Delta ResNet: {arrays['delta_resnet'].shape}")
    print(f"  Baseline Bio: {arrays['baseline_bio'].shape}")
    print(f"  Followup Bio: {arrays['followup_bio'].shape}")
    print(f"  Delta Bio: {arrays['delta_bio'].shape}")
    
    total_features = 512*3 + 6 + 3 + 3
    print(f"\n  TOTAL INPUT FEATURES: {total_features}")
    
    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, **arrays)
    
    print(f"\n✅ COMPLETE - Fusion dataset ready!")
    print(f"✅ No existing files were modified!")

if __name__ == '__main__':
    main()
