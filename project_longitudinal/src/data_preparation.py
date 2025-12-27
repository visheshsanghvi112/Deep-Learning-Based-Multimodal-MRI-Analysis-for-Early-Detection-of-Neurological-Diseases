"""
Longitudinal ADNI Experiment - Data Preparation
================================================
Merges scan inventory with ADNIMERGE clinical data.
Creates progression labels and train/test splits.

Progression Labels:
- Stable:    No change in diagnosis over follow-up
- Converter: Diagnosis worsened (CN→MCI, CN→AD, MCI→AD)

Output: 
- data/processed/longitudinal_dataset.csv
- data/processed/train_test_split.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INVENTORY_PATH = r"D:\discs\project_longitudinal\data\processed\subject_inventory.csv"
ADNIMERGE_PATH = r"C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T\ADNI\ADNIMERGE_23Dec2025.csv_12_19_2025.csv"
ADNIMERGE_BACKUP = r"D:\discs\ADNI\ADNIMERGE_23Dec2025.csv"
OUTPUT_DATASET = r"D:\discs\project_longitudinal\data\processed\longitudinal_dataset.csv"
OUTPUT_SPLIT = r"D:\discs\project_longitudinal\data\processed\train_test_split.csv"

def load_adnimerge():
    """Load ADNIMERGE clinical data."""
    # Try primary path first
    if os.path.exists(ADNIMERGE_PATH):
        print(f"Loading ADNIMERGE from: {ADNIMERGE_PATH}")
        return pd.read_csv(ADNIMERGE_PATH)
    elif os.path.exists(ADNIMERGE_BACKUP):
        print(f"Loading ADNIMERGE from backup: {ADNIMERGE_BACKUP}")
        return pd.read_csv(ADNIMERGE_BACKUP)
    else:
        raise FileNotFoundError("ADNIMERGE not found!")

def parse_diagnosis(dx):
    """Convert diagnosis to numeric severity level."""
    if pd.isna(dx):
        return None
    dx = str(dx).upper().strip()
    if 'CN' in dx or 'NL' in dx:
        return 0  # Cognitively Normal
    elif 'EMCI' in dx:
        return 1  # Early MCI
    elif 'LMCI' in dx:
        return 2  # Late MCI
    elif 'MCI' in dx:
        return 1.5  # Generic MCI
    elif 'AD' in dx or 'DEMENTIA' in dx:
        return 3  # Alzheimer's / Dementia
    else:
        return None

def determine_progression(baseline_dx, last_dx):
    """Determine if subject progressed, stable, or improved."""
    if baseline_dx is None or last_dx is None:
        return 'Unknown'
    
    if last_dx > baseline_dx:
        return 'Converter'  # Worsened
    elif last_dx < baseline_dx:
        return 'Reverter'   # Improved (rare, possible data issue)
    else:
        return 'Stable'     # No change

def prepare_dataset():
    print("="*60)
    print("LONGITUDINAL DATA PREPARATION")
    print("="*60)
    
    # Load inventory
    print("\n[1] Loading scan inventory...")
    if not os.path.exists(INVENTORY_PATH):
        raise FileNotFoundError(f"Run data_inventory.py first! Missing: {INVENTORY_PATH}")
    
    inventory = pd.read_csv(INVENTORY_PATH)
    print(f"    Loaded {len(inventory)} scans from {inventory['subject_id'].nunique()} subjects")
    
    # Load ADNIMERGE
    print("\n[2] Loading ADNIMERGE clinical data...")
    merge_df = load_adnimerge()
    print(f"    Loaded {len(merge_df)} rows, {merge_df['PTID'].nunique() if 'PTID' in merge_df.columns else 'N/A'} subjects")
    
    # Find the correct subject ID column
    if 'PTID' in merge_df.columns:
        merge_df['subject_id'] = merge_df['PTID']
    elif 'Subject' in merge_df.columns:
        merge_df['subject_id'] = merge_df['Subject']
    else:
        print("    Available columns:", merge_df.columns.tolist()[:20])
        raise ValueError("Could not find subject ID column in ADNIMERGE")
    
    # Find diagnosis column
    dx_col = None
    for col in ['DX', 'DX_bl', 'DXCURREN', 'Group']:
        if col in merge_df.columns:
            dx_col = col
            break
    
    if dx_col is None:
        print("    Available columns:", merge_df.columns.tolist()[:30])
        # Try to find any column with 'DX' or 'Group'
        dx_candidates = [c for c in merge_df.columns if 'DX' in c.upper() or 'GROUP' in c.upper()]
        print(f"    Diagnosis candidates: {dx_candidates}")
        if dx_candidates:
            dx_col = dx_candidates[0]
        else:
            raise ValueError("Could not find diagnosis column")
    
    print(f"    Using diagnosis column: {dx_col}")
    
    # Parse diagnoses
    merge_df['dx_numeric'] = merge_df[dx_col].apply(parse_diagnosis)
    
    # Get baseline and last diagnosis per subject
    print("\n[3] Computing progression labels...")
    
    # Sort by subject and visit
    viscode_col = 'VISCODE' if 'VISCODE' in merge_df.columns else None
    if viscode_col:
        merge_df = merge_df.sort_values(['subject_id', viscode_col])
    
    # Get first (baseline) and last diagnosis per subject
    subject_progression = []
    for subj, group in merge_df.groupby('subject_id'):
        valid_dx = group.dropna(subset=['dx_numeric'])
        if len(valid_dx) == 0:
            continue
        
        baseline_dx = valid_dx.iloc[0]['dx_numeric']
        last_dx = valid_dx.iloc[-1]['dx_numeric']
        baseline_dx_str = valid_dx.iloc[0][dx_col]
        last_dx_str = valid_dx.iloc[-1][dx_col]
        
        progression = determine_progression(baseline_dx, last_dx)
        
        # Get clinical features from baseline
        baseline_row = valid_dx.iloc[0]
        age = baseline_row.get('AGE', baseline_row.get('Age', None))
        sex = baseline_row.get('PTGENDER', baseline_row.get('Sex', None))
        education = baseline_row.get('PTEDUCAT', None)
        mmse = baseline_row.get('MMSE', None)
        
        subject_progression.append({
            'subject_id': subj,
            'baseline_dx': baseline_dx_str,
            'baseline_dx_numeric': baseline_dx,
            'last_dx': last_dx_str,
            'last_dx_numeric': last_dx,
            'progression_label': progression,
            'num_visits': len(valid_dx),
            'age': age,
            'sex': sex,
            'education': education,
            'mmse_baseline': mmse
        })
    
    progression_df = pd.DataFrame(subject_progression)
    print(f"    Subjects with progression info: {len(progression_df)}")
    print(f"\n    Progression distribution:")
    print(progression_df['progression_label'].value_counts().to_string())
    
    # Merge with inventory
    print("\n[4] Merging with scan inventory...")
    dataset = inventory.merge(progression_df, on='subject_id', how='inner')
    print(f"    Matched scans: {len(dataset)} from {dataset['subject_id'].nunique()} subjects")
    
    # Filter to subjects with progression labels
    dataset = dataset[dataset['progression_label'].isin(['Stable', 'Converter'])]
    print(f"    After filtering (Stable/Converter only): {len(dataset)} scans from {dataset['subject_id'].nunique()} subjects")
    
    # Create binary label
    dataset['label'] = (dataset['progression_label'] == 'Converter').astype(int)
    
    # Create train/test split at SUBJECT level
    print("\n[5] Creating subject-level train/test split...")
    unique_subjects = dataset['subject_id'].unique()
    
    # Stratify by baseline diagnosis and progression
    strat_key = dataset.groupby('subject_id').first()['label']
    
    train_subjects, test_subjects = train_test_split(
        unique_subjects, 
        test_size=0.2, 
        stratify=strat_key.loc[unique_subjects],
        random_state=42
    )
    
    dataset['split'] = dataset['subject_id'].apply(
        lambda x: 'train' if x in train_subjects else 'test'
    )
    
    # Verify no leakage
    train_set = set(dataset[dataset['split']=='train']['subject_id'])
    test_set = set(dataset[dataset['split']=='test']['subject_id'])
    overlap = train_set & test_set
    assert len(overlap) == 0, f"LEAKAGE DETECTED! {len(overlap)} subjects in both sets!"
    print(f"    ✓ No subject leakage verified")
    
    print(f"\n    Train: {len(train_subjects)} subjects, {len(dataset[dataset['split']=='train'])} scans")
    print(f"    Test:  {len(test_subjects)} subjects, {len(dataset[dataset['split']=='test'])} scans")
    
    # Save outputs
    print("\n[6] Saving outputs...")
    dataset.to_csv(OUTPUT_DATASET, index=False)
    print(f"    Dataset saved to: {OUTPUT_DATASET}")
    
    split_info = dataset[['subject_id', 'split', 'label', 'progression_label', 'baseline_dx']].drop_duplicates()
    split_info.to_csv(OUTPUT_SPLIT, index=False)
    print(f"    Split info saved to: {OUTPUT_SPLIT}")
    
    # Final summary
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Total subjects:     {dataset['subject_id'].nunique()}")
    print(f"Total scans:        {len(dataset)}")
    print(f"Train subjects:     {len(train_subjects)}")
    print(f"Test subjects:      {len(test_subjects)}")
    print(f"\nLabel distribution (subjects):")
    print(f"  Stable:    {(split_info['label']==0).sum()}")
    print(f"  Converter: {(split_info['label']==1).sum()}")
    print("="*60)
    
    return dataset

if __name__ == "__main__":
    dataset = prepare_dataset()
