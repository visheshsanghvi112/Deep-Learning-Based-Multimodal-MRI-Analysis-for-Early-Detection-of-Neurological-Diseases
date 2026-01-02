import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
ADNIMERGE_PATH = r"D:\discs\data\ADNI\ADNIMERGE_23Dec2025.csv"
LEVEL1_TRAIN = r"D:\discs\project_adni\data\features\train_level1.csv"
LEVEL1_TEST = r"D:\discs\project_adni\data\features\test_level1.csv"
OUTPUT_DIR = r"D:\discs\project_adni\data\features"

# Features to add for Level-MAX
# Note: We exclude MMSE, CDRSB, ADAS which are circular/diagnosis proxies.
NEW_CLINICAL_COLS = [
    # Demographics
    'PTEDUCAT', 
    
    # Genetics
    'APOE4',
    
    # Volumetric (Freesurfer)
    'Hippocampus', 
    'Ventricles', 
    'Entorhinal', 
    'Fusiform', 
    'MidTemp', 
    'WholeBrain', 
    'ICV',  # Intracranial Volume (good for normalization)
    
    # CSF Biomarkers
    'ABETA', 
    'TAU', 
    'PTAU'
]

# Mapping to check for strings in CSF
CSF_COLS = ['ABETA', 'TAU', 'PTAU']

def clean_value(x):
    """Clean numeric values that might be strings (e.g. '>1700')."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        # Remove <> and convert
        clean_str = ''.join(c for c in x if c.isdigit() or c == '.')
        if not clean_str:
            return np.nan
        return float(clean_str)
    return float(x)

def process_adnimerge():
    print("Loading ADNIMERGE...")
    adni = pd.read_csv(ADNIMERGE_PATH, low_memory=False)
    
    # Filter to Baseline
    # Note: 'bl' is standard baseline code.
    adni_bl = adni[adni['VISCODE'] == 'bl'].copy()
    print(f"ADNIMERGE Baseline rows: {len(adni_bl)}")
    
    # Select columns
    cols_to_keep = ['PTID'] + NEW_CLINICAL_COLS
    # Check availability
    available_cols = [c for c in cols_to_keep if c in adni_bl.columns]
    missing_cols = set(cols_to_keep) - set(available_cols)
    if missing_cols:
        print(f"WARNING: The following columns are missing in ADNIMERGE: {missing_cols}")
    
    df_features = adni_bl[available_cols].copy()
    
    # Cleaning
    print("Cleaning clinical features...")
    for col in NEW_CLINICAL_COLS:
        if col not in df_features.columns:
            continue
            
        # specifically handle CSF strings
        if col in CSF_COLS:
            df_features[col] = df_features[col].apply(clean_value)
        else:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
            
    return df_features

def impute_missing(df, ref_df=None):
    """
    Impute missing values using median.
    If ref_df is provided (e.g. training set), use its medians.
    """
    if ref_df is None:
        ref_df = df
        
    imputed = df.copy()
    stats = {}
    
    for col in NEW_CLINICAL_COLS:
        if col not in imputed.columns:
            continue
            
        median_val = ref_df[col].median()
        missing_count = imputed[col].isna().sum()
        
        if missing_count > 0:
            imputed[col] = imputed[col].fillna(median_val)
            print(f"  - Imputed {col}: {missing_count} missing values filled with {median_val}")
            
        stats[col] = median_val
        
    return imputed, stats

def create_dataset(input_path, adni_features, output_path, is_train=True, stats=None):
    print(f"\nProcessing {input_path}...")
    base_df = pd.read_csv(input_path)
    print(f"  Base shape: {base_df.shape}")
    
    # Merge on Subject ID
    # base_df has 'Subject', adni_features has 'PTID'
    merged = base_df.merge(adni_features, left_on='Subject', right_on='PTID', how='left')
    
    # Check merge success
    print(f"  Merged shape: {merged.shape}")
    missing_clinical = merged[NEW_CLINICAL_COLS[0]].isna().sum()
    if missing_clinical > 0:
        print(f"  WARNING: {missing_clinical} subjects missing clinical data after merge!")
    
    # Impute
    # For training set, we calculate stats. For test set, we use passed stats.
    if is_train:
        train_stats = {}
        for col in NEW_CLINICAL_COLS:
            if col not in merged.columns: continue
            
            # Simple median imputation
            median_val = merged[col].median()
            
            # Check if all nan
            if pd.isna(median_val):
                print(f"  WARNING: {col} is ALL NaN! Filling with 0.")
                median_val = 0
            
            missing_count = merged[col].isna().sum()
            if missing_count > 0:
                merged[col] = merged[col].fillna(median_val)
                print(f"  - Imputed {col}: {missing_count} missing values filled with {median_val}")
            
            train_stats[col] = median_val
        stats_out = train_stats
    else:
        # Use provided stats for test set
        if stats is None:
            raise ValueError("Must provide stats for test set processing!")
            
        stats_out = stats # Return unchanged
        for col, val in stats.items():
            if col in merged.columns:
                missing_count = merged[col].isna().sum()
                if missing_count > 0:
                     merged[col] = merged[col].fillna(val)
                     print(f"  - Imputed {col} (Test): {missing_count} missing filled with {val}")

    # Drop PTID if strictly needed, or keep it. Level 1 files have 'Subject'.
    if 'PTID' in merged.columns:
        merged.drop(columns=['PTID'], inplace=True)
        
    # Save
    print(f"  Saving to {output_path}...")
    merged.to_csv(output_path, index=False)
    return stats_out

def main():
    # 1. Get Clinical Features
    adni_features = process_adnimerge()
    
    # 2. Process Train
    train_stats = create_dataset(LEVEL1_TRAIN, adni_features, os.path.join(OUTPUT_DIR, "train_level_max.csv"), is_train=True)
    
    # 3. Process Test (using train stats)
    create_dataset(LEVEL1_TEST, adni_features, os.path.join(OUTPUT_DIR, "test_level_max.csv"), is_train=False, stats=train_stats)
    
    print("\nDONE. Created Level-MAX datasets.")

if __name__ == "__main__":
    main()
