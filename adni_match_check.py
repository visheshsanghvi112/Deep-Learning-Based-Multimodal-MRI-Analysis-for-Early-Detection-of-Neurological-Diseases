
import os
import pandas as pd
from pathlib import Path

ADNI_DIR = Path("D:/discs/ADNI")
CSV_1 = ADNI_DIR / "ADNI1_Complete_1Yr_1.5T_12_19_2025.csv"
CSV_2 = ADNI_DIR / "ADNI1_Complete_1Yr_1.5T_12_19_2025 (1).csv"

def check_coverage():
    print("--- ADNI Data Match Report ---")
    
    # 1. Get Physical Folders
    folder_subjects = [d.name for d in ADNI_DIR.iterdir() if d.is_dir() and "_" in d.name]
    folder_set = set(folder_subjects)
    print(f"Physical Subject Folders: {len(folder_set)}")
    
    # 2. Get CSV Data
    dfs = []
    for csv in [CSV_1, CSV_2]:
        if csv.exists():
            try:
                # ADNI CSVs often have quoting issues or weird headers, using default settings first
                df = pd.read_csv(csv, quotechar='"')
                dfs.append(df)
                print(f"Loaded {csv.name}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {csv.name}: {e}")
    
    if not dfs:
        print("No CSVs loaded!")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 3. Analyze Linkage
    # CSV Subject column often looks like '002_S_0295'
    if 'Subject' not in combined_df.columns:
        print("CRITICAL: 'Subject' column not found in CSV keys:", combined_df.columns)
        return
        
    csv_subjects = set(combined_df['Subject'].unique())
    print(f"Unique Subjects in CSV: {len(csv_subjects)}")
    
    # Intersection
    matched = folder_set.intersection(csv_subjects)
    missing_metadata = folder_set - csv_subjects
    missing_data = csv_subjects - folder_set
    
    print(f"\n--- MATCH STATISTICS ---")
    print(f"MATCHED Subjects (Ready to Train): {len(matched)}")
    print(f"Subjects in Folder but MISSING CSV (Unusable): {len(missing_metadata)}")
    print(f"Subjects in CSV but MISSING Images (Unusable): {len(missing_data)}")
    
    # 4. Analyze Data Quality for Matched Subjects
    matched_df = combined_df[combined_df['Subject'].isin(matched)].copy()
    
    # Check for critical columns
    print(f"\n--- DATA QUALITY (Matched Subjects) ---")
    critical_cols = ['Age', 'Group', 'Sex']
    for col in critical_cols:
        if col in matched_df.columns:
            missing_val = matched_df[col].isna().sum()
            print(f"Missing '{col}': {missing_val} rows")
        else:
            print(f"CRITICAL: Column '{col}' COMPLETELY MISSING")

    # Check for diagnosis distribution
    if 'Group' in matched_df.columns:
        print("\nDiagnosis Distribution (Matched):")
        print(matched_df['Group'].value_counts())

if __name__ == "__main__":
    check_coverage()
