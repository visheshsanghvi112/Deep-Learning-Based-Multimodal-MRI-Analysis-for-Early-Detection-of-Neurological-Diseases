"""Quick debug script to check ADNI scans availability by diagnosis."""
import os
import glob
import pandas as pd

ADNI_SOURCE_DIR = r"C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T\ADNI"
csv_path = r"d:\discs\data\ADNI\ADNIMERGE_23Dec2025.csv"

print("Loading CSV...")
df = pd.read_csv(csv_path, low_memory=False)

print(f"\nColumns in CSV: {list(df.columns)[:10]}...")
print(f"\nTotal rows: {len(df)}")

# Check DX distribution
if 'DX' in df.columns:
    print(f"\nDiagnosis distribution:")
    print(df['DX'].value_counts())

# Check what subject directories exist
subject_dirs = [os.path.basename(d) for d in glob.glob(os.path.join(ADNI_SOURCE_DIR, "*_S_*")) if os.path.isdir(d)]
print(f"\nFound {len(subject_dirs)} subject directories in ADNI source")
print(f"First 10: {subject_dirs[:10]}")

# Try to match with CSV
if 'PTID' in df.columns and 'DX' in df.columns:
    # Get baseline only
    baseline = df[df['VISCODE'] == 'bl'] if 'VISCODE' in df.columns else df
    
    matched_subjects = {'CN': [], 'MCI': [], 'AD': []}
    
    for subj_dir in subject_dirs:
        # Try to find in CSV
        for _, row in baseline.iterrows():
            ptid = str(row['PTID'])
            # Try different formats
            if subj_dir in ptid or ptid.replace('_', '_S_') == subj_dir or ptid == subj_dir.replace('_S_', '_'):
                dx = row['DX']
                if pd.notna(dx) and dx in matched_subjects:
                    # Check if has scans
                    scan_dir = os.path.join(ADNI_SOURCE_DIR, subj_dir)
                    nii_files = glob.glob(os.path.join(scan_dir, "**", "*.nii"), recursive=True)
                    if nii_files:
                        matched_subjects[dx].append(subj_dir)
                    break
    
    print(f"\nMatched subjects with scans:")
    print(f"  CN: {len(matched_subjects['CN'])}")
    print(f"  MCI: {len(matched_subjects['MCI'])}")
    print(f"  AD: {len(matched_subjects['AD'])}")
    
    if matched_subjects['AD']:
        print(f"\nAD subjects found: {matched_subjects['AD'][:5]}")
    else:
        print("\n⚠️ NO AD SUBJECTS FOUND!")
