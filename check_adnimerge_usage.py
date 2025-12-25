"""
Check ADNIMERGE file and determine how much has been used
"""
import pandas as pd
import os
from pathlib import Path

# Load ADNIMERGE
print("=" * 80)
print("ADNIMERGE FILE ANALYSIS")
print("=" * 80)

adnimerge_path = Path("ADNI/ADNIMERGE_23Dec2025.csv")
df = pd.read_csv(adnimerge_path)

print(f"\n📊 ADNIMERGE Statistics:")
print(f"  Total Rows: {len(df):,}")
print(f"  Total Columns: {len(df.columns)}")
print(f"  File Size: {os.path.getsize(adnimerge_path) / (1024*1024):.2f} MB")

# Check key columns
print(f"\n🔑 Key Columns Present:")
key_cols = ['PTID', 'DX', 'MMSE', 'CDRSB', 'AGE', 'EXAMDATE', 'VISCODE']
for col in key_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  ✓ {col}: {non_null:,} non-null values ({non_null/len(df)*100:.1f}%)")
    else:
        print(f"  ✗ {col}: NOT FOUND")

# Show all columns
print(f"\n📋 All Columns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    if i % 5 == 0:
        print(f"  {col}")
    else:
        print(f"  {col}", end="")
print()

# Check unique subjects
if 'PTID' in df.columns:
    unique_subjects = df['PTID'].nunique()
    print(f"\n👥 Unique Subjects: {unique_subjects:,}")
    
    # Show sample subjects
    sample_ptids = df['PTID'].dropna().unique()[:5]
    print(f"  Sample PTIDs: {', '.join(str(p) for p in sample_ptids)}")

# Check diagnosis distribution
if 'DX' in df.columns:
    print(f"\n🏥 Diagnosis Distribution:")
    dx_counts = df['DX'].value_counts()
    for dx, count in dx_counts.items():
        print(f"  {dx}: {count:,} ({count/len(df)*100:.1f}%)")

# Now check what we have in our ADNI folder
print("\n" + "=" * 80)
print("MRI FILES IN ADNI FOLDER")
print("=" * 80)

adni_folder = Path("ADNI")
nii_files = list(adni_folder.rglob("*.nii"))
print(f"\n📁 NIfTI Files Found: {len(nii_files)}")

# Extract subject IDs from file paths
subject_ids_from_files = set()
for nii in nii_files:
    # Subject ID is typically 4 levels up from the .nii file
    # E.g., ADNI/002_S_0295/MPR.../2006.../I40966/file.nii
    try:
        subject_id = nii.parts[-4]
        subject_ids_from_files.add(subject_id)
    except:
        pass

print(f"  Unique Subjects with MRI: {len(subject_ids_from_files)}")
print(f"  Sample Subject IDs: {', '.join(list(subject_ids_from_files)[:5])}")

# Check overlap
if 'PTID' in df.columns:
    print("\n" + "=" * 80)
    print("DATA OVERLAP ANALYSIS")
    print("=" * 80)
    
    adnimerge_subjects = set(df['PTID'].dropna().unique())
    
    overlap = subject_ids_from_files & adnimerge_subjects
    only_in_files = subject_ids_from_files - adnimerge_subjects
    only_in_merge = adnimerge_subjects - subject_ids_from_files
    
    print(f"\n🔄 Overlap Statistics:")
    print(f"  Subjects in ADNIMERGE: {len(adnimerge_subjects):,}")
    print(f"  Subjects with MRI files: {len(subject_ids_from_files)}")
    print(f"  Subjects in BOTH: {len(overlap)} ✓")
    print(f"  Only in MRI files: {len(only_in_files)}")
    print(f"  Only in ADNIMERGE: {len(only_in_merge):,}")
    
    if len(overlap) > 0:
        print(f"\n✅ Usable Matched Subjects: {len(overlap)}")
        print(f"   (These have both MRI scans AND clinical data)")
        
        # Show sample matched subjects
        sample_matched = list(overlap)[:10]
        print(f"\n   Sample Matched PTIDs:")
        for ptid in sample_matched:
            # Get some clinical info
            subject_data = df[df['PTID'] == ptid].iloc[0]
            dx = subject_data.get('DX', 'N/A')
            age = subject_data.get('AGE', 'N/A')
            print(f"     {ptid}: DX={dx}, Age={age}")

# Check what features have been extracted
print("\n" + "=" * 80)
print("FEATURE EXTRACTION STATUS")
print("=" * 80)

feature_file = Path("extracted_features/adni_features.csv")
if feature_file.exists():
    feat_df = pd.read_csv(feature_file)
    print(f"\n✅ Features Extracted!")
    print(f"  File: {feature_file}")
    print(f"  Rows: {len(feat_df)}")
    print(f"  Columns: {len(feat_df.columns)}")
    
    if 'subject_id' in feat_df.columns:
        extracted_subjects = set(feat_df['subject_id'].unique())
        print(f"  Unique Subjects: {len(extracted_subjects)}")
        
        if 'PTID' in df.columns:
            matched_with_clinical = extracted_subjects & adnimerge_subjects
            print(f"  Matched with ADNIMERGE: {len(matched_with_clinical)}")
else:
    print(f"\n⚠️  Features NOT extracted yet")
    print(f"  Expected file: {feature_file}")

# Summary
print("\n" + "=" * 80)
print("USAGE SUMMARY")
print("=" * 80)

print(f"""
📊 ADNIMERGE Data:
   - Total records: {len(df):,}
   - Unique subjects: {unique_subjects if 'PTID' in df.columns else 'N/A'}
   - File has clinical data for analysis

📁 MRI Data:
   - NIfTI files: {len(nii_files)}
   - Unique subjects: {len(subject_ids_from_files)}
   
🔄 Matched Data (MRI + Clinical):
   - Subjects with BOTH: {len(overlap) if 'PTID' in df.columns else 'N/A'}
   - Percentage of MRI data matched: {len(overlap)/len(subject_ids_from_files)*100 if 'PTID' in df.columns and len(subject_ids_from_files) > 0 else 0:.1f}%
   - Percentage of ADNIMERGE used: {len(overlap)/len(adnimerge_subjects)*100 if 'PTID' in df.columns else 0:.2f}%

⚙️ Processing Status:
   - Features extracted: {'YES' if feature_file.exists() else 'NO'}
   - Ready for training: {'YES' if feature_file.exists() and len(overlap) > 0 else 'NO'}
""")

print("=" * 80)
