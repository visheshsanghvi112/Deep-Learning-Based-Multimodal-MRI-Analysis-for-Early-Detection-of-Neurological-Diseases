import pandas as pd
from pathlib import Path

print("ADNI MERGE USAGE ANALYSIS")
print("=" * 60)

# Check ADNIMERGE file
merge_path = Path("ADNI/ADNIMERGE_23Dec2025.csv")
if merge_path.exists():
    df_merge = pd.read_csv(merge_path)
    print(f"\n1. ADNIMERGE.csv:")
    print(f"   Total records: {len(df_merge):,}")
    if 'PTID' in df_merge.columns:
        print(f"   Unique subjects: {df_merge['PTID'].nunique():,}")
    print(f"   Columns: {len(df_merge.columns)}")
    print(f"   File size: {merge_path.stat().st_size / (1024**2):.1f} MB")
else:
    print("\n1. ADNIMERGE.csv: NOT FOUND")
    df_merge = None

# Check extracted features
features_path = Path("extracted_features/adni_features.csv")
if features_path.exists():
    df_features = pd.read_csv(features_path)
    print(f"\n2. ADNI Features (extracted_features/adni_features.csv):")
    print(f"   Total rows: {len(df_features):,}")
    if 'subject_id' in df_features.columns:
        unique_subj = df_features['subject_id'].nunique()
        print(f"   Unique subjects: {unique_subj}")
        print(f"   Scans per subject (avg): {len(df_features) / unique_subj:.1f}")
    print(f"   Feature columns: {len(df_features.columns)}")
    print(f"   File size: {features_path.stat().st_size / (1024**2):.1f} MB")
else:
    print("\n2. ADNI Features: NOT EXTRACTED YET")
    df_features = None

# Check MRI files
adni_dir = Path("ADNI")
nii_files = list(adni_dir.rglob("*.nii"))
print(f"\n3. MRI Files (ADNI folder):")
print(f"   Total .nii files: {len(nii_files)}")

subject_ids_from_mri = set()
for nii in nii_files:
    try:
        subject_id = nii.parts[-4]
        subject_ids_from_mri.add(subject_id)
    except:
        pass
print(f"   Unique subjects with MRI: {len(subject_ids_from_mri)}")

# Calculate usage
print("\n" + "=" * 60)
print("USAGE SUMMARY")
print("=" * 60)

if df_merge is not None and 'PTID' in df_merge.columns:
    total_merge_subjects = df_merge['PTID'].nunique()
    
    if df_features is not None and 'subject_id' in df_features.columns:
        extracted_subjects = set(df_features['subject_id'].unique())
        
        # Find overlap
        merge_subjects_set = set(df_merge['PTID'].unique())
        matched = extracted_subjects & merge_subjects_set
        
        print(f"\nADNIMERGE Data:")
        print(f"  Total subjects: {total_merge_subjects:,}")
        print(f"\nMRI Data Available:")
        print(f"  Subjects with scans: {len(subject_ids_from_mri)}")
        print(f"\nExtracted Features:")
        print(f"  Subjects processed: {len(extracted_subjects)}")
        print(f"  Total feature rows: {len(df_features):,}")
        print(f"\nMatched (MRI + Clinical):")
        print(f"  Subjects with BOTH: {len(matched)}")
        print(f"\nUsage Statistics:")
        print(f"  % of ADNIMERGE used: {len(matched) / total_merge_subjects * 100:.2f}%")
        print(f"  % of MRI data processed: {len(extracted_subjects) / len(subject_ids_from_mri) * 100:.1f}%")
        
        print(f"\nData Ready for Training: {'YES ✓' if len(matched) > 0 else 'NO ✗'}")
    else:
        print(f"\nADNIMERGE: {total_merge_subjects:,} subjects")
        print(f"MRI Files: {len(subject_ids_from_mri)} subjects")
        print(f"Features: NOT EXTRACTED")
        print(f"\n% of ADNIMERGE used: 0.00% (features not extracted yet)")
else:
    print("\nCannot calculate usage - ADNIMERGE or key columns not found")

print("\n" + "=" * 60)
