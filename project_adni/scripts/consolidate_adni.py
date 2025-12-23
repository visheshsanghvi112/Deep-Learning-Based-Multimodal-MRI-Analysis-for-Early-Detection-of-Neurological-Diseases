"""
ADNI Data Consolidation Script
================================================================================
Consolidates ADNI data downloaded in multiple parts into a unified dataset
for feature extraction and analysis.

Usage:
  python consolidate_adni.py
================================================================================
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from collections import Counter

# Paths
SOURCE_PATH = Path(r"C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T2")
TARGET_PATH = Path(r"D:\discs\ADNI")

print("=" * 80)
print("ADNI DATA CONSOLIDATION")
print("=" * 80)

# Step 1: Verify both paths exist
print("\n[STEP 1] Verifying paths...")
if not SOURCE_PATH.exists():
    print(f"❌ Source not found: {SOURCE_PATH}")
    exit(1)
if not TARGET_PATH.exists():
    print(f"❌ Target not found: {TARGET_PATH}")
    print("   Creating target directory...")
    TARGET_PATH.mkdir(parents=True, exist_ok=True)

print(f"✅ Source: {SOURCE_PATH}")
print(f"✅ Target: {TARGET_PATH}")

# Step 2: Analyze source data
print("\n[STEP 2] Analyzing source data...")
source_subjects = [d for d in SOURCE_PATH.iterdir() if d.is_dir() and d.name[0].isdigit()]
source_nifti = list(SOURCE_PATH.rglob("*.nii"))
source_csvs = list(SOURCE_PATH.glob("*.csv"))

print(f"  • Subject folders: {len(source_subjects)}")
print(f"  • NIfTI files: {len(source_nifti)}")
print(f"  • CSV files: {len(source_csvs)}")

# Step 3: List CSV files
if source_csvs:
    print("\n[STEP 3] CSV Files Found:")
    for csv_file in source_csvs:
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        print(f"  • {csv_file.name} ({size_mb:.2f} MB)")
        
        # Try to preview first few rows
        try:
            df = pd.read_csv(csv_file)
            print(f"    Columns: {', '.join(df.columns[:5])}...")
            print(f"    Rows: {len(df)}")
        except Exception as e:
            print(f"    Error reading: {e}")

# Step 4: Merge subject folders
print("\n[STEP 4] Copying subject folders to target...")
copied = 0
skipped = 0

for subject_dir in source_subjects:
    target_subject = TARGET_PATH / subject_dir.name
    
    if target_subject.exists():
        print(f"  ⊘ {subject_dir.name} (already exists, skipping)")
        skipped += 1
    else:
        try:
            shutil.copytree(subject_dir, target_subject, dirs_exist_ok=True)
            print(f"  ✓ {subject_dir.name}")
            copied += 1
        except Exception as e:
            print(f"  ✗ {subject_dir.name}: {e}")

print(f"\n  Summary: {copied} copied, {skipped} already existed")

# Step 5: Copy CSV files
print("\n[STEP 5] Copying CSV files to target...")
for csv_file in source_csvs:
    try:
        target_csv = TARGET_PATH / csv_file.name
        shutil.copy2(csv_file, target_csv)
        print(f"  ✓ {csv_file.name}")
    except Exception as e:
        print(f"  ✗ {csv_file.name}: {e}")

# Step 6: Final verification
print("\n[STEP 6] Final Verification...")
target_subjects = [d for d in TARGET_PATH.iterdir() if d.is_dir() and d.name[0].isdigit()]
target_nifti = list(TARGET_PATH.rglob("*.nii"))
target_csvs = list(TARGET_PATH.glob("*.csv"))

print(f"  • Total subject folders: {len(target_subjects)}")
print(f"  • Total NIfTI files: {len(target_nifti)}")
print(f"  • Total CSV files: {len(target_csvs)}")

print("\n" + "=" * 80)
print("✅ CONSOLIDATION COMPLETE")
print("=" * 80)
print(f"\nNext Steps:")
print(f"  1. Review the CSV files in {TARGET_PATH}")
print(f"  2. Run: python adni_feature_extraction.py")
print(f"  3. This will extract ResNet18 features from {len(target_nifti)} scans")
