"""
ADNIMERGE Verification Script
Verifies identifier consistency, column structure, and biomarker availability.
"""
import pandas as pd
import os

# Find ADNIMERGE file
paths = [
    r"D:\discs\data\ADNI\ADNIMERGE_23Dec2025.csv",
    r"C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T\ADNIMERGE_15Jan2025.csv"
]

for path in paths:
    if os.path.exists(path):
        print(f"Found ADNIMERGE at: {path}")
        df = pd.read_csv(path, low_memory=False)
        break
else:
    print("ADNIMERGE not found!")
    exit(1)

print("\n" + "="*70)
print("  ADNIMERGE STRUCTURE VERIFICATION")
print("="*70)

print(f"\n1. DATASET SIZE:")
print(f"   Total rows: {len(df)}")
print(f"   Total columns: {len(df.columns)}")

print(f"\n2. KEY IDENTIFIERS:")
print(f"   RID (unique subjects): {df['RID'].nunique()}")
print(f"   PTID (unique subjects): {df['PTID'].nunique()}")
print(f"   Sample PTIDs: {list(df['PTID'].unique()[:5])}")

print(f"\n3. VISIT STRUCTURE (VISCODE):")
viscode_counts = df['VISCODE'].value_counts().head(10)
for vc, count in viscode_counts.items():
    print(f"   {vc}: {count} rows")

print(f"\n4. VOLUMETRIC BIOMARKERS (FREESURFER-DERIVED):")
vol_cols = ['Hippocampus', 'Ventricles', 'Entorhinal', 'MidTemp', 'Fusiform', 'WholeBrain', 'ICV']
for col in vol_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"   {col}: {non_null} non-null values ({non_null/len(df)*100:.1f}%)")
    else:
        print(f"   {col}: NOT FOUND")

print(f"\n5. COGNITIVE SCORES (SHOULD NOT BE USED AS PREDICTORS):")
cog_cols = ['MMSE', 'CDRSB', 'ADAS11', 'ADAS13', 'RAVLT_immediate']
for col in cog_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"   {col}: {non_null} non-null values (⚠️ DO NOT USE AS PREDICTOR)")

print(f"\n6. DEMOGRAPHIC FEATURES:")
demo_cols = ['AGE', 'PTGENDER', 'APOE4', 'PTEDUCAT']
for col in demo_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"   {col}: {non_null} non-null values")

print(f"\n7. DIAGNOSIS COLUMNS:")
dx_cols = ['DX', 'DX_bl']
for col in dx_cols:
    if col in df.columns:
        print(f"   {col}: {df[col].value_counts().to_dict()}")

print(f"\n8. EXAMDATE AVAILABILITY:")
if 'EXAMDATE' in df.columns:
    non_null = df['EXAMDATE'].notna().sum()
    print(f"   EXAMDATE: {non_null} non-null values ({non_null/len(df)*100:.1f}%)")
else:
    print("   EXAMDATE: NOT FOUND")

print("\n" + "="*70)
print("  VERIFICATION COMPLETE")
print("="*70)
