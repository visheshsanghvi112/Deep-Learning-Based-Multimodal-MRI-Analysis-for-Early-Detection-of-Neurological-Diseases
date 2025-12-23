import pandas as pd
import numpy as np

ADNIMERGE_CSV = r"D:\discs\ADNI\ADNIMERGE_23Dec2025.csv"
BASELINE_CSV = r"D:\discs\adni_baseline_selection.csv"

print("="*70)
print("ADNIMERGE ANALYSIS RESULTS")
print("="*70)

# Load data
merge_df = pd.read_csv(ADNIMERGE_CSV, low_memory=False)
baseline_df = pd.read_csv(BASELINE_CSV)

print(f"\nADNIMERGE: {len(merge_df)} rows, {merge_df['PTID'].nunique()} unique subjects")
print(f"Our Data:  {len(baseline_df)} subjects")

# Check alignment
our_subjects = set(baseline_df['Subject'].unique())
merge_subjects = set(merge_df['PTID'].unique())
overlap = our_subjects.intersection(merge_subjects)
print(f"\nAlignment: {len(overlap)}/{len(our_subjects)} = {len(overlap)/len(our_subjects)*100:.1f}%")

# Filter to baseline
filtered = merge_df[merge_df['PTID'].isin(our_subjects)]
bl = filtered[filtered['VISCODE'].isin(['bl', 'sc', 'scmri', 'm00'])]
print(f"Baseline rows: {len(bl)} ({bl['PTID'].nunique()} unique subjects)")

# Key features
print("\nKey Clinical Features at Baseline:")
for col in ['MMSE', 'CDRSB', 'ADAS13', 'PTEDUCAT', 'APOE4', 'Hippocampus', 'ICV']:
    if col in bl.columns:
        n = bl[col].notna().sum()
        pct = n / len(bl) * 100 if len(bl) > 0 else 0
        print(f"  {col}: {n}/{len(bl)} ({pct:.1f}%)")

print("\n" + "="*70)
print("RECOMMENDATION: ADNIMERGE provides valuable clinical features!")
print("="*70)
