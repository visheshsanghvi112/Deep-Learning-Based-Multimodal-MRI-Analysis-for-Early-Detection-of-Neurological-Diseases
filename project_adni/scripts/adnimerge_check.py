import pandas as pd
import numpy as np

ADNIMERGE_CSV = r"D:\discs\ADNI\ADNIMERGE_23Dec2025.csv"
BASELINE_CSV = r"D:\discs\adni_baseline_selection.csv"
OUTPUT_FILE = r"D:\discs\adnimerge_report.txt"

# Load data
merge_df = pd.read_csv(ADNIMERGE_CSV, low_memory=False)
baseline_df = pd.read_csv(BASELINE_CSV)

# Check alignment
our_subjects = set(baseline_df['Subject'].unique())
merge_subjects = set(merge_df['PTID'].unique())
overlap = our_subjects.intersection(merge_subjects)

# Filter to baseline
filtered = merge_df[merge_df['PTID'].isin(our_subjects)]
bl = filtered[filtered['VISCODE'].isin(['bl', 'sc', 'scmri', 'm00'])]

# Build report
lines = []
lines.append("="*70)
lines.append("ADNIMERGE ANALYSIS REPORT")
lines.append("="*70)
lines.append("")
lines.append(f"ADNIMERGE: {len(merge_df)} rows, {merge_df['PTID'].nunique()} unique subjects")
lines.append(f"Our Data:  {len(baseline_df)} subjects")
lines.append("")
lines.append(f"ALIGNMENT: {len(overlap)}/{len(our_subjects)} = {len(overlap)/len(our_subjects)*100:.1f}%")
lines.append(f"Baseline rows: {len(bl)} ({bl['PTID'].nunique()} unique subjects)")
lines.append("")
lines.append("KEY CLINICAL FEATURES AT BASELINE:")
for col in ['MMSE', 'CDRSB', 'ADAS11', 'ADAS13', 'PTEDUCAT', 'APOE4', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'ICV']:
    if col in bl.columns:
        n = bl[col].notna().sum()
        pct = n / len(bl) * 100 if len(bl) > 0 else 0
        lines.append(f"  {col:<15}: {n:>4}/{len(bl)} ({pct:>5.1f}%)")
    else:
        lines.append(f"  {col:<15}: NOT IN DATA")

lines.append("")
lines.append("="*70)
lines.append("RECOMMENDATION")
lines.append("="*70)
lines.append("")

# Calculate key coverage
mmse_pct = bl['MMSE'].notna().sum() / len(bl) * 100 if 'MMSE' in bl.columns and len(bl) > 0 else 0
cdrsb_pct = bl['CDRSB'].notna().sum() / len(bl) * 100 if 'CDRSB' in bl.columns and len(bl) > 0 else 0
educ_pct = bl['PTEDUCAT'].notna().sum() / len(bl) * 100 if 'PTEDUCAT' in bl.columns and len(bl) > 0 else 0

if mmse_pct > 80 and cdrsb_pct > 80:
    lines.append("[GOOD] ADNIMERGE provides valuable clinical features:")
    lines.append(f"  - MMSE coverage:      {mmse_pct:.1f}%")
    lines.append(f"  - CDRSB coverage:     {cdrsb_pct:.1f}%")
    lines.append(f"  - Education coverage: {educ_pct:.1f}%")
    lines.append("")
    lines.append("RECOMMEND: Merge ADNIMERGE clinical features with MRI features")
    lines.append("This will enable multimodal experiments matching OASIS design")
else:
    lines.append("[PARTIAL] Some clinical features have low coverage")

lines.append("")
lines.append("="*70)

# Write to file
with open(OUTPUT_FILE, 'w') as f:
    f.write('\n'.join(lines))

# Also print
print('\n'.join(lines))
