"""Check available data for expansion."""
import pandas as pd
import numpy as np

# Load ADNIMERGE
df = pd.read_csv(r'D:\discs\data\ADNI\ADNIMERGE_23Dec2025.csv')

print("="*60)
print("ADNIMERGE DATA SUMMARY")
print("="*60)

print(f"\nTotal rows: {len(df)}")
print(f"Total unique subjects: {df['PTID'].nunique()}")

# Check diagnosis distribution
print("\n--- Baseline Diagnosis Distribution ---")
baseline = df[df['VISCODE'] == 'bl']
print(baseline['DX_bl'].value_counts())

# Check MCI subjects
print("\n--- MCI Subjects at Baseline ---")
mci_baseline = baseline[baseline['DX_bl'].str.contains('MCI', na=False)]
print(f"MCI subjects at baseline: {len(mci_baseline)}")

# Check which have complete biomarker data
biomarkers = ['Hippocampus', 'Ventricles', 'Entorhinal', 'MidTemp', 'Fusiform', 'WholeBrain']

print("\n--- Biomarker Availability (MCI subjects) ---")
mci_all = df[df['PTID'].isin(mci_baseline['PTID'])]

for bio in biomarkers:
    available = mci_all[bio].notna().sum()
    subjects_with = mci_all[mci_all[bio].notna()]['PTID'].nunique()
    print(f"  {bio}: {subjects_with} subjects have data")

# Check subjects with at least 2 visits
print("\n--- Subjects with Longitudinal Data (≥2 visits) ---")
visit_counts = mci_all.groupby('PTID').size()
longitudinal = visit_counts[visit_counts >= 2]
print(f"MCI subjects with ≥2 visits: {len(longitudinal)}")

# Check with complete biomarkers at both visits
print("\n--- Subjects with Complete Biomarkers at 2+ Visits ---")
complete_subjects = 0
for subj in longitudinal.index:
    subj_data = mci_all[mci_all['PTID'] == subj]
    # Check baseline and at least one followup
    visits_with_hippo = subj_data['Hippocampus'].notna().sum()
    if visits_with_hippo >= 2:
        complete_subjects += 1
        
print(f"Subjects with Hippocampus at ≥2 visits: {complete_subjects}")

# Current vs Potential
print("\n" + "="*60)
print("POTENTIAL DATA EXPANSION")
print("="*60)
print(f"\nCurrently using: 301 subjects (require ResNet + all biomarkers)")
print(f"Potential with biomarkers only: {complete_subjects}+ subjects")
print(f"\nThis is {complete_subjects/301*100:.0f}% more data!")
