import pandas as pd
import numpy as np

# Re-run the key parts silently and save results
ADNIMERGE_CSV = "D:/discs/ADNI/ADNIMERGE_23Dec2025.csv"
BASELINE_CSV = "D:/discs/adni_baseline_selection.csv"
TRAIN_CSV = "D:/discs/adni_train.csv"
TEST_CSV = "D:/discs/adni_test.csv"

# Load and merge
merge_df = pd.read_csv(ADNIMERGE_CSV, low_memory=False)
baseline_df = pd.read_csv(BASELINE_CSV)
our_subjects = set(baseline_df['Subject'].unique())

baseline_merge = merge_df[
    (merge_df['PTID'].isin(our_subjects)) & 
    (merge_df['VISCODE'].isin(['bl', 'sc', 'scmri', 'm00']))
].copy()

clinical_cols = ['PTID', 'MMSE', 'CDRSB', 'PTEDUCAT', 'AGE', 'APOE4']
clinical_df = baseline_merge[clinical_cols].copy()
clinical_df['APOE4'] = clinical_df['APOE4'].fillna(0)
clinical_df = clinical_df.dropna()
clinical_df = clinical_df.rename(columns={'PTID': 'Subject'})

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_merged = train_df.merge(clinical_df, on='Subject', how='inner')
test_merged = test_df.merge(clinical_df, on='Subject', how='inner')

# Save Level-2 datasets
train_merged.to_csv("D:/discs/adni_train_level2.csv", index=False)
test_merged.to_csv("D:/discs/adni_test_level2.csv", index=False)

print("Level-2 datasets saved:")
print(f"  Train: {len(train_merged)} subjects")
print(f"  Test:  {len(test_merged)} subjects")
print(f"  Clinical features: MMSE, CDRSB, PTEDUCAT, AGE, APOE4")
