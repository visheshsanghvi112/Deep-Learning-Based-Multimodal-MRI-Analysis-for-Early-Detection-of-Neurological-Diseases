"""
ADNI Step 4A: Subject-Wise Stratified Train/Test Split
=======================================================
Performs an 80/20 train/test split within ADNI only.
- Subject-wise splitting (no subject appears in both sets)
- Stratified by diagnosis Group (CN/MCI/AD)
- No OASIS data involved

Input:  adni_subject_features.csv (629 subjects)
Output: adni_train.csv, adni_test.csv
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# --- CONFIG ---
INPUT_CSV = r"D:\discs\adni_subject_features.csv"
TRAIN_CSV = r"D:\discs\adni_train.csv"
TEST_CSV = r"D:\discs\adni_test.csv"
TEST_SIZE = 0.20
RANDOM_STATE = 42  # For reproducibility

def perform_split():
    # 1. Load features
    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total subjects: {len(df)}")
    print(f"Class distribution:\n{df['Group'].value_counts().to_string()}\n")
    
    # 2. Verify one row per subject
    assert df['Subject'].nunique() == len(df), "ERROR: Multiple rows per subject detected!"
    print("✓ Verified: Exactly one row per subject")
    
    # 3. Stratified split by Group
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df['Group'],
        random_state=RANDOM_STATE
    )
    
    # 4. Save splits
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    
    # 5. Report
    print("\n" + "=" * 60)
    print("STEP 4A: ADNI TRAIN/TEST SPLIT REPORT")
    print("=" * 60)
    print(f"Split ratio: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} (train/test)")
    print(f"Random state: {RANDOM_STATE}")
    print("")
    
    print(f"TRAINING SET: {len(train_df)} subjects")
    print(train_df['Group'].value_counts().to_string())
    print("")
    
    print(f"TEST SET: {len(test_df)} subjects")
    print(test_df['Group'].value_counts().to_string())
    print("")
    
    # Verify no subject overlap
    train_subjects = set(train_df['Subject'])
    test_subjects = set(test_df['Subject'])
    overlap = train_subjects.intersection(test_subjects)
    
    if len(overlap) == 0:
        print("✓ Verified: NO subject overlap between train and test")
    else:
        print(f"⚠ WARNING: {len(overlap)} subjects appear in both sets!")
    
    # Verify class proportions are maintained
    print("\nClass proportion verification:")
    for group in ['CN', 'MCI', 'AD']:
        orig_pct = (df['Group'] == group).mean() * 100
        train_pct = (train_df['Group'] == group).mean() * 100
        test_pct = (test_df['Group'] == group).mean() * 100
        print(f"  {group}: Original={orig_pct:.1f}%, Train={train_pct:.1f}%, Test={test_pct:.1f}%")
    
    print("")
    print(f"Train set saved to: {TRAIN_CSV}")
    print(f"Test set saved to:  {TEST_CSV}")
    print("=" * 60)
    
    print("\n⚠ NOTE: No model training performed. Awaiting explicit instruction.")
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = perform_split()
