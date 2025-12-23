"""
ADNI Dataset Summary
Run this AFTER feature extraction is complete to see the dataset distribution.
"""
import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = r"D:\discs\extracted_features\adni_features.csv"

def analyze():
    try:
        df = pd.read_csv(FILE_PATH)
        print(f"Loaded Dataset: {len(df)} samples")
        
        # 1. Class Balance
        print("\n--- Class Distribution ---")
        counts = df["Group"].value_counts()
        print(counts)
        
        # 2. Subject Overlap
        unique_subs = df["Subject"].nunique()
        print(f"\nUnique Subjects: {unique_subs}")
        
        # 3. Missing Data
        print("\n--- Missing Data ---")
        print(df[["Age", "Sex", "Group"]].isnull().sum())
        
        # 4. Feature Stats
        feat_cols = [c for c in df.columns if c.startswith("f")]
        print(f"\nFeature Vector Size: {len(feat_cols)}")
        
        if len(counts) > 0:
            print("\nDataset is ready for training/validation!")
            print(f"Normal (CN) vs Dementia (AD+MCI) ratio: {counts.get('CN',0)} : {counts.get('MCI',0) + counts.get('AD',0)}")
            
    except FileNotFoundError:
        print(f"File not found: {FILE_PATH}")
        print("Please wait for 'adni_batch_feature_extraction.py' to finish.")

if __name__ == "__main__":
    analyze()
