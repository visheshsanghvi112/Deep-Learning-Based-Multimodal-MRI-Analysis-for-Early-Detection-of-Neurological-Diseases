"""
FINAL SANITY AUDIT SCRIPT
=========================
Performs a rigorous check of data integrity, label consistency, and leakage risks.
Does not train; just inspects.

Author: Research Pipeline
Date: December 2025
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Paths
OASIS_NPZ = "D:/discs/extracted_features/oasis_all_features.npz"
ADNI_FEAT_CSV = "D:/discs/project_adni/data/features/subject_features.csv"
ADNI_L1_TRAIN = "D:/discs/project_adni/data/features/train_level1.csv"
ADNI_L1_TEST  = "D:/discs/project_adni/data/features/test_level1.csv"
ADNI_L2_TRAIN = "D:/discs/project_adni/data/features/train_level2.csv"
ADNI_MERGE = "D:/discs/project_adni/data/csv/ADNIMERGE.csv"

REPORT = []

def log(msg, type="INFO"):
    symbol = "✔️" if type=="INFO" else "⚠️" if type=="WARN" else "📝"
    line = f"{symbol} [{type}] {msg}"
    print(line)
    REPORT.append(line)

def check_oasis():
    log("Checking OASIS Pipeline...", "INFO")
    
    if not os.path.exists(OASIS_NPZ):
        log(f"OASIS .npz file NOT FOUND: {OASIS_NPZ}", "WARN")
        return

    data = np.load(OASIS_NPZ, allow_pickle=True)
    mri = data['mri_features']
    clin = data['clinical_features']
    labels = data['labels']
    
    log(f"OASIS Subject Count: {len(labels)}", "INFO")
    log(f"OASIS MRI Shape: {mri.shape} (Expected N x 512)", "INFO")
    
    # Check dimensionality
    if mri.shape[1] != 512:
        log(f"OASIS MRI Dim Mismatch: Found {mri.shape[1]}, Expected 512", "WARN")
        
    # Check labels
    # Use string conversion to handle mixed types safely
    labels_str = [str(l) for l in labels]
    unique_labels = np.unique(labels_str)
    log(f"OASIS Raw Labels (Unique): {unique_labels}", "INFO")
    # Expected: 0, 0.5, 1, 2, 3? Or just 0, 0.5?
    # Our pipeline usually filters to 0 and 0.5.
    
    # Check for NaN / Zero embeddings
    zeros = np.sum(np.linalg.norm(mri, axis=1) == 0)
    if zeros > 0:
        log(f"OASIS: {zeros} subjects have ZERO embeddings!", "WARN")
    
    # Check Duplicate subjects? 
    # .npz doesn't have IDs. Can't check subject overlap internally.
    # Risk: Implicit assumption that rows are unique subjects.
    log("OASIS .npz lacks Subject IDs - Implicitly assuming uniqueness.", "WARN")

def check_adni():
    log("Checking ADNI Pipeline...", "INFO")
    
    # Load Main Features
    feat_df = pd.read_csv(ADNI_FEAT_CSV)
    log(f"ADNI Total Subjects (Feature CSV): {len(feat_df)}", "INFO")
    
    # Check Uniqueness
    if feat_df['Subject'].is_unique:
        log("ADNI Subject IDs are UNIQUE.", "INFO")
    else:
        dupes = feat_df[feat_df['Subject'].duplicated()]['Subject'].unique()
        log(f"ADNI Subject Duplication Detected! {len(dupes)} duplicates: {dupes}", "WARN")
        
    # Check Dimensions
    feat_cols = [c for c in feat_df.columns if c.startswith('f')]
    log(f"ADNI Feature Columns: {len(feat_cols)} (Expected 512)", "INFO")
    if len(feat_cols) != 512:
        log("ADNI Feature Count Mismatch!", "WARN")
        
    # Check L1 Splits
    train_l1 = pd.read_csv(ADNI_L1_TRAIN)
    test_l1 = pd.read_csv(ADNI_L1_TEST)
    
    log(f"ADNI L1 Train: {len(train_l1)}, Test: {len(test_l1)}", "INFO")
    
    # Overlap Check
    train_ids = set(train_l1['Subject'])
    test_ids = set(test_l1['Subject'])
    overlap = train_ids.intersection(test_ids)
    
    if len(overlap) == 0:
        log("ADNI Train/Test Split NO Overlap.", "INFO")
    else:
        log(f"ADNI Train/Test LEAKAGE! {len(overlap)} subjects in both.", "WARN")
        
    # Label Consistency
    train_grps = train_l1['Group'].unique()
    log(f"ADNI Groups found: {train_grps}", "INFO")
    
    # Check Feature Leakage (Level 1)
    # Ensure no MMSE/CDRSB/ADAS in L1 files
    leakage_cols = ['MMSE', 'CDRSB', 'ADAS11', 'ADAS13']
    for col in leakage_cols:
        if col in train_l1.columns:
            log(f"POTENTIAL L1 LEAKAGE: Found {col} in L1 Train CSV!", "WARN")
    
    # Check Feature Leakage (Level 2)
    # Should have them
    train_l2 = pd.read_csv(ADNI_L2_TRAIN)
    has_clinical = all(col in train_l2.columns for col in leakage_cols[:2]) # Check MMSE, CDRS
    if has_clinical:
        log("ADNI Level 2 correctly contains clinical features.", "INFO")
    else:
        log("ADNI Level 2 MISSING clinical features?", "WARN")

def check_cross_dataset():
    log("Checking Cross-Dataset Risks...", "INFO")
    
    # Definition mismatch risk
    log("Risk: OASIS Class 1 is CDR 0.5 (MCI). ADNI Class 1 is MCI+AD.", "WARN")
    log("      Implies label shift. ADNI 'Sick' is sicker than OASIS 'Sick'.", "INFO")
    
    # Feature scaling risk
    log("Risk: Education Scaling. OASIS/ADNI units must match.", "INFO")
    # I already investigated this mentally, but let's log it as a documented decision.

def main():
    print("Running Final Audit...")
    print("-" * 50)
    check_oasis()
    print("-" * 50)
    check_adni()
    print("-" * 50)
    check_cross_dataset()
    print("-" * 50)
    
    # Save Report
    with open("D:/discs/project_adni/results/reports/final_audit_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
        
    print("Audit Complete. Report saved.")

if __name__ == "__main__":
    main()
