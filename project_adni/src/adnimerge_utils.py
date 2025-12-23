"""
ADNIMERGE Analysis & Alignment Check
====================================
Checks whether ADNIMERGE adds relevant clinical signal to the ADNI1 1.5T dataset.
"""
import pandas as pd
import numpy as np

ADNIMERGE_CSV = r"D:\discs\ADNI\ADNIMERGE_23Dec2025.csv"
BASELINE_CSV = r"D:\discs\adni_baseline_selection.csv"

def analyze_adnimerge():
    print("="*70)
    print("ADNIMERGE ANALYSIS & ALIGNMENT CHECK")
    print("="*70)
    
    # 1. Load ADNIMERGE
    print("\n[1] Loading ADNIMERGE...")
    merge_df = pd.read_csv(ADNIMERGE_CSV, low_memory=False)
    print(f"    Total rows: {len(merge_df)}")
    print(f"    Total columns: {len(merge_df.columns)}")
    print(f"    Unique subjects (PTID): {merge_df['PTID'].nunique()}")
    
    # 2. Key clinical columns
    print("\n[2] Key Clinical Columns Available:")
    key_cols = ['PTID', 'VISCODE', 'DX', 'DX_bl', 'CDRSB', 'MMSE', 'ADAS11', 'ADAS13', 
                'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 
                'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV']
    
    for col in key_cols:
        if col in merge_df.columns:
            non_null = merge_df[col].notna().sum()
            pct = non_null / len(merge_df) * 100
            unique = merge_df[col].nunique()
            print(f"    {col:<15} {non_null:>6} non-null ({pct:>5.1f}%)  |  {unique} unique values")
        else:
            print(f"    {col:<15} NOT FOUND")
    
    # 3. Load our baseline selection
    print("\n[3] Loading our ADNI1 1.5T baseline selection...")
    baseline_df = pd.read_csv(BASELINE_CSV)
    our_subjects = set(baseline_df['Subject'].unique())
    print(f"    Our subjects: {len(our_subjects)}")
    
    # 4. Check alignment
    print("\n[4] Alignment Check...")
    
    # PTID format in ADNIMERGE vs our format
    # ADNIMERGE uses: 002_S_0295
    # Our format uses: 002_S_0295 (same!)
    merge_subjects = set(merge_df['PTID'].unique())
    
    overlap = our_subjects.intersection(merge_subjects)
    missing_from_merge = our_subjects - merge_subjects
    
    print(f"    ADNIMERGE subjects: {len(merge_subjects)}")
    print(f"    Overlap with our data: {len(overlap)} / {len(our_subjects)} ({len(overlap)/len(our_subjects)*100:.1f}%)")
    print(f"    Missing from ADNIMERGE: {len(missing_from_merge)}")
    
    if len(missing_from_merge) > 0 and len(missing_from_merge) <= 10:
        print(f"    Missing subjects: {list(missing_from_merge)[:10]}")
    
    # 5. Filter to our subjects + baseline visit
    print("\n[5] Extracting baseline data for our subjects...")
    
    # Filter to our subjects
    filtered = merge_df[merge_df['PTID'].isin(our_subjects)].copy()
    print(f"    Rows for our subjects: {len(filtered)}")
    
    # Filter to baseline visits (bl, sc, scmri)
    bl_visits = ['bl', 'sc', 'scmri', 'm00']
    baseline_merge = filtered[filtered['VISCODE'].isin(bl_visits)]
    print(f"    Baseline visit rows: {len(baseline_merge)}")
    print(f"    Unique subjects at baseline: {baseline_merge['PTID'].nunique()}")
    
    # 6. Check clinical feature completeness at baseline
    print("\n[6] Clinical Feature Completeness at Baseline:")
    
    clinical_features = {
        'MMSE': 'Mini-Mental State Exam (0-30)',
        'CDRSB': 'Clinical Dementia Rating Sum of Boxes',
        'ADAS11': 'ADAS-Cog 11-item',
        'ADAS13': 'ADAS-Cog 13-item',
        'PTEDUCAT': 'Education (years)',
        'APOE4': 'APOE4 allele count (0/1/2)',
        'Hippocampus': 'Hippocampal volume',
        'WholeBrain': 'Whole brain volume',
        'ICV': 'Intracranial volume'
    }
    
    for col, desc in clinical_features.items():
        if col in baseline_merge.columns:
            non_null = baseline_merge[col].notna().sum()
            pct = non_null / len(baseline_merge) * 100 if len(baseline_merge) > 0 else 0
            print(f"    {col:<12}: {non_null:>4}/{len(baseline_merge)} ({pct:>5.1f}%)  - {desc}")
        else:
            print(f"    {col:<12}: NOT IN DATA")
    
    # 7. Summary recommendation
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATION")
    print("="*70)
    
    coverage = len(overlap) / len(our_subjects) * 100
    
    if coverage >= 90:
        print(f"\n[OK] High alignment: {coverage:.1f}% of our subjects found in ADNIMERGE")
    else:
        print(f"\n[!] Low alignment: Only {coverage:.1f}% of our subjects found in ADNIMERGE")
    
    # Check key features
    if len(baseline_merge) > 0:
        mmse_coverage = baseline_merge['MMSE'].notna().sum() / len(baseline_merge) * 100 if 'MMSE' in baseline_merge.columns else 0
        cdrsb_coverage = baseline_merge['CDRSB'].notna().sum() / len(baseline_merge) * 100 if 'CDRSB' in baseline_merge.columns else 0
        educ_coverage = baseline_merge['PTEDUCAT'].notna().sum() / len(baseline_merge) * 100 if 'PTEDUCAT' in baseline_merge.columns else 0
        
        print(f"\nKey Feature Coverage @ Baseline:")
        print(f"  - MMSE:      {mmse_coverage:.1f}%")
        print(f"  - CDRSB:     {cdrsb_coverage:.1f}%")
        print(f"  - Education: {educ_coverage:.1f}%")
        
        if mmse_coverage > 80 and cdrsb_coverage > 80:
            print("\n[GOOD] ADNIMERGE provides valuable clinical features (MMSE, CDR)")
            print("       RECOMMEND: Merge ADNIMERGE features with our MRI features")
        else:
            print("\n[PARTIAL] Some clinical features have low coverage")
            print("         Consider imputation or limiting to complete cases")
    
    print("\n" + "="*70)
    
    return merge_df, baseline_merge

if __name__ == "__main__":
    merge_df, baseline_data = analyze_adnimerge()
