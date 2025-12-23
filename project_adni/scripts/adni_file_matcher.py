"""
ADNI Step 2: File Matching
===========================
Matches the 629 selected Image_Data_IDs to actual .nii files on disk.
Logs and drops any unmatched IDs.

Input:  adni_baseline_selection.csv (629 subjects)
Output: adni_matched_files.csv (matched subjects only)
"""
import os
import re
import pandas as pd

BASELINE_CSV = r"D:\discs\adni_baseline_selection.csv"
NIFTI_DIR = r"C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T"
OUTPUT_CSV = r"D:\discs\adni_matched_files.csv"
DROPPED_LOG = r"D:\discs\adni_dropped_ids.txt"

def extract_image_id(filename):
    """Extract Image ID (Ixxxxx) from NIfTI filename."""
    # Pattern: looks for I followed by digits, typically at end before .nii
    match = re.search(r'_I(\d+)\.nii', filename)
    if match:
        return f"I{match.group(1)}"
    return None

def build_file_index():
    """Scan disk and build a mapping of Image ID -> file path."""
    print(f"Scanning NIfTI files in: {NIFTI_DIR}")
    file_index = {}
    total_files = 0
    
    for root, _, files in os.walk(NIFTI_DIR):
        for f in files:
            if f.endswith(".nii"):
                total_files += 1
                img_id = extract_image_id(f)
                if img_id:
                    full_path = os.path.join(root, f)
                    # Store only first occurrence (avoid duplicates)
                    if img_id not in file_index:
                        file_index[img_id] = full_path
    
    print(f"Total NIfTI files scanned: {total_files}")
    print(f"Unique Image IDs indexed: {len(file_index)}")
    return file_index

def match_files():
    # 1. Load baseline selection
    print(f"\nLoading baseline selection: {BASELINE_CSV}")
    df = pd.read_csv(BASELINE_CSV)
    print(f"Subjects to match: {len(df)}")
    
    # 2. Build file index
    file_index = build_file_index()
    
    # 3. Match each Image ID
    matched_rows = []
    dropped_ids = []
    
    for _, row in df.iterrows():
        img_id = str(row['Image_Data_ID']).strip()
        
        if img_id in file_index:
            matched_rows.append({
                'Subject': row['Subject'],
                'Image_Data_ID': img_id,
                'Group': row['Group'],
                'Sex': row['Sex'],
                'Age': row['Age'],
                'Visit': row['Visit'],
                'Acq_Date': row['Acq_Date'],
                'NIfTI_Path': file_index[img_id]
            })
        else:
            dropped_ids.append({
                'Subject': row['Subject'],
                'Image_Data_ID': img_id,
                'Group': row['Group'],
                'Reason': 'No matching .nii file found on disk'
            })
    
    # 4. Save matched results
    matched_df = pd.DataFrame(matched_rows)
    matched_df.to_csv(OUTPUT_CSV, index=False)
    
    # 5. Log dropped IDs
    if dropped_ids:
        with open(DROPPED_LOG, 'w') as f:
            f.write("DROPPED IMAGE IDs - No matching .nii file found\n")
            f.write("=" * 60 + "\n\n")
            for d in dropped_ids:
                f.write(f"Subject: {d['Subject']}, Image ID: {d['Image_Data_ID']}, Group: {d['Group']}\n")
    
    # 6. Report
    print("\n" + "=" * 60)
    print("STEP 2: FILE MATCHING REPORT")
    print("=" * 60)
    print(f"Subjects in baseline selection: {len(df)}")
    print(f"Successfully matched:           {len(matched_rows)}")
    print(f"Dropped (no file found):        {len(dropped_ids)}")
    print("")
    
    if len(matched_rows) > 0:
        matched_df_temp = pd.DataFrame(matched_rows)
        print("Class Distribution (matched subjects):")
        print(matched_df_temp['Group'].value_counts().to_string())
    
    print("")
    print(f"Output saved to: {OUTPUT_CSV}")
    
    if dropped_ids:
        print(f"Dropped IDs logged to: {DROPPED_LOG}")
        print("\nDropped subjects preview:")
        for d in dropped_ids[:10]:
            print(f"  - {d['Subject']} ({d['Image_Data_ID']})")
        if len(dropped_ids) > 10:
            print(f"  ... and {len(dropped_ids) - 10} more")
    
    print("=" * 60)
    
    return matched_df, dropped_ids

if __name__ == "__main__":
    matched, dropped = match_files()
