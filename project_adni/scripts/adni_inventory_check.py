"""
ADNI Inventory Check
Scans multiple directories for NIfTI files and matches them against the ADNI CSV registry.
"""
import os
import glob
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
SEARCH_PATHS = [
    r"D:\discs\ADNI",
    r"C:\Users\gener\Downloads"
]
CSV_PATH = r"D:\discs\ADNI\ADNI1_Complete_1Yr_1.5T_12_19_2025.csv"

def find_nifti_files(roots):
    """Recursively find all .nii files in the given root directories."""
    nifti_files = {}
    print(f"Scanning for NIfTI files in: {roots}")
    
    for root_dir in roots:
        if not os.path.exists(root_dir):
            print(f"Warning: Path not found: {root_dir}")
            continue
            
        print(f"Crawling {root_dir}...")
        count_in_dir = 0
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".nii"):
                    # ADNI filenames usually contain the Image ID, e.g., "..._I63874.nii"
                    # We map Image ID -> Full Path
                    # Extract Image ID (Ixxxxx) from filename
                    parts = file.replace(".nii", "").split("_")
                    # robust search for I-number
                    img_id = next((p for p in parts if p.startswith("I") and p[1:].isdigit()), None)
                    
                    if img_id:
                        nifti_files[img_id] = os.path.join(root, file)
                        count_in_dir += 1
                        
        print(f"  -> Found {count_in_dir} NIfTI files in {root_dir}")
    
    print(f"Total Unique NIfTI Files Found: {len(nifti_files)}")
    return nifti_files

def check_inventory():
    # 1. Load CSV
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"CSV Entries: {len(df)}")
    
    # Clean ID column
    if "Image Data ID" in df.columns:
        df["ImageID"] = df["Image Data ID"].astype(str).str.strip()
    else:
        print("Error: 'Image Data ID' column not found in CSV.")
        return

    # 2. Find Files
    found_files = find_nifti_files(SEARCH_PATHS)
    
    # 3. Match
    df["File_Path"] = df["ImageID"].map(found_files)
    df["Found"] = df["File_Path"].notna()
    
    found_count = df["Found"].sum()
    missing_count = len(df) - found_count
    
    # 4. Count Subjects
    # Filter to only found rows
    found_df = df[df["Found"] == True]
    unique_subjects = found_df["Subject"].nunique()
    total_subjects_csv = df["Subject"].nunique()

    print("\n--- INVENTORY REPORT ---")
    print(f"Total Records in CSV: {len(df)}")
    print(f"Files Found on Disk:  {found_count} ({found_count/len(df)*100:.1f}%)")
    print(f"Files Missing:        {missing_count}")
    print(f"Unique Subjects Found: {unique_subjects} / {total_subjects_csv}")
    
    # 4. Save Validated List
    output_path = r"D:\discs\adni_validated_inventory.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved validated inventory to: {output_path}")
    
    # Preview locations
    if found_count > 0:
        print("\nSample locations:")
        print(df[df["Found"]]["File_Path"].head().values)

if __name__ == "__main__":
    check_inventory()
