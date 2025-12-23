"""
ADNI CSV vs DISK Comparison (Validation)
Checks specifically if SUBJECTS found in disk folders are truly unique or duplicates.
"""
import pandas as pd
import os
import re

CSV_PATH = r"D:\discs\ADNI\ADNI1_Complete_1Yr_1.5T_12_19_2025.csv"
DISK_PATHS = [
    r"D:\discs\ADNI",
    r"C:\Users\gener\Downloads"
]

def get_subject_id(filename):
    match = re.search(r"(\d{3}_S_\d{4})", filename)
    if match: return match.group(1)
    return None

def compare_sources():
    print("--- 1. LOADING CSV ---")
    df_csv = pd.read_csv(CSV_PATH)
    csv_subjects = set(df_csv["Subject"].unique())
    print(f"Total Unique Subjects in CSV Registry: {len(csv_subjects)}")
    
    print("\n--- 2. SCANNING DISK FOR OVERLAP ---")
    subjects_in_project = set()
    subjects_in_downloads = set()
    
    # Check Project Folder
    print("Scanning Project Folder...")
    if os.path.exists(DISK_PATHS[0]):
        for root, _, files in os.walk(DISK_PATHS[0]):
            for f in files:
                if f.endswith(".nii"):
                    sid = get_subject_id(f)
                    if sid: subjects_in_project.add(sid)
    
    # Check Downloads
    print("Scanning Downloads...")
    if os.path.exists(DISK_PATHS[1]):
        for root, _, files in os.walk(DISK_PATHS[1]):
            for f in files:
                if f.endswith(".nii") and "ADNI1_Complete" in root: # Specific scope
                    sid = get_subject_id(f)
                    if sid: subjects_in_downloads.add(sid)

    print(f"\nResults:")
    print(f"Subjects physically in Project Folder:  {len(subjects_in_project)}")
    print(f"Subjects physically in Downloads:       {len(subjects_in_downloads)}")
    
    # INTERSECTION (The Proof)
    overlap = subjects_in_project.intersection(subjects_in_downloads)
    print(f"\n[CRITICAL] Overlap Count: {len(overlap)} subjects are in BOTH locations.")
    
    unique_total = subjects_in_project.union(subjects_in_downloads)
    print(f"TRUE Total Unique Subjects on Disk:   {len(unique_total)}")
    
    print("\n--- 3. MISSING DATA CHECK ---")
    missing_from_disk = csv_subjects - unique_total
    print(f"Subjects in CSV but MISSING from Disk: {len(missing_from_disk)}")
    
    print("\n--- CONCLUSION ---")
    if len(overlap) > 0:
        print(f"CONFIRMED: The project folder is a SUBSET/DUPLICATE of the Downloads data.")
        print(f"You do NOT have {len(subjects_in_project) + len(subjects_in_downloads)} unique subjects.")
        print(f"You have {len(unique_total)} unique subjects.")
    else:
        print("WOW! No overlap. You truly have unique data in both.")

if __name__ == "__main__":
    compare_sources()
