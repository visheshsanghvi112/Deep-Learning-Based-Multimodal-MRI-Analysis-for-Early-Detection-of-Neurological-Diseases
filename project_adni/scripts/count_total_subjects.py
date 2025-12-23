"""
Total Subject Count Script
Scans D:/discs/ADNI and C:/Users/gener/Downloads (ADNI folders) to count UNIQUE subjects.
It extracts Subject IDs (Format: XXX_S_XXXX) from NIfTI filenames.
"""
import os
import re

PATHS_TO_SCAN = [
    r"D:\discs\ADNI",
    r"C:\Users\gener\Downloads"
]

def get_subject_id(filename):
    # Pattern to match ADNI Subject IDs: 3 digits, S, 4 digits (e.g., 002_S_0295)
    match = re.search(r"(\d{3}_S_\d{4})", filename)
    if match:
        return match.group(1)
    return None

def scan_and_count():
    all_subjects_downloads = set()
    all_subjects_discs = set()
    
    print("--- DETAILED BREAKDOWN BY FOLDER ---\n")
    
    # 1. SCAN DOWNLOADS (Granular)
    downloads_path = r"C:\Users\gener\Downloads"
    if os.path.exists(downloads_path):
        print(f"Scanning Downloads ({downloads_path})...")
        # Find all ADNI subfolders
        try:
            items = os.listdir(downloads_path)
            adni_folders = [f for f in items if "ADNI1_Complete" in f and os.path.isdir(os.path.join(downloads_path, f))]
            
            if not adni_folders:
                print("  [!] No 'ADNI1_Complete' folders found in Downloads top-level.")
            
            for folder in sorted(adni_folders):
                folder_path = os.path.join(downloads_path, folder)
                subs_in_folder = set()
                # Walk this specific folder
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith(".nii"):
                            sub_id = get_subject_id(file)
                            if sub_id:
                                subs_in_folder.add(sub_id)
                                
                print(f"  • {folder:<30} -> {len(subs_in_folder)} subjects")
                all_subjects_downloads.update(subs_in_folder)
                
        except Exception as e:
            print(f"  Error scanning Downloads: {e}")
    else:
        print(f"  Downloads path not found: {downloads_path}")

    print(f"\n  >> TOTAL UNIQUE IN DOWNLOADS: {len(all_subjects_downloads)}")

    # 2. SCAN DISCS (Project Folder)
    discs_path = r"D:\discs\ADNI"
    if os.path.exists(discs_path):
        print(f"\nScanning Project Folder ({discs_path})...")
        for root, _, files in os.walk(discs_path):
            for file in files:
                if file.endswith(".nii"):
                    sub_id = get_subject_id(file)
                    if sub_id:
                        all_subjects_discs.add(sub_id)
        print(f"  >> TOTAL UNIQUE IN PROJECT FOLDER: {len(all_subjects_discs)}")
    else:
        print(f"  Project path not found: {discs_path}")

    # 3. COMBINED TOTAL
    grand_total = all_subjects_downloads.union(all_subjects_discs)
    
    print("\n" + "="*40)
    print("FINAL INTEGRATED DATASET STATS")
    print("="*40)
    print(f"Subjects in Downloads:      {len(all_subjects_downloads)}")
    print(f"Subjects in Project Folder: {len(all_subjects_discs)}")
    print("-" * 40)
    print(f"COMBINED UNIQUE SUBJECTS:   {len(grand_total)}")
    print("="*40)
    
    return grand_total

if __name__ == "__main__":
    scan_and_count()
