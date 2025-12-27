"""
Longitudinal ADNI Experiment - Data Inventory
==============================================
Scans the ADNI data folder and builds a complete subject/visit inventory.

Output: data/processed/subject_inventory.csv
"""

import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configuration
ADNI_DATA_PATH = r"C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T\ADNI"
OUTPUT_PATH = r"D:\discs\project_longitudinal\data\processed\subject_inventory.csv"

def parse_date_from_path(path_str):
    """Extract date from ADNI folder path like '2006-09-07_09_36_49.0'"""
    # Look for date pattern YYYY-MM-DD
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', path_str)
    if match:
        try:
            return datetime.strptime(match.group(0), "%Y-%m-%d")
        except:
            return None
    return None

def extract_image_id(nii_path):
    """Extract Image ID from NIfTI filename like 'ADNI_002_S_0295_MR_..._I89575.nii'"""
    match = re.search(r'I(\d+)\.nii', nii_path)
    if match:
        return f"I{match.group(1)}"
    return None

def build_inventory():
    print(f"Scanning ADNI data at: {ADNI_DATA_PATH}")
    
    if not os.path.exists(ADNI_DATA_PATH):
        raise FileNotFoundError(f"ADNI data path not found: {ADNI_DATA_PATH}")
    
    inventory = []
    subject_folders = [f for f in os.listdir(ADNI_DATA_PATH) 
                       if os.path.isdir(os.path.join(ADNI_DATA_PATH, f)) 
                       and re.match(r'^\d{3}_S_\d{4}$', f)]
    
    print(f"Found {len(subject_folders)} subject folders")
    
    for i, subject_id in enumerate(subject_folders):
        subject_path = os.path.join(ADNI_DATA_PATH, subject_id)
        
        # Find all NIfTI files for this subject
        for root, dirs, files in os.walk(subject_path):
            for file in files:
                if file.endswith('.nii'):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, ADNI_DATA_PATH)
                    
                    # Parse scan date from path
                    scan_date = parse_date_from_path(full_path)
                    
                    # Extract image ID
                    image_id = extract_image_id(file)
                    
                    # Determine processing type from folder name
                    processing_type = "Unknown"
                    parts = relative_path.split(os.sep)
                    if len(parts) > 1:
                        processing_type = parts[1]  # e.g., MPR__GradWarp__N3__Scaled
                    
                    inventory.append({
                        'subject_id': subject_id,
                        'nii_path': full_path,
                        'relative_path': relative_path,
                        'scan_date': scan_date,
                        'image_id': image_id,
                        'processing_type': processing_type,
                        'filename': file
                    })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(subject_folders)} subjects...")
    
    # Create DataFrame
    df = pd.DataFrame(inventory)
    
    # Sort by subject and date
    df = df.sort_values(['subject_id', 'scan_date']).reset_index(drop=True)
    
    # Add visit number per subject
    df['visit_num'] = df.groupby('subject_id').cumcount() + 1
    
    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("INVENTORY SUMMARY")
    print("="*60)
    print(f"Total subjects:        {df['subject_id'].nunique()}")
    print(f"Total NIfTI scans:     {len(df)}")
    print(f"Avg scans per subject: {len(df) / df['subject_id'].nunique():.2f}")
    print(f"\nScans per subject distribution:")
    visit_counts = df.groupby('subject_id').size()
    print(f"  Min:    {visit_counts.min()}")
    print(f"  Max:    {visit_counts.max()}")
    print(f"  Median: {visit_counts.median():.0f}")
    print(f"\nDate range:")
    print(f"  Earliest: {df['scan_date'].min()}")
    print(f"  Latest:   {df['scan_date'].max()}")
    print(f"\nOutput saved to: {OUTPUT_PATH}")
    print("="*60)
    
    return df

if __name__ == "__main__":
    df = build_inventory()
