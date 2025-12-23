
import os
from pathlib import Path
import nibabel as nib
import numpy as np

BASE_PATH = Path("D:/discs/ADNI")

def scan_headers():
    print(f"Scanning headers in {BASE_PATH}...")
    
    nifti_files = list(BASE_PATH.rglob("*.nii"))
    print(f"Found {len(nifti_files)} NIfTI files.")
    
    hidden_files = list(BASE_PATH.rglob(".*"))
    # Filter out . and ..
    hidden_files = [f for f in hidden_files if f.name not in ['.', '..']]
    print(f"Found {len(hidden_files)} hidden files (potential metadata?).")
    for f in hidden_files[:10]:
        print(f"  Hidden file: {f.name}")

    print("\n--- Deep Header Inspection (Sample 20) ---")
    
    for i, nii_path in enumerate(nifti_files[:20]):
        try:
            img = nib.load(nii_path)
            hdr = img.header
            
            print(f"\nFile: {nii_path.name}")
            print(f"  Descrip: {hdr['descrip'].tobytes().decode('utf-8', errors='ignore').strip()}")
            print(f"  Aux File: {hdr['aux_file'].tobytes().decode('utf-8', errors='ignore').strip()}")
            print(f"  Intent Name: {hdr['intent_name'].tobytes().decode('utf-8', errors='ignore').strip()}")
            print(f"  Slice Code: {hdr['slice_code']}")
            print(f"  Cal Max: {hdr['cal_max']}")
            print(f"  Cal Min: {hdr['cal_min']}")
            
            # Check for extensions
            if hdr.extensions:
                print(f"  Extensions found: {len(hdr.extensions)}")
                for ext in hdr.extensions:
                    print(f"    Code: {ext.get_code()}")
                    content = ext.get_content()
                    try:
                        print(f"    Content (str): {content.decode('utf-8', errors='ignore')[:50]}...")
                    except:
                         print(f"    Content (bytes): {content[:20]}...")
            else:
                print("  No extensions.")

        except Exception as e:
            print(f"  Error reading header: {e}")

if __name__ == "__main__":
    scan_headers()
