import os
import re

PATH = r"C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T"

def get_subject_id(filename):
    match = re.search(r"(\d{3}_S_\d{4})", filename)
    return match.group(1) if match else None

subjects = set()
nii_count = 0

print(f"Scanning: {PATH}")
for root, _, files in os.walk(PATH):
    for f in files:
        if f.endswith(".nii"):
            nii_count += 1
            sid = get_subject_id(f)
            if sid:
                subjects.add(sid)

print(f"\nTotal NIfTI files: {nii_count}")
print(f"Unique Subjects: {len(subjects)}")
