
import pandas as pd
from pathlib import Path

ADNI_DIR = Path("D:/discs/ADNI")
NEW_CSVS = [
    "All_Subjects_Key_MRI_19Dec2025.csv",
    "All_Subjects_Key_PET_19Dec2025.csv"
]

def check_new_csvs():
    print("--- Checking New CSV Content ---")
    for filename in NEW_CSVS:
        filepath = ADNI_DIR / filename
        if not filepath.exists():
            print(f"File not found: {filename}")
            continue
            
        print(f"\nFile: {filename}")
        try:
            df = pd.read_csv(filepath, nrows=5)
            print(f"Columns: {list(df.columns)}")
            print("First row data:")
            print(df.iloc[0].to_dict())
            
            # Check for education or volumetric data
            print("\nPotential Education Columns:", [c for c in df.columns if 'edu' in c.lower()])
            print("Potential Volume Columns:", [c for c in df.columns if 'vol' in c.lower()])
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    check_new_csvs()
