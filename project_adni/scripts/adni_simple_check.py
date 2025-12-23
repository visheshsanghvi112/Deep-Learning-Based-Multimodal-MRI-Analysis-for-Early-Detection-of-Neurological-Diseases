
import pandas as pd
from pathlib import Path

ADNI_DIR = Path("D:/discs/ADNI")
NEW_CSVS = [
    "All_Subjects_Key_MRI_19Dec2025.csv",
    "All_Subjects_Key_PET_19Dec2025.csv"
]

def check_new_csvs():
    print("CHECKING CSV COLUMNS")
    print("="*20)
    for filename in NEW_CSVS:
        filepath = ADNI_DIR / filename
        if not filepath.exists():
            continue
            
        print(f"\nFILE: {filename}")
        try:
            df = pd.read_csv(filepath, nrows=1)
            cols = list(df.columns)
            for c in cols:
                print(f" - {c}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    check_new_csvs()
