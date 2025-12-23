"""
ADNI Step 1: Baseline Scan Selection
=====================================
Selects exactly ONE scan per subject from the CSV registry.

Selection Priority:
1. Visit == 'sc' (screening/baseline)
2. If no 'sc' available, select earliest 'Acq Date'

Output: adni_baseline_selection.csv (one row per subject)
"""
import pandas as pd
from datetime import datetime

CSV_PATH = r"D:\discs\ADNI\ADNI1_Complete_1Yr_1.5T_12_19_2025.csv"
OUTPUT_PATH = r"D:\discs\adni_baseline_selection.csv"

def parse_date(date_str):
    """Parse date string to datetime for comparison."""
    try:
        # Format in CSV appears to be M/DD/YYYY or MM/DD/YYYY
        return datetime.strptime(date_str.strip(), "%m/%d/%Y")
    except:
        try:
            return datetime.strptime(date_str.strip(), "%d/%m/%Y")
        except:
            return None

def select_baseline_scans():
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Total scans in CSV: {len(df)}")
    print(f"Unique subjects: {df['Subject'].nunique()}")
    print(f"Visit codes: {df['Visit'].unique().tolist()}")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Parse dates
    df['ParsedDate'] = df['Acq Date'].apply(parse_date)
    
    selected_rows = []
    subjects_with_sc = 0
    subjects_earliest_date = 0
    
    for subject_id, group_df in df.groupby('Subject'):
        # Priority 1: Look for 'sc' (screening/baseline) visit
        sc_scans = group_df[group_df['Visit'] == 'sc']
        
        if len(sc_scans) > 0:
            # If multiple 'sc' scans, pick the earliest one
            sc_scans_sorted = sc_scans.sort_values('ParsedDate')
            selected = sc_scans_sorted.iloc[0]
            subjects_with_sc += 1
        else:
            # Priority 2: No 'sc' available, select earliest date
            sorted_by_date = group_df.sort_values('ParsedDate')
            selected = sorted_by_date.iloc[0]
            subjects_earliest_date += 1
        
        selected_rows.append({
            'Subject': subject_id,
            'Image_Data_ID': selected['Image Data ID'],
            'Group': selected['Group'],
            'Sex': selected['Sex'],
            'Age': selected['Age'],
            'Visit': selected['Visit'],
            'Acq_Date': selected['Acq Date'],
            'Description': selected['Description']
        })
    
    # Create output DataFrame
    result_df = pd.DataFrame(selected_rows)
    
    # Save to CSV
    result_df.to_csv(OUTPUT_PATH, index=False)
    
    # Report
    print("\n" + "="*60)
    print("BASELINE SELECTION REPORT")
    print("="*60)
    print(f"Total Subjects Selected:     {len(result_df)}")
    print(f"  - With 'sc' (baseline):    {subjects_with_sc}")
    print(f"  - Earliest date fallback:  {subjects_earliest_date}")
    print("")
    print("Class Distribution (selected scans):")
    print(result_df['Group'].value_counts().to_string())
    print("")
    print("Visit Distribution (selected scans):")
    print(result_df['Visit'].value_counts().to_string())
    print("")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("="*60)
    
    # Print all selected Image IDs for verification
    print("\nSELECTED IMAGE DATA IDs:")
    print("-"*60)
    image_ids = result_df['Image_Data_ID'].tolist()
    # Print in columns for readability
    for i in range(0, len(image_ids), 10):
        print("  " + ", ".join(image_ids[i:i+10]))
    
    return result_df

if __name__ == "__main__":
    result = select_baseline_scans()
