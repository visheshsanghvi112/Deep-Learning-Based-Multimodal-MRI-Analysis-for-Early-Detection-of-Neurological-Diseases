import pandas as pd
import json

# Load ADNI data
df = pd.read_csv('d:/discs/project_adni/data/features/subject_features.csv')

print(f"Loaded {len(df)} subjects")

# The columns are: Subject, Image_Data_ID, Group, Sex, Age, Visit, + MRI features
# Select key columns for frontend
df_subset = df[['Subject', 'Group', 'Sex', 'Age']].copy()

# Rename for frontend
df_subset = df_subset.rename(columns={
    'Subject': 'subject_id', 
    'Group': 'diagnosis',
    'Sex': 'gender',
    'Age': 'age',
})

# Clean data
df_subset = df_subset.dropna(subset=['subject_id'])

# Get unique subjects (remove duplicates from multiple visits)
df_unique = df_subset.drop_duplicates(subset=['subject_id'])

# Convert to records
records = df_unique.head(200).to_dict('records')  # First 200 for frontend

# Create summary
group_counts = df_unique['diagnosis'].value_counts().to_dict()

summary = {
    'dataset': 'ADNI-1',
    'total_subjects': len(df_unique),
    'groups': group_counts,
    'age_range': [int(df_unique['age'].min()), int(df_unique['age'].max())],
    'subjects': records
}

# Save
with open('d:/discs/project/frontend/public/adni-data.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"Created adni-data.json with {len(records)} subjects")
print(f"Groups: {group_counts}")
print(f"Age range: {summary['age_range']}")
