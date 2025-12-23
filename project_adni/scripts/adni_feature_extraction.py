"""
ADNI Step 3: Subject-Wise Feature Extraction
=============================================
Extracts ResNet18 features from the 629 matched .nii files.
- Exactly ONE feature vector per subject
- No data augmentation
- No visit duplication

Input:  adni_matched_files.csv (629 subjects with NIfTI paths)
Output: adni_subject_features.csv (629 rows x [metadata + 512 features])
"""
import os
import pandas as pd
import numpy as np
import nibabel as nib
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

# --- CONFIG ---
MATCHED_CSV = r"D:\discs\adni_matched_files.csv"
OUTPUT_CSV = r"D:\discs\adni_subject_features.csv"
FAILURE_LOG = r"D:\discs\adni_extraction_failures.txt"
NUM_SLICES = 5  # Middle slices to extract

# --- MODEL SETUP ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load ResNet18 (remove final FC layer to get 512-dim embeddings)
_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
RESNET = torch.nn.Sequential(*list(_resnet.children())[:-1]).to(DEVICE).eval()

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_nifti_slices(nii_path: str, num_slices: int = 5):
    """Load middle sagittal slices from a NIfTI volume."""
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        if data is None or data.size == 0:
            return None
        
        # Get middle slice index (sagittal plane = axis 0 typically)
        mid = data.shape[0] // 2
        
        slices = []
        for i in range(-num_slices // 2, num_slices // 2 + 1):
            idx = mid + i
            if 0 <= idx < data.shape[0]:
                sl = data[idx, :, :]
                
                # Normalize to 0-255 uint8
                sl = sl.astype(np.float32)
                rng = sl.max() - sl.min()
                if rng > 1e-6:
                    sl = (sl - sl.min()) / rng
                sl = (sl * 255.0).clip(0, 255).astype(np.uint8)
                slices.append(sl)
        
        return slices if slices else None
    except Exception as e:
        print(f"Error loading {nii_path}: {e}")
        return None

def extract_features(slices):
    """Pass slices through ResNet18 and average the embeddings."""
    if not slices:
        return None
    
    embeddings = []
    with torch.no_grad():
        for sl in slices:
            # Convert numpy -> PIL -> Tensor
            img = Image.fromarray(sl).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
            
            # Forward pass
            emb = RESNET(tensor).squeeze().cpu().numpy()  # (512,)
            embeddings.append(emb)
    
    if len(embeddings) == 0:
        return None
    
    # Mean pooling across slices
    return np.mean(np.stack(embeddings, axis=0), axis=0)

def extract_all_features():
    # 1. Load matched files
    print(f"\nLoading matched files: {MATCHED_CSV}")
    df = pd.read_csv(MATCHED_CSV)
    print(f"Subjects to process: {len(df)}")
    
    # 2. Process each subject
    results = []
    failures = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        subject = row['Subject']
        nii_path = row['NIfTI_Path']
        
        # Load slices
        slices = load_nifti_slices(nii_path, NUM_SLICES)
        
        if slices is None:
            failures.append({
                'Subject': subject,
                'Image_Data_ID': row['Image_Data_ID'],
                'Reason': 'Failed to load NIfTI slices'
            })
            continue
        
        # Extract features
        features = extract_features(slices)
        
        if features is None:
            failures.append({
                'Subject': subject,
                'Image_Data_ID': row['Image_Data_ID'],
                'Reason': 'Feature extraction returned None'
            })
            continue
        
        # Build result row
        result = {
            'Subject': subject,
            'Image_Data_ID': row['Image_Data_ID'],
            'Group': row['Group'],
            'Sex': row['Sex'],
            'Age': row['Age'],
            'Visit': row['Visit']
        }
        
        # Add 512 feature columns
        for i, val in enumerate(features):
            result[f'f{i}'] = float(val)
        
        results.append(result)
    
    # 3. Save results
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)
    
    # 4. Log failures
    if failures:
        with open(FAILURE_LOG, 'w') as f:
            f.write("FEATURE EXTRACTION FAILURES\n")
            f.write("=" * 60 + "\n\n")
            for fail in failures:
                f.write(f"Subject: {fail['Subject']}, ID: {fail['Image_Data_ID']}, Reason: {fail['Reason']}\n")
    
    # 5. Report
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE EXTRACTION REPORT")
    print("=" * 60)
    print(f"Subjects processed:    {len(df)}")
    print(f"Successfully extracted: {len(results)}")
    print(f"Failures:              {len(failures)}")
    print("")
    
    # Feature matrix shape
    feature_cols = [c for c in result_df.columns if c.startswith('f')]
    print(f"Feature matrix shape:  ({len(result_df)}, {len(feature_cols)})")
    print("")
    
    if len(results) > 0:
        print("Class Distribution (final dataset):")
        print(result_df['Group'].value_counts().to_string())
    
    print("")
    print(f"Output saved to: {OUTPUT_CSV}")
    
    if failures:
        print(f"Failures logged to: {FAILURE_LOG}")
        print("\nFailed subjects:")
        for f in failures[:5]:
            print(f"  - {f['Subject']} ({f['Image_Data_ID']}): {f['Reason']}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")
    
    print("=" * 60)
    
    return result_df, failures

if __name__ == "__main__":
    features_df, failures = extract_all_features()
