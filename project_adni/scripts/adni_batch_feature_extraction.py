"""
ADNI Distributed Feature Extraction
Reads the validated inventory and extracts ResNet18 features from NIfTI files.
Supports processing files from both workspace and Downloads folder.
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
INVENTORY_PATH = r"D:\discs\adni_validated_inventory.csv"
OUTPUT_DIR = r"D:\discs\extracted_features"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "adni_features.csv")
BATCH_SIZE = 10  # Save every N files

# --- MODEL SETUP ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load ResNet18 (Feature Extractor)
# We remove the last fc layer to get 512-dim embeddings
try:
    _resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    RESNET = torch.nn.Sequential(*list(_resnet.children())[:-1]).to(DEVICE).eval()
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback for offline/no-internet environments if needed, but weights usually cached
    # For now assuming internet or cache exists.

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_nifti_slices(nii_path: str, num_slices: int = 5):
    """Load middle sagittal neighborhood slices from a NIfTI volume."""
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # Check integrity
        if data is None or data.size == 0:
            return None

        # Middle slice index (Sagittal plane usually axis 0 or 2 depending on orientation)
        # ADNI MPRAGE is typically (192, 192, 160) or similar.
        # We'll assume standard orientation or simple heuristic: mid-point of axis 0
        mid = data.shape[0] // 2
        
        slices = []
        for i in range(-num_slices // 2, num_slices // 2 + 1):
            idx = mid + i
            if 0 <= idx < data.shape[0]:
                # Extract 2D slice
                sl = data[idx, :, :]
                
                # Normalize to 0-255 uint8
                sl = sl.astype(np.float32)
                rng = sl.max() - sl.min()
                if rng > 1e-6:
                    sl = (sl - sl.min()) / rng
                sl = (sl * 255.0).clip(0, 255).astype(np.uint8)
                slices.append(sl)
                
        return slices
    except Exception as e:
        print(f"Error reading {nii_path}: {e}")
        return None

def extract_features(slices):
    """Pass slices through ResNet18 and average the embeddings."""
    if not slices:
        return np.zeros(512, dtype=np.float32)
        
    embeddings = []
    with torch.no_grad():
        for sl in slices:
            # Convert numpy -> PIL -> Tensor
            img = Image.fromarray(sl).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
            
            # Forward pass
            emb = RESNET(tensor).squeeze().cpu().numpy() # (512,)
            embeddings.append(emb)
            
    # Mean pooling
    if len(embeddings) == 0:
        return np.zeros(512, dtype=np.float32)
    return np.mean(np.stack(embeddings, axis=0), axis=0)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Load Inventory
    df = pd.read_csv(INVENTORY_PATH)
    valid_df = df[df["Found"] == True].copy()
    print(f"Processing {len(valid_df)} files from inventory.")
    
    # 2. Check for existing progress
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        if "ImageID" in existing_df.columns:
            processed_ids = set(existing_df["ImageID"].astype(str))
        print(f"Resuming... {len(processed_ids)} already processed.")
    
    # 3. Processing Loop
    new_results = []
    
    # Iterate with index to update original df potentially or just collect results
    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df)):
        img_id = str(row["ImageID"])
        file_path = row["File_Path"]
        
        if img_id in processed_ids:
            continue
            
        # Extract
        slices = load_nifti_slices(file_path)
        if slices:
            feat = extract_features(slices)
            
            # Build result row
            # Copy relevant metadata from inventory
            res = {
                "ImageID": img_id,
                "Subject": row.get("Subject", ""),
                "Group": row.get("Group", ""),
                "Sex": row.get("Sex", ""),
                "Age": row.get("Age", 0),
                "Visit": row.get("Visit", ""),
                "Path": file_path
            }
            # Add features f0..f511
            for i, f_val in enumerate(feat):
                res[f"f{i}"] = f_val
                
            new_results.append(res)
            
        # Save Batch
        if len(new_results) >= BATCH_SIZE:
            save_batch(new_results)
            new_results = []
            
    # Final Save
    if new_results:
        save_batch(new_results)
    
    print("Done!")

def save_batch(rows):
    """Append batch to CSV."""
    df_batch = pd.DataFrame(rows)
    # Check if file exists to determine if we write header
    mode = 'a' if os.path.exists(OUTPUT_FILE) else 'w'
    header = not os.path.exists(OUTPUT_FILE)
    df_batch.to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)
    print(f"Saved {len(rows)} records.")

if __name__ == "__main__":
    main()
