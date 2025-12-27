"""
Longitudinal ADNI Experiment - Feature Extraction
==================================================
Extracts ResNet18 CNN features for ALL scans (not just baseline).
Preserves temporal ordering and visit metadata.

Output: data/features/longitudinal_features.npz
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
import nibabel as nib
from tqdm import tqdm

# Configuration
DATASET_PATH = r"D:\discs\project_longitudinal\data\processed\longitudinal_dataset.csv"
OUTPUT_PATH = r"D:\discs\project_longitudinal\data\features\longitudinal_features.npz"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Image transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_resnet18():
    """Load pretrained ResNet18 as feature extractor."""
    model = models.resnet18(weights='IMAGENET1K_V1')
    # Remove final classification layer
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(DEVICE)
    return model

def extract_slices(nii_path, num_slices=9):
    """Extract representative slices from 3D MRI volume."""
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # Normalize to 0-255
        data = (data - data.min()) / (data.max() - data.min() + 1e-8) * 255
        data = data.astype(np.uint8)
        
        slices = []
        
        # Get center indices for each dimension
        x_center, y_center, z_center = [s // 2 for s in data.shape[:3]]
        offsets = [-20, 0, 20]  # Offsets from center
        
        # Sagittal slices (YZ plane, varying X)
        for offset in offsets:
            idx = max(0, min(data.shape[0]-1, x_center + offset))
            slice_2d = data[idx, :, :]
            slices.append(slice_2d)
        
        # Coronal slices (XZ plane, varying Y)
        for offset in offsets:
            idx = max(0, min(data.shape[1]-1, y_center + offset))
            slice_2d = data[:, idx, :]
            slices.append(slice_2d)
        
        # Axial slices (XY plane, varying Z)
        for offset in offsets:
            idx = max(0, min(data.shape[2]-1, z_center + offset))
            slice_2d = data[:, :, idx]
            slices.append(slice_2d)
        
        return slices
    except Exception as e:
        print(f"Error loading {nii_path}: {e}")
        return None

def extract_features_for_scan(model, nii_path):
    """Extract 512-dim features for a single scan."""
    slices = extract_slices(nii_path)
    if slices is None:
        return None
    
    slice_features = []
    
    for slice_2d in slices:
        # Convert grayscale to RGB (stack 3 times)
        rgb = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)
        
        # Apply transforms
        tensor = transform(rgb).unsqueeze(0).to(DEVICE)
        
        # Extract features
        with torch.no_grad():
            features = model(tensor)
            features = features.squeeze().cpu().numpy()
        
        slice_features.append(features)
    
    # Mean pool across slices
    features_512 = np.mean(slice_features, axis=0)
    return features_512

def extract_all_features():
    print("="*60)
    print("LONGITUDINAL FEATURE EXTRACTION")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Load dataset
    print("\n[1] Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Run data_preparation.py first! Missing: {DATASET_PATH}")
    
    dataset = pd.read_csv(DATASET_PATH)
    print(f"    Total scans to process: {len(dataset)}")
    
    # Load model
    print("\n[2] Loading ResNet18...")
    model = load_resnet18()
    print("    Model loaded successfully")
    
    # Extract features
    print("\n[3] Extracting features...")
    all_features = []
    all_subject_ids = []
    all_visit_nums = []
    all_scan_dates = []
    all_labels = []
    all_splits = []
    failed_scans = []
    
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Extracting"):
        nii_path = row['nii_path']
        
        if not os.path.exists(nii_path):
            failed_scans.append(nii_path)
            continue
        
        features = extract_features_for_scan(model, nii_path)
        
        if features is not None:
            all_features.append(features)
            all_subject_ids.append(row['subject_id'])
            all_visit_nums.append(row['visit_num'])
            all_scan_dates.append(row['scan_date'])
            all_labels.append(row['label'])
            all_splits.append(row['split'])
    
    # Convert to arrays
    features_array = np.array(all_features)
    
    # Save to npz
    print("\n[4] Saving features...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        features=features_array,
        subject_ids=np.array(all_subject_ids),
        visit_nums=np.array(all_visit_nums),
        scan_dates=np.array(all_scan_dates),
        labels=np.array(all_labels),
        splits=np.array(all_splits)
    )
    
    # Summary
    print("\n" + "="*60)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*60)
    print(f"Scans processed:   {len(all_features)}")
    print(f"Failed scans:      {len(failed_scans)}")
    print(f"Feature dimension: {features_array.shape[1]}")
    print(f"Output saved to:   {OUTPUT_PATH}")
    print(f"File size:         {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.2f} MB")
    print("="*60)
    
    if failed_scans:
        print(f"\nFailed scans (first 10):")
        for path in failed_scans[:10]:
            print(f"  {path}")
    
    return features_array

if __name__ == "__main__":
    extract_all_features()
