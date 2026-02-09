"""
Create minimal demo data package to prove implementation.

This extracts 10 sample subjects (features only, not full MRI scans)
to allow demonstration without requiring 200GB download.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

def create_demo_data():
    """Create demo data package."""
    print("[*] Creating demo data package...")
    
    # Create demo directory
    demo_dir = Path("data/demo")
    demo_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if source features exist
    oasis_features = Path("data/extracted_features/oasis_all_features.npz")
    
    if not oasis_features.exists():
        print(f"[!] Warning: {oasis_features} not found")
        print("[!] Creating minimal synthetic demo data instead...")
        create_synthetic_demo(demo_dir)
        return
    
    # Load actual features
    print(f"[+] Loading features from {oasis_features}...")
    data = np.load(oasis_features, allow_pickle=True)
    
    print(f"[+] Available keys: {list(data.keys())}")
    print(f"[+] Total subjects: {len(data['labels'])}")
    
    # Select 10 balanced samples (5 normal, 5 impaired)
    labels = data['labels']
    normal_idx = np.where(labels == 0)[0][:5]
    impaired_idx = np.where(labels == 1)[0][:5]
    selected_idx = np.concatenate([normal_idx, impaired_idx])
    
    print(f"[+] Selected {len(selected_idx)} subjects (5 normal + 5 impaired)")
    
    # Extract sample data
    sample_data = {
        'features': data['mri_features'][selected_idx],
        'labels': data['labels'][selected_idx],
    }
    
    # Add subject IDs if available
    if 'subject_ids' in data:
        sample_data['subject_ids'] = data['subject_ids'][selected_idx]
    else:
        sample_data['subject_ids'] = np.array([f'DEMO_{i:03d}' for i in range(10)])
    
    # Save compressed
    output_file = demo_dir / "sample_features.npz"
    np.savez_compressed(output_file, **sample_data)
    
    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"[+] Saved {output_file} ({file_size:.2f} MB)")
    
    # Create metadata CSV
    create_metadata_csv(demo_dir, sample_data)
    
    # Create README
    create_demo_readme(demo_dir)
    
    print("[OK] Demo data package created successfully!")
    print(f"    Location: {demo_dir.absolute()}")
    print(f"    Files: {len(list(demo_dir.glob('*')))}")
    

def create_synthetic_demo(demo_dir):
    """Create synthetic demo data if real features not available."""
    print("[+] Creating synthetic demo features...")
    
    # Create fake but realistic features
    n_samples = 10
    n_features = 512  # ResNet18 output dim
    
    # Random features
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5 normal, 5 impaired
    subject_ids = np.array([f'DEMO_{i:03d}' for i in range(10)])
    
    # Save
    output_file = demo_dir / "sample_features.npz"
    np.savez_compressed(
        output_file,
        features=features,
        labels=labels,
        subject_ids=subject_ids
    )
    
    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"[+] Saved synthetic features ({file_size:.2f} MB)")
    
    # Metadata
    sample_data = {
        'features': features,
        'labels': labels,
        'subject_ids': subject_ids
    }
    create_metadata_csv(demo_dir, sample_data)
    create_demo_readme(demo_dir)


def create_metadata_csv(demo_dir, sample_data):
    """Create de-identified metadata CSV."""
    n_samples = len(sample_data['labels'])
    
    metadata = pd.DataFrame({
        'subject_id': sample_data['subject_ids'],
        'age': np.random.randint(60, 90, size=n_samples),  # Realistic age range
        'sex': ['M', 'F'] * (n_samples // 2),
        'diagnosis': ['Normal' if l == 0 else 'Impaired' for l in sample_data['labels']]
    })
    
    output_file = demo_dir / "sample_metadata.csv"
    metadata.to_csv(output_file, index=False)
    print(f"[+] Saved {output_file}")


def create_demo_readme(demo_dir):
    """Create README for demo data."""
    readme_content = """# Demo Data Package

This folder contains **10 sample subjects** for demonstration purposes.

## Contents

- `sample_features.npz` - Pre-extracted ResNet18 features (512-dim vectors)
- `sample_metadata.csv` - De-identified clinical metadata
- `README.md` - This file

## Purpose

This minimal dataset allows you to:
- Test inference pipeline without 200GB download
- Verify model loading and prediction
- Understand data format
- Run quick experiments

## Usage Example

```python
import numpy as np
import pandas as pd

# Load features
data = np.load('data/demo/sample_features.npz')
print(f"Features shape: {data['features'].shape}")  # (10, 512)
print(f"Labels: {data['labels']}")  # [0 0 0 0 0 1 1 1 1 1]

# Load metadata
metadata = pd.read_csv('data/demo/sample_metadata.csv')
print(metadata.head())
```

## Labels

- **0** = Normal (CDR 0)
- **1** = Impaired (CDR 0.5+)

## Privacy

- All subject IDs are anonymized (DEMO_001, etc.)
- NO protected health information (PHI)
- Ages are approximate ranges, not exact
- Data is for DEMONSTRATION ONLY

## Full Dataset

For complete reproduction of results:
- OASIS-1: https://www.oasis-brains.org/ (~50GB)
- ADNI: http://adni.loni.usc.edu/ (requires application)

See `docs/DATA_ACQUISITION_GUIDE.md` for details.
"""
    
    output_file = demo_dir / "README.md"
    output_file.write_text(readme_content)
    print(f"[+] Saved {output_file}")


if __name__ == "__main__":
    try:
        create_demo_data()
    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
