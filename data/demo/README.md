# Demo Data Package

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
