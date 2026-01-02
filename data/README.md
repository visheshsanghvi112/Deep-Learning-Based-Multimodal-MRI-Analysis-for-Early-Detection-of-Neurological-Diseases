# Data Directory

This folder contains all raw and processed datasets used in the project.

## Directory Structure

### OASIS Dataset (disc1-12)
- **disc1-12/** - OASIS neuroimaging dataset organized across 12 disc folders
  - Contains MRI scans and associated metadata
  - Used for training and validation of the deep learning models

### ADNI Dataset
- **ADNI/** - Alzheimer's Disease Neuroimaging Initiative dataset
  - Contains longitudinal neuroimaging data
  - Includes various imaging modalities and clinical assessments
  - See `docs/ADNIMERGE_USAGE_SUMMARY.md` for detailed usage information

### Processed Data
- **extracted_features/** - Extracted features from raw neuroimaging data
  - Contains preprocessed and feature-engineered data
  - Ready for model training and analysis

## Data Access

⚠️ **Note**: The actual data files are not included in version control due to size and privacy considerations. 

To use this project:
1. Download the OASIS dataset and place it in the respective disc folders
2. Download the ADNI dataset (requires registration at adni.loni.usc.edu)
3. Run preprocessing scripts from the `scripts/` folder to generate extracted features

## Data Processing

For information on data cleaning and preprocessing steps, refer to:
- `docs/DATA_CLEANING_AND_PREPROCESSING.md`
- `scripts/README.md` for available processing scripts
