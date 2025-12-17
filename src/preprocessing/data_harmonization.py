"""
================================================================================
Data Harmonization Pipeline
================================================================================
Harmonizes OASIS-1 and ADNI datasets:
1. Feature alignment
2. Label harmonization
3. Covariate alignment
4. Combined dataset creation

Part of: Master Research Plan - Phase 1.3
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("D:/discs")
DATA_DIR = BASE_DIR / "project" / "data" / "processed"
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DataHarmonizer:
    """Harmonize OASIS-1 and ADNI datasets"""
    
    def __init__(self):
        self.oasis_data = None
        self.adni_data = None
        self.combined_data = None
        self.feature_mapping = {}
        self.label_mapping = {}
        
    def load_oasis_data(self, oasis_file: Optional[Path] = None) -> pd.DataFrame:
        """Load OASIS-1 features"""
        print("=" * 80)
        print("LOADING OASIS-1 DATA")
        print("=" * 80)
        
        if oasis_file is None:
            oasis_file = DATA_DIR / "oasis_complete_features.csv"
        
        if not oasis_file.exists():
            print(f"ERROR: OASIS data file not found: {oasis_file}")
            print("Please run oasis_expansion.py first")
            return None
        
        self.oasis_data = pd.read_csv(oasis_file)
        print(f"Loaded OASIS-1 data: {oasis_file}")
        print(f"Shape: {self.oasis_data.shape}")
        print(f"Columns: {len(self.oasis_data.columns)}")
        print(f"{'='*80}\n")
        
        return self.oasis_data
    
    def load_adni_data(self, adni_file: Optional[Path] = None) -> pd.DataFrame:
        """Load ADNI features"""
        print("=" * 80)
        print("LOADING ADNI DATA")
        print("=" * 80)
        
        if adni_file is None:
            adni_file = DATA_DIR / "adni_features.csv"
        
        if not adni_file.exists():
            print(f"WARNING: ADNI data file not found: {adni_file}")
            print("ADNI processing may not be complete yet")
            return None
        
        self.adni_data = pd.read_csv(adni_file)
        print(f"Loaded ADNI data: {adni_file}")
        print(f"Shape: {self.adni_data.shape}")
        print(f"Columns: {len(self.adni_data.columns)}")
        print(f"{'='*80}\n")
        
        return self.adni_data
    
    def align_features(self) -> Dict[str, List[str]]:
        """Align feature names between datasets"""
        print("=" * 80)
        print("FEATURE ALIGNMENT")
        print("=" * 80)
        
        if self.oasis_data is None or self.adni_data is None:
            print("ERROR: Both datasets must be loaded first")
            return {}
        
        oasis_cols = set(self.oasis_data.columns)
        adni_cols = set(self.adni_data.columns)
        
        # Common features
        common_features = oasis_cols.intersection(adni_cols)
        oasis_only = oasis_cols - adni_cols
        adni_only = adni_cols - oasis_cols
        
        print(f"\nCommon features: {len(common_features)}")
        print(f"OASIS-only features: {len(oasis_only)}")
        print(f"ADNI-only features: {len(adni_only)}")
        
        # Feature mapping
        self.feature_mapping = {
            'common': sorted(list(common_features)),
            'oasis_only': sorted(list(oasis_only)),
            'adni_only': sorted(list(adni_only))
        }
        
        print(f"\n{'='*80}\n")
        
        return self.feature_mapping
    
    def harmonize_labels(self) -> Dict[str, str]:
        """Harmonize labels between datasets"""
        print("=" * 80)
        print("LABEL HARMONIZATION")
        print("=" * 80)
        
        if self.oasis_data is None or self.adni_data is None:
            print("ERROR: Both datasets must be loaded first")
            return {}
        
        # OASIS uses CDR
        # ADNI uses Diagnosis (CN/MCI/AD)
        
        # Create unified label scheme
        label_mapping = {
            'label_cdr': 'CDR score (0, 0.5, 1.0, etc.)',
            'label_mmse': 'MMSE score (continuous)',
            'label_diagnosis': 'Diagnosis group (CN/MCI/AD)',
            'label_binary': 'Binary (Normal vs Impaired)'
        }
        
        # Map ADNI diagnosis to CDR
        if 'Diagnosis' in self.adni_data.columns:
            def map_diagnosis_to_cdr(diagnosis):
                if pd.isna(diagnosis):
                    return np.nan
                diagnosis = str(diagnosis).upper()
                if 'CN' in diagnosis or 'NORMAL' in diagnosis:
                    return 0.0
                elif 'MCI' in diagnosis:
                    return 0.5
                elif 'AD' in diagnosis or 'DEMENTED' in diagnosis:
                    return 1.0
                else:
                    return np.nan
            
            self.adni_data['label_cdr'] = self.adni_data['Diagnosis'].apply(map_diagnosis_to_cdr)
        
        # Create binary label
        if 'label_cdr' in self.oasis_data.columns:
            self.oasis_data['label_binary'] = (self.oasis_data['label_cdr'] > 0).astype(int)
        if 'label_cdr' in self.adni_data.columns:
            self.adni_data['label_binary'] = (self.adni_data['label_cdr'] > 0).astype(int)
        
        self.label_mapping = label_mapping
        
        print("Label harmonization complete")
        print(f"{'='*80}\n")
        
        return self.label_mapping
    
    def align_covariates(self):
        """Align covariates (Age, Gender, etc.)"""
        print("=" * 80)
        print("COVARIATE ALIGNMENT")
        print("=" * 80)
        
        if self.oasis_data is None or self.adni_data is None:
            print("ERROR: Both datasets must be loaded first")
            return
        
        # Standardize Age
        if 'AGE' in self.oasis_data.columns:
            self.oasis_data['Age'] = self.oasis_data['AGE']
        if 'Age' not in self.oasis_data.columns and 'AGE' not in self.oasis_data.columns:
            print("WARNING: Age not found in OASIS data")
        
        # Standardize Gender
        if 'GENDER' in self.oasis_data.columns:
            # Convert to binary (M=1, F=0)
            self.oasis_data['Gender'] = (self.oasis_data['GENDER'] == 'M').astype(int)
        elif 'M/F' in self.oasis_data.columns:
            self.oasis_data['Gender'] = (self.oasis_data['M/F'] == 'M').astype(int)
        
        if 'Gender' in self.adni_data.columns:
            # Ensure binary encoding
            if self.adni_data['Gender'].dtype == 'object':
                self.adni_data['Gender'] = (self.adni_data['Gender'].str.upper() == 'M').astype(int)
        
        print("Covariate alignment complete")
        print(f"{'='*80}\n")
    
    def create_combined_dataset(self) -> pd.DataFrame:
        """Create combined harmonized dataset"""
        print("=" * 80)
        print("CREATING COMBINED DATASET")
        print("=" * 80)
        
        if self.oasis_data is None:
            print("ERROR: OASIS data not loaded")
            return None
        
        if self.adni_data is None:
            print("WARNING: ADNI data not loaded, creating OASIS-only dataset")
            self.combined_data = self.oasis_data.copy()
            self.combined_data['dataset'] = 'OASIS-1'
            return self.combined_data
        
        # Ensure both have dataset indicator
        if 'dataset' not in self.oasis_data.columns:
            self.oasis_data['dataset'] = 'OASIS-1'
        if 'dataset' not in self.adni_data.columns:
            self.adni_data['dataset'] = 'ADNI'
        
        # Get common features
        if not self.feature_mapping:
            self.align_features()
        
        common_features = self.feature_mapping['common']
        
        # Add essential columns
        essential_cols = ['SUBJECT_ID', 'dataset']
        if 'SUBJECT_ID' not in common_features:
            common_features = ['SUBJECT_ID'] + [c for c in common_features if c != 'SUBJECT_ID']
        
        # Select common features from both datasets
        oasis_subset = self.oasis_data[[c for c in essential_cols + common_features if c in self.oasis_data.columns]]
        adni_subset = self.adni_data[[c for c in essential_cols + common_features if c in self.adni_data.columns]]
        
        # Combine
        self.combined_data = pd.concat([oasis_subset, adni_subset], ignore_index=True)
        
        print(f"Combined dataset created")
        print(f"Shape: {self.combined_data.shape}")
        print(f"OASIS subjects: {len(self.oasis_data)}")
        print(f"ADNI subjects: {len(self.adni_data)}")
        print(f"Total subjects: {len(self.combined_data)}")
        print(f"{'='*80}\n")
        
        return self.combined_data
    
    def save_harmonized_data(self):
        """Save harmonized dataset"""
        if self.combined_data is None:
            print("ERROR: No combined data to save")
            return
        
        output_file = OUTPUT_DIR / "combined_features_harmonized.csv"
        self.combined_data.to_csv(output_file, index=False)
        print(f"Harmonized dataset saved to: {output_file}")
        
        # Save metadata
        metadata_file = OUTPUT_DIR / "harmonization_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write("DATA HARMONIZATION METADATA\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total subjects: {len(self.combined_data)}\n")
            f.write(f"Total features: {len(self.combined_data.columns)}\n")
            f.write(f"OASIS subjects: {len(self.oasis_data) if self.oasis_data is not None else 0}\n")
            f.write(f"ADNI subjects: {len(self.adni_data) if self.adni_data is not None else 0}\n")
            f.write(f"\nCommon features: {len(self.feature_mapping.get('common', []))}\n")
            f.write(f"OASIS-only features: {len(self.feature_mapping.get('oasis_only', []))}\n")
            f.write(f"ADNI-only features: {len(self.feature_mapping.get('adni_only', []))}\n")
        
        print(f"Metadata saved to: {metadata_file}")


def main():
    """Main execution"""
    print("=" * 80)
    print("DATA HARMONIZATION")
    print("=" * 80)
    print("\nThis script harmonizes OASIS-1 and ADNI datasets")
    print("Part of Master Research Plan - Phase 1.3\n")
    
    harmonizer = DataHarmonizer()
    
    # Load datasets
    harmonizer.load_oasis_data()
    harmonizer.load_adni_data()  # May not exist yet
    
    # Align features
    harmonizer.align_features()
    
    # Harmonize labels
    harmonizer.harmonize_labels()
    
    # Align covariates
    harmonizer.align_covariates()
    
    # Create combined dataset
    harmonizer.create_combined_dataset()
    
    # Save
    harmonizer.save_harmonized_data()
    
    print("\n" + "=" * 80)
    print("HARMONIZATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

