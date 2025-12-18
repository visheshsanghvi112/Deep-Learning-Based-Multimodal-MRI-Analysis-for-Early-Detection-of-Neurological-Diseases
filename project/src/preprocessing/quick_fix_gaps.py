"""
================================================================================
QUICK GAP FIX - Merge Existing Features + Extract CNN Embeddings
================================================================================
Fast approach to fix critical gaps:
1. Merge existing 39 deep features with basic features
2. Extract CNN embeddings for all subjects (faster than full features)
3. Create ready-to-use feature set for retraining

Time: ~30 minutes (vs 6-12 hours for full extraction)
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torchvision import models

BASE_DIR = Path("D:/discs")
sys.path.append(str(BASE_DIR))

# Try to import MRI extraction components
try:
    from mri_feature_extraction import MRILoader, MRIPreprocessor, MRIFeatureExtractor
    MRI_AVAILABLE = True
except Exception as e:
    MRI_AVAILABLE = False
    print(f"WARNING: mri_feature_extraction not available: {e!r}")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


class QuickGapFix:
    """Quick fix for critical gaps"""
    
    def __init__(self):
        self.base_dir = BASE_DIR
        self.output_dir = BASE_DIR / "project" / "data" / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CNN extractor if available
        if MRI_AVAILABLE:
            self.mri_loader = MRILoader(str(self.base_dir))
            self.preprocessor = MRIPreprocessor()
            self.cnn_extractor = MRIFeatureExtractor()
            self.cnn_extractor.eval()
            self.cnn_extractor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            print("CNN extractor initialized (ResNet18)")
        else:
            self.mri_loader = None
            self.preprocessor = None
            self.cnn_extractor = None
    
    def merge_deep_features(self):
        """Merge existing 39 deep features with basic features"""
        print("=" * 80)
        print("STEP 1: MERGING DEEP FEATURES")
        print("=" * 80)
        
        # Load basic features
        basic_file = self.output_dir / "oasis_complete_features.csv"
        if not basic_file.exists():
            print(f"ERROR: {basic_file} not found!")
            return None
        
        basic_df = pd.read_csv(basic_file)
        print(f"Loaded basic features: {basic_df.shape}")
        
        # Load deep features
        deep_file = self.base_dir / "oasis_deep_features_ALL.csv"
        if not deep_file.exists():
            print(f"WARNING: {deep_file} not found - skipping deep features merge")
            return basic_df
        
        deep_df = pd.read_csv(deep_file)
        print(f"Loaded deep features: {deep_df.shape}")
        
        # Merge on SUBJECT_ID
        merged_df = basic_df.merge(
            deep_df, 
            on='SUBJECT_ID', 
            how='left', 
            suffixes=('', '_deep')
        )
        
        print(f"Merged features: {merged_df.shape}")
        
        # Count how many have deep features
        deep_cols = [c for c in merged_df.columns if c.endswith('_deep') or c in deep_df.columns]
        has_deep = merged_df[deep_cols].notna().any(axis=1).sum()
        print(f"Subjects with deep features: {has_deep}/{len(merged_df)}")
        
        return merged_df
    
    def extract_cnn_embeddings_batch(self, subject_ids: list, batch_size: int = 10):
        """Extract CNN embeddings for subjects (batch processing)"""
        if not MRI_AVAILABLE or not self.mri_loader or not self.cnn_extractor or not self.preprocessor:
            print("WARNING: CNN extraction not available - returning zeros")
            return {sid: np.zeros(512) for sid in subject_ids}
        
        print("=" * 80)
        print("STEP 2: EXTRACTING CNN EMBEDDINGS")
        print("=" * 80)
        print(f"Processing {len(subject_ids)} subjects...\n")
        
        embeddings_dict = {}
        success_count = 0
        
        for i, subject_id in enumerate(subject_ids, 1):
            try:
                # Load MRI volume
                volume = self.mri_loader.load_volume(subject_id)
                if volume is None:
                    embeddings_dict[subject_id] = np.zeros(512)
                    continue
                
                # Preprocess to slices tensor (N,3,H,W)
                slices_tensor = self.preprocessor.process_volume(volume)
                
                # Extract features
                with torch.no_grad():
                    features = self.cnn_extractor.extract_features(slices_tensor.to(next(self.cnn_extractor.parameters()).device))
                
                embeddings_dict[subject_id] = features
                success_count += 1
                
                if i % 10 == 0:
                    print(f"  Processed {i}/{len(subject_ids)} (success: {success_count})")
                    
            except Exception as e:
                print(f"  [ERROR] {subject_id}: {e}")
                embeddings_dict[subject_id] = np.zeros(512)
        
        print(f"\nCNN embeddings extracted: {success_count}/{len(subject_ids)}")
        return embeddings_dict
    
    def save_results(self, merged_df: pd.DataFrame, embeddings_dict: dict):
        """Save merged features and embeddings"""
        print("=" * 80)
        print("STEP 3: SAVING RESULTS")
        print("=" * 80)
        
        # Save merged features
        output_file = self.output_dir / "oasis_features_merged.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"\nMerged features saved: {output_file}")
        print(f"Shape: {merged_df.shape}")
        print(f"Features: {len(merged_df.columns)}")
        
        # Save CNN embeddings
        if embeddings_dict:
            subject_ids = list(embeddings_dict.keys())
            embeddings_array = np.array([embeddings_dict[sid] for sid in subject_ids])
            
            cnn_file = self.output_dir / "oasis_cnn_embeddings_quick.npz"
            np.savez(cnn_file,
                    embeddings=embeddings_array,
                    subject_ids=np.array(subject_ids))
            print(f"\nCNN embeddings saved: {cnn_file}")
            print(f"Shape: {embeddings_array.shape}")
            print(f"Non-zero embeddings: {(embeddings_array.sum(axis=1) != 0).sum()}/{len(subject_ids)}")
        
        print("\n" + "=" * 80)
        print("QUICK FIX COMPLETE!")
        print("=" * 80)
        print("\nNext: Retrain model with merged features + real CNN embeddings")
        print("Expected: Performance improvement (AUC: 0.76 -> 0.78-0.82)")


def main():
    """Main execution"""
    print("=" * 80)
    print("QUICK GAP FIX")
    print("=" * 80)
    print("\nThis will:")
    print("  1. Merge existing 39 deep features (5 min)")
    print("  2. Extract CNN embeddings for all subjects (20-30 min)")
    print("  3. Create ready-to-use feature set")
    print("\nTotal time: ~30 minutes\n")
    
    fixer = QuickGapFix()
    
    # Step 1: Merge deep features
    merged_df = fixer.merge_deep_features()
    if merged_df is None:
        print("ERROR: Could not merge features")
        return
    
    # Step 2: Extract CNN embeddings
    subject_ids = merged_df['SUBJECT_ID'].tolist()
    embeddings_dict = fixer.extract_cnn_embeddings_batch(subject_ids)
    
    # Step 3: Save results
    fixer.save_results(merged_df, embeddings_dict)
    
    print("\nQuick fix complete! Ready to retrain model.")


if __name__ == "__main__":
    main()

