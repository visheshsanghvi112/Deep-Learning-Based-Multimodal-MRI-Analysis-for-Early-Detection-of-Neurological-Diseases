"""
Script 01: Prepare Data
=======================
Extract and merge ResNet features with clinical biomarkers.

Usage:
    python scripts/01_prepare_data.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import load_and_prepare_data
from src.utils.helpers import set_seed, get_timestamp, save_json
from config import DATA_DIR, METRICS_DIR


def main():
    """Main data preparation pipeline."""
    print("\n" + "="*70)
    print("  LONGITUDINAL MULTIMODAL FUSION - DATA PREPARATION")
    print("="*70 + "\n")
    
    # Set seed for reproducibility
    set_seed()
    
    # Load and prepare data
    train_data, test_data, scalers = load_and_prepare_data(force_reload=True)
    
    # Print summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print(f"\nTrain set: {len(train_data['labels'])} subjects")
    print(f"  - Stable (0): {sum(train_data['labels'] == 0)}")
    print(f"  - Converter (1): {sum(train_data['labels'] == 1)}")
    
    print(f"\nTest set: {len(test_data['labels'])} subjects")
    print(f"  - Stable (0): {sum(test_data['labels'] == 0)}")
    print(f"  - Converter (1): {sum(test_data['labels'] == 1)}")
    
    print("\nFeature dimensions:")
    print(f"  - Baseline ResNet: {train_data['baseline_resnet'].shape[1]}")
    print(f"  - Followup ResNet: {train_data['followup_resnet'].shape[1]}")
    print(f"  - Delta ResNet: {train_data['delta_resnet'].shape[1]}")
    print(f"  - Baseline Bio: {train_data['baseline_bio'].shape[1]}")
    print(f"  - Followup Bio: {train_data['followup_bio'].shape[1]}")
    print(f"  - Delta Bio: {train_data['delta_bio'].shape[1]}")
    
    total_features = (
        train_data['baseline_resnet'].shape[1] * 3 +  # 3 ResNet
        train_data['baseline_bio'].shape[1] +          # baseline bio
        train_data['followup_bio'].shape[1] +          # followup bio
        train_data['delta_bio'].shape[1]               # delta bio
    )
    print(f"\nTotal features per subject: {total_features}")
    
    # Save data summary
    summary = {
        'timestamp': get_timestamp(),
        'train_samples': int(len(train_data['labels'])),
        'test_samples': int(len(test_data['labels'])),
        'train_positive_rate': float(sum(train_data['labels'] == 1) / len(train_data['labels'])),
        'test_positive_rate': float(sum(test_data['labels'] == 1) / len(test_data['labels'])),
        'feature_dimensions': {
            'baseline_resnet': int(train_data['baseline_resnet'].shape[1]),
            'followup_resnet': int(train_data['followup_resnet'].shape[1]),
            'delta_resnet': int(train_data['delta_resnet'].shape[1]),
            'baseline_bio': int(train_data['baseline_bio'].shape[1]),
            'followup_bio': int(train_data['followup_bio'].shape[1]),
            'delta_bio': int(train_data['delta_bio'].shape[1])
        },
        'total_features': total_features
    }
    
    save_json(summary, METRICS_DIR / 'data_summary.json')
    
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nNext step: python scripts/02_train_baselines.py")


if __name__ == "__main__":
    main()
