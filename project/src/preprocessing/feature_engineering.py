"""
================================================================================
Feature Engineering & Selection Pipeline
================================================================================
Phase 2: Feature Engineering & Selection
- Feature prioritization (neuroanatomically grounded)
- Statistical feature selection
- Age confounding analysis
- Feature normalization

Part of: Master Research Plan - Phase 2
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("D:/discs")
DATA_DIR = BASE_DIR / "project" / "data" / "processed"
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class FeatureEngineer:
    """Feature engineering and selection for OASIS-1 dataset"""
    
    def __init__(self, data_file: Optional[Path] = None):
        self.data_file = data_file or (DATA_DIR / "oasis_complete_features_full.csv")
        self.df = None
        self.selected_features = []
        self.feature_importance = {}
        self.age_confounding_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load OASIS feature data"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        if not self.data_file.exists():
            # Try alternative file
            alt_file = DATA_DIR / "oasis_complete_features.csv"
            if alt_file.exists():
                self.data_file = alt_file
                print(f"Using alternative file: {alt_file}")
            else:
                raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        self.df = pd.read_csv(self.data_file)
        print(f"Loaded: {self.data_file}")
        print(f"Shape: {self.df.shape}")
        print(f"Subjects: {len(self.df)}")
        print(f"Features: {len(self.df.columns)}")
        print(f"{'='*80}\n")
        
        return self.df
    
    def identify_priority_features(self) -> Dict[str, List[str]]:
        """Identify neuroanatomically grounded priority features"""
        print("=" * 80)
        print("FEATURE PRIORITIZATION (NEUROANATOMICALLY GROUNDED)")
        print("=" * 80)
        
        if self.df is None:
            self.load_data()
        
        all_features = list(self.df.columns)
        
        # Tier 1: Established AD Biomarkers (MUST INCLUDE)
        tier1_keywords = [
            'hippocamp', 'medial_temporal', 'ventricle', 'ventricular',
            'posterior_cingulate', 'precuneus'
        ]
        
        # Tier 2: Global Atrophy Measures (MUST INCLUDE)
        tier2_keywords = [
            'gm_volume', 'wm_volume', 'csf_volume', 'brain_percentage',
            'etiv', 'nwbv', 'total_intracranial'
        ]
        
        # Tier 3: Regional Volumes
        tier3_keywords = [
            'frontal', 'parietal', 'temporal', 'occipital',
            'thalamus', 'basal_ganglia'
        ]
        
        # Tier 4: Derived Measures
        tier4_keywords = [
            'ratio', 'fraction', 'asymmetry', 'percentage'
        ]
        
        # Tier 5: Intensity Statistics
        tier5_keywords = [
            'intensity', 'percentile', 'skewness', 'kurtosis',
            'mean', 'std', 'median'
        ]
        
        priority_features = {
            'tier1': [],
            'tier2': [],
            'tier3': [],
            'tier4': [],
            'tier5': [],
            'other': []
        }
        
        for feature in all_features:
            feature_lower = feature.lower()
            matched = False
            
            for keyword in tier1_keywords:
                if keyword in feature_lower:
                    priority_features['tier1'].append(feature)
                    matched = True
                    break
            
            if not matched:
                for keyword in tier2_keywords:
                    if keyword in feature_lower:
                        priority_features['tier2'].append(feature)
                        matched = True
                        break
            
            if not matched:
                for keyword in tier3_keywords:
                    if keyword in feature_lower:
                        priority_features['tier3'].append(feature)
                        matched = True
                        break
            
            if not matched:
                for keyword in tier4_keywords:
                    if keyword in feature_lower:
                        priority_features['tier4'].append(feature)
                        matched = True
                        break
            
            if not matched:
                for keyword in tier5_keywords:
                    if keyword in feature_lower:
                        priority_features['tier5'].append(feature)
                        matched = True
                        break
            
            if not matched and feature not in ['SUBJECT_ID', 'dataset', 'disc_folder']:
                priority_features['other'].append(feature)
        
        # Print summary
        print("\nPriority Feature Distribution:")
        for tier, features in priority_features.items():
            print(f"  {tier.upper()}: {len(features)} features")
            if tier in ['tier1', 'tier2'] and features:
                print(f"    Examples: {features[:5]}")
        
        print(f"\n{'='*80}\n")
        
        return priority_features
    
    def analyze_age_confounding(self) -> Dict:
        """Analyze age confounding in features"""
        print("=" * 80)
        print("AGE CONFOUNDING ANALYSIS")
        print("=" * 80)
        
        if self.df is None:
            self.load_data()
        
        # Get age column
        age_col = None
        for col in ['AGE', 'Age', 'age']:
            if col in self.df.columns:
                age_col = col
                break
        
        if age_col is None:
            print("WARNING: Age column not found!")
            return {}
        
        age_data = self.df[age_col].dropna()
        
        # Get numeric features (exclude IDs, labels)
        exclude_cols = ['SUBJECT_ID', 'dataset', 'disc_folder', age_col, 'CDR', 'MMSE', 'label_cdr', 'label_binary']
        numeric_features = [col for col in self.df.columns 
                          if col not in exclude_cols 
                          and self.df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        age_correlations = {}
        for feature in numeric_features:
            feature_data = self.df[feature].dropna()
            # Align indices
            common_idx = age_data.index.intersection(feature_data.index)
            if len(common_idx) > 10:  # Need sufficient data
                age_vals = age_data.loc[common_idx]
                feat_vals = feature_data.loc[common_idx]
                corr, pval = stats.pearsonr(age_vals, feat_vals)
                age_correlations[feature] = {
                    'correlation': corr,
                    'pvalue': pval,
                    'abs_correlation': abs(corr)
                }
        
        # Sort by absolute correlation
        sorted_corrs = sorted(age_correlations.items(), 
                            key=lambda x: x[1]['abs_correlation'], 
                            reverse=True)
        
        print(f"\nTop 20 Age-Correlated Features:")
        print(f"{'Feature':<40} {'Correlation':<15} {'P-value':<15}")
        print("-" * 70)
        for feature, stats_dict in sorted_corrs[:20]:
            print(f"{feature:<40} {stats_dict['correlation']:>10.3f}     {stats_dict['pvalue']:>10.3e}")
        
        # Identify highly age-confounded features (|r| > 0.5)
        high_confounded = [f for f, s in age_correlations.items() 
                          if abs(s['correlation']) > 0.5]
        
        print(f"\nHighly Age-Confounded Features (|r| > 0.5): {len(high_confounded)}")
        if high_confounded:
            print(f"  Examples: {high_confounded[:10]}")
        
        self.age_confounding_results = {
            'correlations': age_correlations,
            'highly_confounded': high_confounded,
            'sorted': sorted_corrs
        }
        
        print(f"\n{'='*80}\n")
        
        return self.age_confounding_results
    
    def univariate_feature_selection(self, target_col: str = 'CDR', k: int = 50) -> List[str]:
        """Univariate feature selection using statistical tests"""
        print("=" * 80)
        print("UNIVARIATE FEATURE SELECTION")
        print("=" * 80)
        
        if self.df is None:
            self.load_data()
        
        # Prepare data
        if target_col not in self.df.columns:
            print(f"WARNING: Target column '{target_col}' not found!")
            print(f"Available columns: {list(self.df.columns)[:10]}...")
            return []
        
        # Get features and target
        exclude_cols = ['SUBJECT_ID', 'dataset', 'disc_folder', target_col]
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        X = self.df[feature_cols].select_dtypes(include=[np.number])
        y = self.df[target_col].dropna()
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Create binary target if needed (for classification)
        if y.dtype in [np.float64, np.float32]:
            # Check if it's binary or continuous
            unique_vals = y.unique()
            if len(unique_vals) <= 5:
                # Likely categorical, convert to binary
                y_binary = (y > 0).astype(int)
            else:
                # Continuous, use as is for regression
                y_binary = y
        else:
            y_binary = y
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
        selector.fit(X, y_binary)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        scores = selector.scores_[selected_mask]
        
        # Create feature importance dict
        feature_scores = dict(zip(selected_features, scores))
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {k} Selected Features:")
        print(f"{'Feature':<40} {'F-Score':<15}")
        print("-" * 55)
        for feature, score in sorted_features[:20]:
            print(f"{feature:<40} {score:>10.3f}")
        
        self.selected_features = selected_features
        self.feature_importance = feature_scores
        
        print(f"\n{'='*80}\n")
        
        return selected_features
    
    def normalize_features(self, method: str = 'standard') -> pd.DataFrame:
        """Normalize features"""
        print("=" * 80)
        print("FEATURE NORMALIZATION")
        print("=" * 80)
        
        if self.df is None:
            self.load_data()
        
        # Get numeric features
        exclude_cols = ['SUBJECT_ID', 'dataset', 'disc_folder', 'CDR', 'MMSE', 
                       'label_cdr', 'label_binary', 'AGE', 'Age', 'age']
        numeric_cols = [col for col in self.df.columns 
                       if col not in exclude_cols 
                       and self.df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        # Create normalized dataframe
        df_normalized = self.df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Normalize
        df_normalized[numeric_cols] = scaler.fit_transform(self.df[numeric_cols].fillna(0))
        
        print(f"Normalized {len(numeric_cols)} features using {method} scaling")
        print(f"{'='*80}\n")
        
        return df_normalized
    
    def create_feature_report(self):
        """Create comprehensive feature engineering report"""
        print("=" * 80)
        print("GENERATING FEATURE ENGINEERING REPORT")
        print("=" * 80)
        
        # Priority features
        priority = self.identify_priority_features()
        
        # Age confounding
        age_conf = self.analyze_age_confounding()
        
        # Feature selection
        selected = self.univariate_feature_selection()
        
        # Save report
        report_file = OUTPUT_DIR / "feature_engineering_report.txt"
        with open(report_file, 'w') as f:
            f.write("FEATURE ENGINEERING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("PRIORITY FEATURES:\n")
            f.write("-" * 80 + "\n")
            for tier, features in priority.items():
                f.write(f"\n{tier.upper()}: {len(features)} features\n")
                if features:
                    for feat in features[:10]:
                        f.write(f"  - {feat}\n")
            
            f.write("\n\nAGE CONFOUNDING ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Highly age-confounded features (|r| > 0.5): {len(age_conf.get('highly_confounded', []))}\n")
            if age_conf.get('highly_confounded'):
                for feat in age_conf['highly_confounded'][:20]:
                    f.write(f"  - {feat}\n")
            
            f.write("\n\nSELECTED FEATURES:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total selected: {len(selected)}\n")
            for feat in selected[:30]:
                f.write(f"  - {feat}\n")
        
        print(f"Report saved to: {report_file}")
        print(f"{'='*80}\n")
        
        return report_file


def main():
    """Main execution"""
    print("=" * 80)
    print("FEATURE ENGINEERING & SELECTION")
    print("=" * 80)
    print("\nPhase 2: Feature Engineering & Selection")
    print("Part of Master Research Plan\n")
    
    engineer = FeatureEngineer()
    
    # Load data
    engineer.load_data()
    
    # Generate report
    engineer.create_feature_report()
    
    # Normalize features
    df_normalized = engineer.normalize_features(method='standard')
    
    # Save normalized data
    output_file = OUTPUT_DIR / "oasis_features_normalized.csv"
    df_normalized.to_csv(output_file, index=False)
    print(f"Normalized features saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE!")
    print("=" * 80)
    print("\nNext: Proceed to Phase 3 - Deep Learning Model Development")


if __name__ == "__main__":
    main()

