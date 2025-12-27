"""
Longitudinal ADNI Experiment - Evaluation & Comparison
========================================================
Compares all three models and generates final report.

Output: results/comparison_report.md, results/metrics_comparison.json
"""

import os
import json
import numpy as np
import pandas as pd

# Configuration
RESULTS_DIR = r"D:\discs\project_longitudinal\results"
SINGLE_DIR = os.path.join(RESULTS_DIR, "single_scan")
DELTA_DIR = os.path.join(RESULTS_DIR, "delta_model")
SEQUENCE_DIR = os.path.join(RESULTS_DIR, "sequence_model")

def load_metrics(path):
    """Load metrics from a model directory."""
    metrics_file = os.path.join(path, 'metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def generate_comparison():
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Load all metrics
    single_metrics = load_metrics(SINGLE_DIR)
    delta_metrics = load_metrics(DELTA_DIR)
    sequence_metrics = load_metrics(SEQUENCE_DIR)
    
    comparison = {
        'Single-Scan (Baseline)': single_metrics,
        'Delta Model (Change)': delta_metrics,
        'Sequence Model (LSTM)': sequence_metrics
    }
    
    # Print comparison table
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"\n{'Model':<30} {'AUC':>10} {'AUPRC':>10} {'Accuracy':>10}")
    print("-"*60)
    
    for model_name, metrics in comparison.items():
        if metrics:
            print(f"{model_name:<30} {metrics['auc']:>10.4f} {metrics.get('auprc', 0):>10.4f} {metrics.get('accuracy', 0):>10.4f}")
        else:
            print(f"{model_name:<30} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    
    print("="*60)
    
    # Determine winner
    valid_models = {k: v for k, v in comparison.items() if v is not None}
    if valid_models:
        best_model = max(valid_models.keys(), key=lambda x: valid_models[x]['auc'])
        best_auc = valid_models[best_model]['auc']
        
        if 'Single-Scan' in best_model:
            conclusion = "Longitudinal information does NOT help - single scan is sufficient."
        else:
            single_auc = valid_models.get('Single-Scan (Baseline)', {}).get('auc', 0)
            improvement = (best_auc - single_auc) / single_auc * 100 if single_auc > 0 else 0
            conclusion = f"Longitudinal information HELPS - {improvement:.1f}% improvement over single scan."
        
        print(f"\n🏆 Best Model: {best_model} (AUC: {best_auc:.4f})")
        print(f"📊 Conclusion: {conclusion}")
    
    # Save comparison
    comparison_file = os.path.join(RESULTS_DIR, 'metrics_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Generate markdown report
    report = generate_markdown_report(comparison)
    report_file = os.path.join(RESULTS_DIR, 'comparison_report.md')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nReports saved to:")
    print(f"  - {comparison_file}")
    print(f"  - {report_file}")
    
    return comparison

def generate_markdown_report(comparison):
    """Generate a detailed markdown report."""
    
    report = """# Longitudinal ADNI Experiment Results

## Research Question
> Does observing CHANGE over time (multiple MRIs of the same person) help detect or predict dementia progression more reliably?

---

## Model Comparison

| Model | Description | AUC | AUPRC | Accuracy |
|-------|-------------|-----|-------|----------|
"""
    
    for model_name, metrics in comparison.items():
        if metrics:
            desc = {
                'Single-Scan (Baseline)': 'Uses only first visit per subject',
                'Delta Model (Change)': 'Uses baseline + followup + delta',
                'Sequence Model (LSTM)': 'Uses all visits as temporal sequence'
            }.get(model_name, 'N/A')
            
            auprc_val = metrics.get('auprc', 0)
            acc_val = metrics.get('accuracy', 0)
            report += f"| {model_name} | {desc} | {metrics['auc']:.4f} | {auprc_val:.4f} | {acc_val:.4f} |\n"
        else:
            report += f"| {model_name} | N/A | N/A | N/A | N/A |\n"
    
    # Analysis
    valid_models = {k: v for k, v in comparison.items() if v is not None}
    
    if valid_models:
        best_model = max(valid_models.keys(), key=lambda x: valid_models[x]['auc'])
        best_auc = valid_models[best_model]['auc']
        single_auc = valid_models.get('Single-Scan (Baseline)', {}).get('auc', 0)
        
        report += f"""
---

## Key Findings

### Best Performing Model
**{best_model}** with AUC = {best_auc:.4f}

### Does Longitudinal Data Help?
"""
        
        if single_auc > 0:
            if 'Single-Scan' in best_model:
                report += """
**NO** - The single-scan baseline model performs best or equally well.

This suggests that:
1. Temporal progression patterns may not add discriminative signal
2. Baseline brain state is already informative for progression prediction
3. Multi-scan models may overfit due to limited longitudinal samples
"""
            else:
                improvement = (best_auc - single_auc) / single_auc * 100
                report += f"""
**YES** - Multi-scan models show **+{improvement:.1f}%** improvement over single-scan.

This suggests that:
1. Temporal changes contain additional discriminative information
2. Observing disease trajectory helps prediction
3. Change patterns (delta) may capture early neurodegeneration
"""
    
    report += """
---

## Leakage Prevention

The following measures were taken to prevent data leakage:

1. **Subject-Level Splitting**: No subject appears in both train and test sets
2. **Future Labels Only**: Progression labels derived from FINAL diagnosis, not baseline
3. **Separate Normalization**: Training statistics used to normalize test data
4. **Isolated Experiment**: This is completely separate from baseline cross-sectional work

---

## Limitations

1. Sample size may limit sequence model capacity
2. Variable follow-up intervals not explicitly modeled
3. Missing visits not handled with sophisticated imputation
4. Single random seed - should run multiple seeds for robust comparison

---

## Conclusion

This experiment provides evidence on whether longitudinal MRI information genuinely helps early dementia detection.
The comparison between single-scan, delta, and sequence models allows us to understand WHEN and WHY multi-scan information may (or may not) add value.

**Negative results are equally valid** - if longitudinal data doesn't help, that's an important finding for the field.
"""
    
    return report

if __name__ == "__main__":
    generate_comparison()
