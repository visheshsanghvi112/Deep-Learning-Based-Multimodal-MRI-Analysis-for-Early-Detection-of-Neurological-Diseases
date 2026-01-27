import json

# Check Level-MAX
with open(r'd:\discs\project_adni\results\level_max\results.json') as f:
    lmax = json.load(f)
    
print("LEVEL-MAX VERIFICATION:")
print(f"Late Fusion AUC: {lmax['Late_Fusion']['AUC']:.4f} (claimed: 0.808)")
print(f"Attention Fusion AUC: {lmax['Attention_Fusion']['AUC']:.4f} (claimed: 0.808)")
print(f"MRI-Only AUC: {lmax['MRI_Only']['AUC']:.4f} (claimed: 0.643)")

# Check Level-1
with open(r'd:\discs\project_adni\results\level1\metrics.json') as f:
    l1 = json.load(f)
    
print("\nLEVEL-1 VERIFICATION:")
print(f"Late Fusion AUC: {l1['results']['Late Fusion']['auc']:.3f} (claimed: 0.60)")
print(f"MRI-Only AUC: {l1['results']['MRI-Only']['auc']:.3f} (claimed: 0.583)")

# Check Longitudinal
with open(r'd:\discs\project_longitudinal_fusion\results\full_cohort\full_cohort_results.json') as f:
    long = json.load(f)
    
print("\nLONGITUDINAL VERIFICATION:")
print(f"Random Forest AUC: {long['model_results']['RandomForest']['mean_auc']:.4f} (claimed: 0.848)")
print(f"Sample size: {long['n_subjects']} (claimed: 341)")
print(f"Converters: {long['n_converters']}")
print(f"Stable: {long['n_stable']}")
