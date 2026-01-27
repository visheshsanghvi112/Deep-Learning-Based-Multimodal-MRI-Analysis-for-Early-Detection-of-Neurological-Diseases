import json

with open(r'd:\discs\project_longitudinal_fusion\results\full_cohort\full_cohort_results.json') as f:
    data = json.load(f)

rf = data['model_results']['RandomForest']
print(f"Random Forest mean_auc: {rf['mean_auc']}")
print(f"Rounded to 3 decimals: {rf['mean_auc']:.3f}")
print(f"Rounded to 2 decimals: {rf['mean_auc']:.2f}")
