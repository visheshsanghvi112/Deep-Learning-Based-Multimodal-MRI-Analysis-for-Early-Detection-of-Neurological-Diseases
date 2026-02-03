import json
import numpy as np
import scipy.stats as stats

print("-" * 50)
print("LONGITUDINAL STATISTICS VERIFICATION")
print("-" * 50)

# Load full cohort results
with open(r'd:\discs\project_longitudinal_fusion\results\full_cohort\full_cohort_results.json', 'r') as f:
    data = json.load(f)

rf_data = data['model_results']['RandomForest']
folds = rf_data['fold_aucs']

print(f"Raw Fold AUCs: {folds}")

# Calculate statistics
n = len(folds)
mean_auc = np.mean(folds)
std_auc = np.std(folds, ddof=1) # Sample standard deviation
se_auc = std_auc / np.sqrt(n)

print(f"\nCalculated Mean: {mean_auc:.5f}")
print(f"Documented Mean: {rf_data['mean_auc']:.5f}")

# Calculate 95% CI using t-distribution (correct for small sample size)
t_crit = stats.t.ppf(0.975, df=n-1)
ci_lower_t = mean_auc - (t_crit * se_auc)
ci_upper_t = mean_auc + (t_crit * se_auc)

print(f"\nCalculated 95% CI (t-dist): [{ci_lower_t:.5f}, {ci_upper_t:.5f}]")

# Calculate 95% CI using z-distribution (often used incorrectly for small samples, checking if they used this)
z_crit = 1.96
ci_lower_z = mean_auc - (z_crit * se_auc)
ci_upper_z = mean_auc + (z_crit * se_auc)

print(f"Calculated 95% CI (z-dist): [{ci_lower_z:.5f}, {ci_upper_z:.5f}]")

print(f"\nDocumented CI in JSON: [{rf_data['ci_lower']:.5f}, {rf_data['ci_upper']:.5f}]")

# Determine which method was used
if abs(ci_lower_t - rf_data['ci_lower']) < 0.001:
    print("\n✅ Verification: Method used was T-DISTRIBUTION (Correct)")
elif abs(ci_lower_z - rf_data['ci_lower']) < 0.001:
     print("\n⚠️ Verification: Method used was Z-DISTRIBUTION")
else:
     print("\n❓ Verification: Method Unclear")

print("-" * 50)
print("SAMPLE SIZE & POWER CHECK")
print("-" * 50)
n_subjects = data['n_subjects']
print(f"N Subjects: {n_subjects}")
if n_subjects == 341:
    print("✅ Sample size N=341 matches documentation.")
else:
    print(f"❌ Sample size mismatch! Found {n_subjects}")
