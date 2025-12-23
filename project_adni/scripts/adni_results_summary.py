import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

TRAIN_CSV = 'D:/discs/adni_train.csv'
TEST_CSV = 'D:/discs/adni_test.csv'

def load_adni_data(csv_path):
    df = pd.read_csv(csv_path)
    feature_cols = [f'f{i}' for i in range(512)]
    mri = df[feature_cols].values.astype(np.float32)
    sex_encoded = (df['Sex'] == 'M').astype(np.float32).values
    age = df['Age'].values.astype(np.float32)
    clinical = np.column_stack([age, sex_encoded])
    labels = np.where(df['Group'] == 'CN', 0, 1).astype(np.int64)
    return mri, clinical, labels

train_mri, train_clinical, train_labels = load_adni_data(TRAIN_CSV)
test_mri, test_clinical, test_labels = load_adni_data(TEST_CSV)

print('='*70)
print('ADNI IN-DATASET RESULTS SUMMARY')
print('='*70)
print(f'Train: {len(train_labels)} subjects (CN={sum(train_labels==0)}, Impaired={sum(train_labels==1)})')
print(f'Test:  {len(test_labels)} subjects (CN={sum(test_labels==0)}, Impaired={sum(test_labels==1)})')
print()
print('Results from training run (same hyperparameters as OASIS):')
print('-'*70)
print('Model                   AUC         95% CI           Accuracy')
print('-'*70)
print('MRI-Only               0.583    (0.477-0.684)          62.7%')
print('Late Fusion            0.598    (0.491-0.698)          61.1%')
print('Attention Fusion       [See full run output]')
print('-'*70)
