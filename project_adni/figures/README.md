# Level Max Visualizations

These figures visualize the performance of the "Level Max" experiments (Late Fusion vs Attention Fusion vs MRI Only).

## Figures Created:

1.  **auc_comparison_level_max.png**: 
    - Bar chart comparing Area Under the Curve (AUC) for the three models.
    - Includes Confidence Intervals (CI) as error bars.
    - MRI Only vs Late Fusion vs Attention Fusion.

2.  **accuracy_comparison_level_max.png**:
    - Bar chart comparing classification accuracy.
    - Shows the clear improvement of fusion models over MRI Only.

3.  **combined_summary_level_max.png**:
    - A grouped bar chart showing both AUC and Accuracy side-by-side for a comprehensive view.

## Source Data:
Data was taken from `D:\discs\project_adni\results\level_max\results.json`.

| Model | AUC | Accuracy |
|-------|-----|----------|
| MRI Only | 0.643 | 62.7% |
| Late Fusion | 0.808 | 74.6% |
| Attention Fusion | 0.808 | 76.2% |
