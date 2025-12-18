"""
================================================================================
Training Progress Monitor
================================================================================
Monitor training progress and visualize metrics
================================================================================
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np


def plot_training_history(history_file: Path, output_dir: Path):
    """Plot training history"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation metrics
    if history.get('val_metrics'):
        # Extract metrics
        epochs = range(len(history['val_metrics']))
        metrics_to_plot = ['cdr_mae', 'mmse_mae', 'binary_acc', 'binary_auc']
        
        for metric in metrics_to_plot:
            values = []
            for epoch_metrics in history['val_metrics']:
                if isinstance(epoch_metrics, dict) and metric in epoch_metrics:
                    values.append(epoch_metrics[metric])
                elif isinstance(epoch_metrics, str):
                    # Try to parse if it's a string representation
                    try:
                        parsed = eval(epoch_metrics)
                        if isinstance(parsed, dict) and metric in parsed:
                            values.append(parsed[metric])
                    except:
                        pass
            
            if values:
                axes[0, 1].plot(values, label=metric)
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Metric Value')
        axes[0, 1].set_title('Validation Metrics')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    plt.tight_layout()
    output_file = output_dir / "training_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    # Example usage
    history_file = Path("project/results/training_history_*.json")
    output_dir = Path("project/results/figures")
    # plot_training_history(history_file, output_dir)

