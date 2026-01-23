"""
Run Full Pipeline
=================
Execute the complete longitudinal fusion experiment.

Usage:
    python run_pipeline.py
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_script(script_name: str):
    """Run a Python script and handle errors."""
    script_path = SCRIPTS_DIR / script_name
    print(f"\n{'='*70}")
    print(f"  RUNNING: {script_name}")
    print('='*70 + "\n")
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT)
    )
    
    if result.returncode != 0:
        print(f"\n❌ Error in {script_name}. Stopping pipeline.")
        sys.exit(1)
    
    print(f"\n✅ {script_name} completed successfully!")
    return True


def main():
    """Run full pipeline."""
    print("\n" + "="*70)
    print("  LONGITUDINAL MULTIMODAL FUSION - FULL PIPELINE")
    print("="*70)
    print("""
This will run the complete experiment:
  1. Data Preparation (extract & merge features)
  2. Baseline Training (LogReg, RF, XGBoost)
  3. Deep Fusion Training (Transformer model + 5-fold CV)
  4. Comprehensive Evaluation (statistical tests)
  5. Figure Generation (publication-quality)
    """)
    
    scripts = [
        "01_prepare_data.py",
        "02_train_baselines.py",
        "03_train_fusion.py",
        "04_evaluate_all.py",
        "05_generate_figures.py"
    ]
    
    for script in scripts:
        run_script(script)
    
    print("\n" + "="*70)
    print("  🎉 FULL PIPELINE COMPLETE!")
    print("="*70)
    print("""
Results saved to:
  - results/metrics/       → JSON metrics
  - results/figures/       → Publication figures
  - results/checkpoints/   → Model weights
    """)


if __name__ == "__main__":
    main()
