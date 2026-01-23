"""
Utility Helper Functions
========================
"""

import os
import json
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RANDOM_SEED, DEVICE


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to: {seed}")


def get_device(preferred: str = DEVICE) -> str:
    """Get available device."""
    if preferred == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_parameters(n_params: int) -> str:
    """Format parameter count as human-readable string."""
    if n_params >= 1e9:
        return f"{n_params/1e9:.2f}B"
    elif n_params >= 1e6:
        return f"{n_params/1e6:.2f}M"
    elif n_params >= 1e3:
        return f"{n_params/1e3:.2f}K"
    return str(n_params)


def save_json(data: Dict[str, Any], path: Path):
    """Save dictionary to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(path, 'w') as f:
        json.dump(convert(data), f, indent=2)
    
    print(f"Saved: {path}")


def load_json(path: Path) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_model_summary(model: torch.nn.Module, input_shapes: Optional[Dict] = None):
    """Print model architecture summary."""
    print("="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"\nArchitecture:\n{model}")
    print(f"\nTotal parameters: {format_parameters(count_parameters(model, trainable_only=False))}")
    print(f"Trainable parameters: {format_parameters(count_parameters(model, trainable_only=True))}")
    print("="*60)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.elapsed = time.time() - self.start_time
        if self.name:
            print(f"{self.name}: {self.elapsed:.2f}s")
