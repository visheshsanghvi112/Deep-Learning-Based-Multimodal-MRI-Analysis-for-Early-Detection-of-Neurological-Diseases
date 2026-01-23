"""Model architectures for multimodal fusion."""

from .fusion_model import MultimodalTransformerFusion, SimpleMLPFusion
from .attention import MultiHeadSelfAttention, CrossModalAttention, GatedFusion
from .baselines import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
    MLPBaseline,
    train_sklearn_baseline
)
