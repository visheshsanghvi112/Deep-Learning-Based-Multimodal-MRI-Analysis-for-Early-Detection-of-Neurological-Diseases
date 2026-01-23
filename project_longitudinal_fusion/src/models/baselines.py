"""
Baseline Models for Comparison
==============================
Traditional ML baselines: LogReg, RF, XGBoost, SVM, MLP.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class BaselineModel:
    """Base class for sklearn-based baseline models."""
    
    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self.scaler = StandardScaler()
        
    def prepare_features(
        self,
        data: Dict[str, np.ndarray],
        include_resnet: bool = True,
        include_bio: bool = True
    ) -> np.ndarray:
        """Prepare features from data dictionary."""
        features = []
        
        if include_resnet:
            features.extend([
                data['baseline_resnet'],
                data['followup_resnet'],
                data['delta_resnet']
            ])
            
        if include_bio:
            features.extend([
                data['baseline_bio'],
                data['followup_bio'],
                data['delta_bio']
            ])
            
        return np.concatenate(features, axis=1)
    
    def fit(
        self,
        train_data: Dict[str, np.ndarray],
        include_resnet: bool = True,
        include_bio: bool = True
    ):
        """Fit the model."""
        X = self.prepare_features(train_data, include_resnet, include_bio)
        y = train_data['labels']
        
        # Standardize
        X = self.scaler.fit_transform(X)
        
        self.model.fit(X, y)
        return self
    
    def predict_proba(
        self,
        data: Dict[str, np.ndarray],
        include_resnet: bool = True,
        include_bio: bool = True
    ) -> np.ndarray:
        """Get probability predictions."""
        X = self.prepare_features(data, include_resnet, include_bio)
        X = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            # For SVM with probability=False
            return self.model.decision_function(X)
    
    def predict(
        self,
        data: Dict[str, np.ndarray],
        include_resnet: bool = True,
        include_bio: bool = True
    ) -> np.ndarray:
        """Get class predictions."""
        X = self.prepare_features(data, include_resnet, include_bio)
        X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def evaluate(
        self,
        data: Dict[str, np.ndarray],
        include_resnet: bool = True,
        include_bio: bool = True
    ) -> Dict[str, float]:
        """Evaluate model."""
        y_true = data['labels']
        y_pred = self.predict(data, include_resnet, include_bio)
        y_prob = self.predict_proba(data, include_resnet, include_bio)
        
        return {
            'auc': roc_auc_score(y_true, y_prob),
            'accuracy': accuracy_score(y_true, y_pred)
        }


def LogisticRegressionBaseline(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42
) -> BaselineModel:
    """Create Logistic Regression baseline."""
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs',
        class_weight='balanced'
    )
    return BaselineModel(model, "LogisticRegression")


def RandomForestBaseline(
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    random_state: int = 42
) -> BaselineModel:
    """Create Random Forest baseline."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    return BaselineModel(model, "RandomForest")


def XGBoostBaseline(
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> BaselineModel:
    """Create XGBoost baseline."""
    if not HAS_XGBOOST:
        print("Warning: XGBoost not installed, using GradientBoosting instead")
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
        return BaselineModel(model, "GradientBoosting")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    return BaselineModel(model, "XGBoost")


def SVMBaseline(
    C: float = 1.0,
    kernel: str = 'rbf',
    random_state: int = 42
) -> BaselineModel:
    """Create SVM baseline."""
    model = SVC(
        C=C,
        kernel=kernel,
        random_state=random_state,
        probability=True,
        class_weight='balanced'
    )
    return BaselineModel(model, "SVM")


def MLPBaseline(
    hidden_layer_sizes: Tuple[int, ...] = (128, 64),
    max_iter: int = 500,
    random_state: int = 42
) -> BaselineModel:
    """Create MLP baseline."""
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1
    )
    return BaselineModel(model, "MLP")


def train_sklearn_baseline(
    model_name: str,
    train_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    include_resnet: bool = True,
    include_bio: bool = True,
    random_state: int = 42
) -> Tuple[BaselineModel, Dict[str, float]]:
    """
    Train and evaluate a sklearn baseline model.
    
    Args:
        model_name: One of 'logistic_regression', 'random_forest', 
                   'xgboost', 'svm', 'mlp'
        train_data: Training data dictionary
        test_data: Test data dictionary
        include_resnet: Whether to include ResNet features
        include_bio: Whether to include biomarker features
        random_state: Random seed
        
    Returns:
        Trained model, evaluation metrics
    """
    # Create model
    if model_name == 'logistic_regression':
        model = LogisticRegressionBaseline(random_state=random_state)
    elif model_name == 'random_forest':
        model = RandomForestBaseline(random_state=random_state)
    elif model_name == 'xgboost':
        model = XGBoostBaseline(random_state=random_state)
    elif model_name == 'svm':
        model = SVMBaseline(random_state=random_state)
    elif model_name == 'mlp':
        model = MLPBaseline(random_state=random_state)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train
    model.fit(train_data, include_resnet=include_resnet, include_bio=include_bio)
    
    # Evaluate
    train_metrics = model.evaluate(train_data, include_resnet, include_bio)
    test_metrics = model.evaluate(test_data, include_resnet, include_bio)
    
    results = {
        'model_name': model.name,
        'train_auc': train_metrics['auc'],
        'test_auc': test_metrics['auc'],
        'train_accuracy': train_metrics['accuracy'],
        'test_accuracy': test_metrics['accuracy']
    }
    
    return model, results


def train_all_baselines(
    train_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Train all baseline models with different feature combinations.
    
    Returns:
        Dictionary of results for each model configuration
    """
    results = {}
    
    model_names = ['logistic_regression', 'random_forest', 'xgboost', 'mlp']
    
    feature_configs = [
        ('resnet_only', True, False),
        ('bio_only', False, True),
        ('fusion', True, True)
    ]
    
    for model_name in model_names:
        for config_name, use_resnet, use_bio in feature_configs:
            key = f"{model_name}_{config_name}"
            print(f"Training {key}...")
            
            try:
                _, metrics = train_sklearn_baseline(
                    model_name=model_name,
                    train_data=train_data,
                    test_data=test_data,
                    include_resnet=use_resnet,
                    include_bio=use_bio,
                    random_state=random_state
                )
                results[key] = metrics
                print(f"  Test AUC: {metrics['test_auc']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                results[key] = {'error': str(e)}
    
    return results
