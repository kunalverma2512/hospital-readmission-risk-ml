"""Model definition, training, and artifact serialization module.

Provides training routines for:
1. Decision Tree baseline (interpretable clinical decision tree).
2. XGBoost (histogram gradient boosted trees with early stopping).
3. LightGBM (leaf-wise gradient boosting with imbalance weighting).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from src.config import MODEL_SAVE_PATH, RANDOM_SEED


def build_decision_tree(
    max_depth: int = 5,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    random_state: int = RANDOM_SEED,
) -> DecisionTreeClassifier:
    """Instantiate an interpretable Decision Tree baseline model.

    Args:
        max_depth: Maximum tree depth to constrain overfitting.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples in leaf nodes.
        random_state: Random state seed.

    Returns:
        DecisionTreeClassifier: Configured baseline classifier.
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        criterion="gini",
        random_state=random_state,
    )


def build_xgboost(
    max_depth: int = 5,
    learning_rate: float = 0.05,
    n_estimators: int = 200,
    subsample: float = 0.8,
    min_child_weight: int = 1,
    scale_pos_weight: float = 1.0,
    random_state: int = RANDOM_SEED,
) -> Any:
    """Instantiate an XGBoost classifier configured for CPU execution.

    Args:
        max_depth: Maximum depth of a tree.
        learning_rate: Step size shrinkage.
        n_estimators: Number of boosting rounds.
        subsample: Subsample ratio of training instances.
        min_child_weight: Minimum sum of instance weight in child.
        scale_pos_weight: Balancing of positive and negative weights.
        random_state: Random state seed.

    Returns:
        XGBClassifier: Configured XGBoost model.
    """
    if not HAS_XGB:
        raise ImportError("xgboost is not installed. Please run `pip install xgboost`.")

    return XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def build_lightgbm(
    max_depth: int = 5,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    n_estimators: int = 200,
    random_state: int = RANDOM_SEED,
) -> Any:
    """Instantiate a LightGBM classifier with imbalance handling.

    Args:
        max_depth: Maximum tree depth.
        num_leaves: Maximum tree leaves for base learners.
        learning_rate: Boosting learning rate.
        n_estimators: Number of boosting trees.
        random_state: Random state seed.

    Returns:
        LGBMClassifier: Configured LightGBM model.
    """
    if not HAS_LGBM:
        raise ImportError("lightgbm is not installed. Please run `pip install lightgbm`.")

    return LGBMClassifier(
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        is_unbalance=True,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Any:
    """Fit model on resampled training data with optional validation monitoring.

    Args:
        model: Estimator instance (DecisionTree, XGBoost, or LightGBM).
        X_train: Training feature array.
        y_train: Training labels array.
        X_val: Optional validation features for early stopping.
        y_val: Optional validation labels.

    Returns:
        Fitted model instance.
    """
    model_name = model.__class__.__name__
    print(f"Fitting {model_name} on {X_train.shape[0]:,} samples...")

    if HAS_XGB and isinstance(model, XGBClassifier) and X_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    print(f"Successfully fitted {model_name}.")
    return model


def save_pipeline(
    model: Any,
    scaler: Any,
    feature_names: List[str],
    optimal_threshold: float,
    output_path: Path = MODEL_SAVE_PATH,
) -> None:
    """Serialize the trained model, standard scaler, and metadata bundle.

    Args:
        model: Trained classifier.
        scaler: Fitted StandardScaler.
        feature_names: List of expected input features in exact column order.
        optimal_threshold: Clinically tuned decision threshold.
        output_path: Target path for .pkl file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "optimal_threshold": optimal_threshold,
    }

    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Serialized model pipeline bundle saved to: {output_path.resolve()}")


def load_pipeline(model_path: Path = MODEL_SAVE_PATH) -> Dict[str, Any]:
    """Load serialized model bundle from disk.

    Args:
        model_path: Path to .pkl bundle.

    Returns:
        Dict containing model, scaler, feature_names, and optimal_threshold.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path.resolve()}")

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    # Handle legacy flat model pickles gracefully
    if not isinstance(bundle, dict):
        bundle = {
            "model": bundle,
            "scaler": None,
            "feature_names": None,
            "optimal_threshold": 0.5,
        }

    return bundle
