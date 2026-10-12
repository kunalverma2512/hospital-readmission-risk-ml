"""Feature selection, scaling, and class-imbalance resampling module.

This module provides train-test splitting, standard normalization, Recursive Feature
Elimination (RFE) using Random Forest, and adaptive synthetic oversampling (ADASYN)
to balance the severely skewed 30-day readmission minority class.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import ADASYN
    HAS_ADASYN = True
except ImportError:
    HAS_ADASYN = False

from src.config import (
    N_FEATURES_RFE,
    RANDOM_SEED,
    SELECTED_30_FEATURES,
    TEST_SIZE,
)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partition patient samples into independent training and holdout evaluation sets.

    Args:
        X: Feature matrix.
        y: Ground truth binary target series.
        test_size: Fraction of samples reserved for testing (default 0.20).
        random_state: Seed for reproducibility.
        stratify: If True, preserves class balance across splits.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_target
    )
    print(
        f"Data split: {X_train.shape[0]:,} train samples, {X_test.shape[0]:,} holdout test samples."
    )
    return X_train, X_test, y_train, y_test


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = N_FEATURES_RFE,
    use_preset: bool = True,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, List[str]]:
    """Select the most predictive clinical features using RFE or pre-validated feature set.

    Args:
        X: Candidate feature matrix.
        y: Target label series.
        n_features: Target count of features to select (default 30).
        use_preset: If True, uses the validated RFE-selected top-30 features for efficiency.
        random_state: Random state for Random Forest estimator.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Reduced feature DataFrame and list of feature names.
    """
    if use_preset:
        available_cols = [c for c in SELECTED_30_FEATURES if c in X.columns]
        if len(available_cols) >= n_features:
            selected_cols = available_cols[:n_features]
            print(f"Using validated top-{len(selected_cols)} clinical features.")
            return X[selected_cols], selected_cols

    print(f"Executing Recursive Feature Elimination (RFE) for top {n_features} features...")
    rf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=n_features, step=2)
    rfe.fit(X, y)

    selected_cols = X.columns[rfe.support_].tolist()
    print(f"RFE successfully identified top {len(selected_cols)} features.")
    return X[selected_cols], selected_cols


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a standard normal scaler on train features and apply it to both splits.

    Fitting the scaler strictly on the training partition prevents data leakage
    into the holdout evaluation set.

    Args:
        X_train: Training features.
        X_test: Testing features.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled using zero-mean unit-variance StandardScaler.")
    return X_train_scaled, X_test_scaled, scaler


def resample_minority_class(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply ADASYN oversampling to balance the minority 30-day readmission class.

    Adaptive Synthetic (ADASYN) focuses oversampling on harder-to-learn minority
    samples located near decision boundaries rather than generating uniformly
    interpolated points like standard SMOTE.

    Args:
        X_train_scaled: Scaled feature matrix of training set.
        y_train: Imbalanced target vector.
        random_state: Random state seed.

    Returns:
        Tuple of (X_train_resampled, y_train_resampled).
    """
    initial_positives = int((y_train == 1).sum())
    initial_negatives = int((y_train == 0).sum())
    print(
        f"Pre-resampling class distribution: {initial_negatives:,} negative, {initial_positives:,} positive."
    )

    if HAS_ADASYN:
        print("Applying ADASYN (Adaptive Synthetic Sampling)...")
        adasyn = ADASYN(random_state=random_state)
        X_resampled, y_resampled = adasyn.fit_resample(X_train_scaled, y_train)
    else:
        print("imbalanced-learn not installed; falling back to random oversampling.")
        pos_indices = np.where(y_train == 1)[0]
        neg_indices = np.where(y_train == 0)[0]
        oversampled_pos = np.random.RandomState(random_state).choice(
            pos_indices, size=len(neg_indices), replace=True
        )
        combined_idx = np.concatenate([neg_indices, oversampled_pos])
        np.random.RandomState(random_state).shuffle(combined_idx)
        X_resampled = X_train_scaled[combined_idx]
        y_resampled = y_train.iloc[combined_idx].values

    final_positives = int((y_resampled == 1).sum())
    final_negatives = int((y_resampled == 0).sum())
    print(
        f"Post-resampling class distribution: {final_negatives:,} negative, {final_positives:,} positive (Total: {len(y_resampled):,})."
    )

    return X_resampled, y_resampled
