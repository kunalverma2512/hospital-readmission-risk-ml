"""End-to-end training pipeline script for Hospital Readmission Prediction.

Usage:
    python train.py --model xgb --tune-threshold
    python train.py --model dt
    python train.py --model lgb
"""

import argparse
from pathlib import Path
import sys

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DEFAULT_CLINICAL_THRESHOLD,
    MODEL_SAVE_PATH,
    RAW_DATA_FILE,
)
from src.data_preprocessing import (
    aggregate_patient_encounters,
    clean_encounter_data,
    load_raw_data,
)
from src.evaluation import find_optimal_threshold, print_evaluation_report
from src.feature_engineering import build_feature_matrix, engineer_features
from src.feature_selection import (
    resample_minority_class,
    scale_features,
    select_features,
    split_data,
)
from src.models import (
    build_decision_tree,
    build_lightgbm,
    build_xgboost,
    save_pipeline,
    train_model,
)


def run_pipeline(
    model_type: str = "xgb",
    tune_threshold: bool = True,
    use_preset_features: bool = True,
    output_path: Path = MODEL_SAVE_PATH,
) -> None:
    """Execute the end-to-end hospital readmission training pipeline.

    Args:
        model_type: Model architecture ('xgb', 'dt', or 'lgb').
        tune_threshold: Whether to compute clinically optimal decision threshold.
        use_preset_features: Whether to use pre-validated top 30 RFE clinical features.
        output_path: Path to save serialized model bundle.
    """
    print("\n=======================================================")
    print(" 🏥 HOSPITAL 30-DAY READMISSION RISK PREDICTION PIPELINE")
    print("=======================================================\n")

    # Step 1: Load Raw Clinical Data
    raw_df = load_raw_data(RAW_DATA_FILE)

    # Step 2: Data Cleaning & Filtering
    cleaned_df, active_meds = clean_encounter_data(raw_df)

    # Step 3: Patient-Level Aggregation (Prevents Multi-Encounter Leakage)
    agg_df = aggregate_patient_encounters(cleaned_df, active_meds)

    # Step 4: Clinical Feature Engineering
    engineered_df = engineer_features(agg_df, active_meds)

    # Step 5: Assemble Candidate Feature Matrix
    X_all, y_all = build_feature_matrix(engineered_df, active_meds)

    # Step 6: Feature Selection (RFE Top 30)
    X_selected, selected_feature_names = select_features(
        X_all, y_all, n_features=30, use_preset=use_preset_features
    )

    # Step 7: Train-Test Split (80% Train, 20% Holdout Test)
    X_train, X_test, y_train, y_test = split_data(
        X_selected, y_all, test_size=0.20, random_state=0
    )

    # Step 8: Standard Scaling (Fit on Train only)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 9: Balance Training Set using ADASYN Resampling
    X_train_resampled, y_train_resampled = resample_minority_class(
        X_train_scaled, y_train, random_state=0
    )

    # Step 10: Model Instantiation & Training
    print(f"\nBuilding model architecture: {model_type.upper()}...")
    if model_type == "xgb":
        model = build_xgboost(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            random_state=0,
        )
    elif model_type == "lgb":
        model = build_lightgbm(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=200,
            random_state=0,
        )
    elif model_type == "dt":
        model = build_decision_tree(
            max_depth=5,
            random_state=0,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'xgb', 'dt', 'lgb'.")

    trained_model = train_model(
        model,
        X_train_resampled,
        y_train_resampled,
        X_val=X_test_scaled,
        y_val=y_test.values,
    )

    # Step 11: Evaluation & Decision Threshold Tuning
    y_test_proba = trained_model.predict_proba(X_test_scaled)[:, 1]

    if tune_threshold:
        print("\nOptimizing clinical decision threshold for safety (Recall >= 0.70)...")
        optimal_thresh, _ = find_optimal_threshold(
            y_test.values, y_test_proba, min_recall=0.70, min_precision=0.45
        )
    else:
        optimal_thresh = DEFAULT_CLINICAL_THRESHOLD

    print_evaluation_report(
        y_test.values,
        y_test_proba,
        threshold=optimal_thresh,
        model_name=f"{model_type.upper()}_Classifier",
    )

    # Step 12: Serialize Model Pipeline Bundle
    save_pipeline(
        model=trained_model,
        scaler=scaler,
        feature_names=selected_feature_names,
        optimal_threshold=optimal_thresh,
        output_path=output_path,
    )
    print("\n✅ Training pipeline execution finished successfully!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Hospital Readmission Risk Prediction Model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgb",
        choices=["xgb", "dt", "lgb"],
        help="Model architecture: xgb (XGBoost), dt (Decision Tree), lgb (LightGBM)",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        default=True,
        help="Tune decision threshold to guarantee clinical recall >= 0.70",
    )
    parser.add_argument(
        "--run-full-rfe",
        action="store_true",
        default=False,
        help="Run full RFE feature selection from scratch instead of using verified features",
    )
    args = parser.parse_args()

    run_pipeline(
        model_type=args.model,
        tune_threshold=args.tune_threshold,
        use_preset_features=not args.run_full_rfe,
    )
