"""Standalone model evaluation and inference script for Hospital Readmission Prediction.

Usage:
    python evaluate.py
    python evaluate.py --model-path models/hospital_readmission_model.pkl --threshold 0.678
"""

import argparse
from pathlib import Path
import sys

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MODEL_SAVE_PATH, RAW_DATA_FILE
from src.data_preprocessing import (
    aggregate_patient_encounters,
    clean_encounter_data,
    load_raw_data,
)
from src.evaluation import print_evaluation_report
from src.feature_engineering import build_feature_matrix, engineer_features
from src.feature_selection import split_data
from src.models import load_pipeline


def run_evaluation(
    model_path: Path = MODEL_SAVE_PATH,
    threshold_override: float = None,
) -> None:
    """Evaluate trained model pipeline against holdout patient test data.

    Args:
        model_path: Path to serialized model pipeline pickle.
        threshold_override: Optional custom threshold to override model's optimal threshold.
    """
    print(f"Loading trained model pipeline from: {Path(model_path).resolve()}")
    bundle = load_pipeline(model_path)

    model = bundle["model"]
    scaler = bundle.get("scaler")
    feature_names = bundle.get("feature_names")
    stored_threshold = bundle.get("optimal_threshold", 0.5)

    threshold = threshold_override if threshold_override is not None else stored_threshold
    print(f"Using operating decision threshold: {threshold:.3f}")

    # Prepare holdout evaluation data
    print("\nPreparing holdout evaluation dataset...")
    raw_df = load_raw_data(RAW_DATA_FILE)
    cleaned_df, active_meds = clean_encounter_data(raw_df)
    agg_df = aggregate_patient_encounters(cleaned_df, active_meds)
    engineered_df = engineer_features(agg_df, active_meds)
    X_all, y_all = build_feature_matrix(engineered_df, active_meds)

    # Use exact features expected by model
    if feature_names is not None:
        X_eval = X_all[feature_names]
    else:
        X_eval = X_all

    _, X_test, _, y_test = split_data(X_eval, y_all, test_size=0.20, random_state=0)

    # Apply saved scaler
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test.values

    # Predict probabilities
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Generate and print full clinical report
    print_evaluation_report(
        y_test.values,
        y_test_proba,
        threshold=threshold,
        model_name="Evaluated_Model",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Hospital Readmission Risk Prediction Model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_SAVE_PATH),
        help="Path to serialized model bundle .pkl file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold override (e.g. 0.678)",
    )
    args = parser.parse_args()

    run_evaluation(
        model_path=Path(args.model_path),
        threshold_override=args.threshold,
    )
