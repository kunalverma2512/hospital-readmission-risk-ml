"""Feature engineering pipeline for hospital readmission risk prediction.

This module constructs clinical risk indicators including ICD-9 diagnosis clustering,
physician specialty binning, clinical ratios, log-scale normalization, interaction terms,
and one-hot encoded categorical vectors.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_COLS,
    INTERACTION_TERMS,
    NUMERIC_COLS,
    TARGET_COL,
)


def group_icd9(code: str) -> str:
    """Classify raw ICD-9 billing codes into high-level diagnostic disease clusters.

    Clusters:
    - Diabetes: 250.xx
    - Circulatory: 390-459, 410 (Acute MI), 428 (Heart Failure)
    - Respiratory: 460-519, 786 (Symptoms involving respiratory system)
    - Missing: 'Missing' placeholder
    - Other: All other non-target diagnostic codes

    Args:
        code: String representation of ICD-9 diagnosis code.

    Returns:
        str: Clinical category name.
    """
    if pd.isna(code):
        return "Missing"
    code_str = str(code).strip()
    if code_str.startswith("250"):
        return "Diabetes"
    elif code_str.startswith(("390", "410", "428")):
        return "Circulatory"
    elif code_str.startswith(("460", "786")):
        return "Respiratory"
    elif code_str == "Missing" or code_str == "?":
        return "Missing"
    return "Other"


def bin_specialty(specialty: str) -> str:
    """Bin primary admitting specialties to reduce sparsity and high cardinality.

    Args:
        specialty: String name of medical specialty.

    Returns:
        str: Binned medical specialty label.
    """
    if pd.isna(specialty):
        return "Other"
    top_specialties = {"InternalMedicine", "Cardiology", "Family/GeneralPractice"}
    return specialty if specialty in top_specialties else "Other"


def engineer_features(
    df_agg: pd.DataFrame, active_med_cols: List[str]
) -> pd.DataFrame:
    """Apply complete feature engineering transformations to aggregated patient data.

    Transformations:
    1. Medication change velocity (numchange) and composite service utilization.
    2. ICD-9 disease cluster categorization across primary, secondary, and tertiary diagnoses.
    3. Medical specialty consolidation.
    4. Logarithmic transformation of heavy-tailed clinical count features.
    5. Clinical ratio metrics: medications per diagnosis, hospital days per age tier.
    6. Non-linear interaction terms capturing medication load across length of stay.
    7. Standardized ordinal mappings for lab tests (A1C, Glucose) and demographics.
    8. Dummy encoding of categorical attributes.

    Args:
        df_agg: Aggregated patient-level DataFrame.
        active_med_cols: Retained active medication column names.

    Returns:
        pd.DataFrame: Engineered feature DataFrame ready for selection and scaling.
    """
    df = df_agg.copy()
    print("Executing feature engineering transformations...")

    # 1. Total active medications and healthcare utilization index
    valid_meds = [m for m in active_med_cols if m in df.columns]
    df["numchange"] = df[valid_meds].gt(0).sum(axis=1)
    df["service_utilization"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    # 2. ICD-9 Diagnostic Clustering
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col in df.columns:
            df[f"{diag_col}_group"] = df[diag_col].apply(group_icd9)
        else:
            df[f"{diag_col}_group"] = "Other"

    # 3. Medical Specialty Binning
    if "medical_specialty" in df.columns:
        df["medical_specialty_group"] = df["medical_specialty"].apply(bin_specialty)
    else:
        df["medical_specialty_group"] = "Other"

    # 4. Log Transforms on Heavy-Tailed Features
    log_targets = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_medications",
        "service_utilization",
        "number_inpatient",
    ]
    for col in log_targets:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    # 5. Clinical Ratio Metrics
    safe_diag_count = df["number_diagnoses"].replace(0, 1)
    safe_age = df["age"].replace(0, 1)
    df["meds_per_diag"] = df["num_medications"] / safe_diag_count
    df["hospital_per_age"] = df["time_in_hospital"] / safe_age

    # 6. Interaction Terms
    for f1, f2 in INTERACTION_TERMS:
        if f1 in df.columns and f2 in df.columns:
            df[f"{f1}|{f2}"] = df[f1] * df[f2]

    # 7. Categorical Ordinal Encodings
    df["gender"] = df["gender"].replace({"Male": 1, "Female": 0}).fillna(0).astype(int)
    df["diabetesMed"] = df["diabetesMed"].replace({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Ordinal encoding for glycemic serum testing
    df["max_glu_serum"] = (
        df["max_glu_serum"]
        .replace({">200": 1, ">300": 1, "Norm": 0, "Not Tested": -1})
        .fillna(-1)
        .astype(int)
    )

    # Ordinal encoding for HbA1c control status
    df["A1Cresult"] = (
        df["A1Cresult"]
        .replace({">7": 1, ">8": 1, "Norm": 0, "Not Tested": -1})
        .fillna(-1)
        .astype(int)
    )

    # 8. Dummy Encoding for Multiclass Categoricals
    cat_cols_to_encode = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols_to_encode, drop_first=True)

    print(f"Feature engineering complete. Dataset transformed to {df.shape[1]} total columns.")
    return df


def build_feature_matrix(
    df: pd.DataFrame, active_med_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Assemble candidate feature matrix X and ground-truth target vector y.

    Args:
        df: Feature-engineered DataFrame.
        active_med_cols: Retained medication column names.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target series y.
    """
    dummy_prefixes = (
        "medical_specialty_group_",
        "max_glu_serum_",
        "A1Cresult_",
        "diag_1_group_",
        "diag_2_group_",
        "diag_3_group_",
    )
    dummy_cols = [c for c in df.columns if c.startswith(dummy_prefixes)]
    demographic_cols = ["gender", "change", "diabetesMed"]
    valid_meds = [m for m in active_med_cols if m in df.columns]

    feature_cols = [c for c in NUMERIC_COLS if c in df.columns] + dummy_cols + valid_meds + demographic_cols
    # Deduplicate while preserving order
    feature_cols = list(dict.fromkeys(feature_cols))

    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)

    print(f"Candidate feature matrix assembled: {X.shape[1]} features across {X.shape[0]:,} patient samples.")
    return X, y
