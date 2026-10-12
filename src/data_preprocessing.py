"""Data loading, cleaning, and patient-level aggregation module.

This module processes raw hospital encounter data from the UCI Diabetes dataset,
handles missing clinical values, removes deceased cases, and aggregates multiple
hospital visits per patient to prevent cross-encounter data leakage.
"""

from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from src.config import (
    ALL_MEDICATION_COLS,
    DROP_RAW_COLUMNS,
    EXCLUDED_DISCHARGE_IDS,
    RAW_DATA_FILE,
)


def load_raw_data(file_path: Union[str, Path] = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw diabetic encounters dataset from CSV.

    Args:
        file_path: Path to the diabetic_data.csv file.

    Returns:
        pd.DataFrame: Raw encounter records.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path.resolve()}")
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]:,} encounters across {df.shape[1]} columns.")
    return df


def clean_encounter_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Clean clinical encounter fields, map codes, and drop invalid rows.

    Performs the following cleaning steps:
    1. Encodes ordinal age intervals '[10-20)' -> numeric 1..10.
    2. Binarizes medication utilization (0 for 'No', 1 for 'Steady'/'Up'/'Down').
    3. Imputes missing markers ('?' in race, specialty, and ICD-9 diagnosis codes).
    4. Drops invalid gender entries and uninformative/sparse columns (e.g. weight).
    5. Filters out expired/deceased patients (discharge_disposition_id == 11).
    6. Encodes medication change and binary 30-day readmission target.
    7. Applies VarianceThreshold to retain active medication columns.

    Args:
        df: Raw encounters DataFrame.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Cleaned DataFrame and retained medication column names.
    """
    df = df.copy()

    # 1. Encode age intervals as ordinal numeric (1 to 10)
    for i in range(10):
        interval_str = f"[{10 * i}-{10 * (i + 1)})"
        df["age"] = df["age"].replace(interval_str, i + 1)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # 2. Encode medication dosage changes to binary usage (0: No, 1: Prescribed/Adjusted)
    medication_cols = [c for c in ALL_MEDICATION_COLS if c in df.columns]
    for col in medication_cols:
        df[col] = df[col].map({"No": 0, "Steady": 1, "Up": 1, "Down": 1}).fillna(0).astype(int)

    # 3. Handle missing values marked by '?'
    df["race"] = df["race"].replace("?", "Unknown")
    df["medical_specialty"] = df["medical_specialty"].replace("?", "Unknown")
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col in df.columns:
            df[diag_col] = df[diag_col].replace("?", "Missing")

    # 4. Remove ambiguous gender records
    initial_count = len(df)
    df = df[df["gender"] != "Unknown/Invalid"]

    # 5. Drop sparse and zero-variance columns
    cols_to_drop = [c for c in DROP_RAW_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Impute missing lab test indications
    df["max_glu_serum"] = df["max_glu_serum"].fillna("Not Tested")
    df["A1Cresult"] = df["A1Cresult"].fillna("Not Tested")

    # 6. Exclude deceased patients where 30-day readmission is not possible
    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)]
    print(f"Filtered out deceased/invalid encounters: {initial_count - len(df):,} rows removed.")

    # 7. Encode target and medication change flags
    df["change"] = df["change"].replace({"No": 0, "Ch": 1}).fillna(0).astype(int)
    # Binary classification: 1 for readmitted within 30 days (<30), 0 otherwise (>30 or NO)
    df["readmitted"] = df["readmitted"].replace({"<30": 1, ">30": 0, "NO": 0}).astype(int)

    # 8. Drop low-variance medication columns (threshold = 0.01)
    selector = VarianceThreshold(threshold=0.01)
    med_features = df[medication_cols]
    selector.fit(med_features)
    active_med_cols = [
        medication_cols[i] for i, supported in enumerate(selector.get_support()) if supported
    ]
    print(f"Retained {len(active_med_cols)} active medication columns with variance >= 0.01.")

    return df, active_med_cols


def aggregate_patient_encounters(
    df: pd.DataFrame, active_med_cols: List[str]
) -> pd.DataFrame:
    """Aggregate multi-encounter records per patient to prevent data leakage.

    Because a patient may have multiple hospital admissions, evaluating encounters
    independently leaks patient-specific baselines into test sets. This aggregation
    synthesizes a single patient-level profile summarizing their clinical trajectory.

    Args:
        df: Cleaned encounter-level DataFrame.
        active_med_cols: List of retained medication columns.

    Returns:
        pd.DataFrame: Aggregated patient-level DataFrame.
    """
    print("Aggregating hospital encounters by patient_nbr...")

    agg_rules = {
        "time_in_hospital": "mean",
        "num_lab_procedures": "mean",
        "num_procedures": "mean",
        "num_medications": "mean",
        "number_outpatient": "sum",
        "number_emergency": "sum",
        "number_inpatient": "sum",
        "number_diagnoses": "mean",
        "readmitted": "max",          # 1 if readmitted <30 days in ANY encounter
        "gender": "first",
        "age": "first",
        "max_glu_serum": lambda x: x.mode()[0] if not x.mode().empty else "Not Tested",
        "A1Cresult": lambda x: x.mode()[0] if not x.mode().empty else "Not Tested",
        "change": "sum",               # Total count of medication adjustments across visits
        "diabetesMed": "max",          # Ever prescribed diabetes medication
        "medical_specialty": "first",
        "diag_1": "first",
        "diag_2": "first",
        "diag_3": "first",
    }

    # Add aggregation for each active medication
    for med in active_med_cols:
        agg_rules[med] = "max"

    df_agg = df.groupby("patient_nbr").agg(agg_rules).reset_index()

    # Calculate total encounter frequency per patient
    encounter_counts = (
        df.groupby("patient_nbr")
        .size()
        .reset_index(name="encounter_count")
    )
    df_agg = df_agg.merge(encounter_counts, on="patient_nbr", how="left")

    print(
        f"Aggregated {len(df):,} encounters into {len(df_agg):,} unique patient profiles. "
        f"Readmission rate: {df_agg['readmitted'].mean():.2%}"
    )

    return df_agg
