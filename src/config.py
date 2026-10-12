"""Configuration settings and constants for Hospital Readmission Risk Prediction.

This module provides path management, column definitions, clinical mapping rules,
and modeling hyperparameters for reproducible pipeline execution.
"""

from pathlib import Path
from typing import List, Tuple

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# File Paths
RAW_DATA_FILE = DATA_RAW_DIR / "diabetic_data.csv"
MAPPING_DATA_FILE = DATA_RAW_DIR / "IDS_mapping.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "patient_aggregated_data.parquet"
MODEL_SAVE_PATH = MODELS_DIR / "hospital_readmission_model.pkl"
SCALER_SAVE_PATH = MODELS_DIR / "standard_scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "selected_features.json"

# Pipeline & Modeling Constants
RANDOM_SEED = 0
TEST_SIZE = 0.20
CV_FOLDS = 5
N_FEATURES_RFE = 30
DEFAULT_CLINICAL_THRESHOLD = 0.678
TARGET_COL = "readmitted"

# Columns with excessive missingness (>40%) or constant values
DROP_RAW_COLUMNS: List[str] = [
    "weight",        # ~97% missing
    "payer_code",    # ~40% missing
    "citoglipton",   # Zero variance
    "examide",       # Zero variance
]

# Discharge disposition IDs where readmission is clinically not applicable
# (ID 11 = Expired/deceased, 19/20 = Expired at home/medical facility)
EXCLUDED_DISCHARGE_IDS: List[int] = [11]

# 21 Diabetes medications tracked across encounters
ALL_MEDICATION_COLS: List[str] = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "insulin", "glyburide-metformin", "tolazamide",
    "metformin-pioglitazone", "metformin-rosiglitazone", "glimepiride-pioglitazone",
    "glipizide-metformin", "troglitazone", "tolbutamide", "acetohexamide"
]

# Numeric columns used for modeling after aggregation and log transforms
NUMERIC_COLS: List[str] = [
    "age",
    "time_in_hospital_log",
    "num_lab_procedures_log",
    "num_procedures",
    "num_medications_log",
    "number_outpatient",
    "number_emergency",
    "number_inpatient_log",
    "number_diagnoses",
    "service_utilization_log",
    "numchange",
    "encounter_count",
    "meds_per_diag",
    "hospital_per_age",
    "num_medications|time_in_hospital",
    "num_medications|number_diagnoses",
]

# Feature pairs for clinical interaction modeling
INTERACTION_TERMS: List[Tuple[str, str]] = [
    ("num_medications", "time_in_hospital"),
    ("num_medications", "number_diagnoses"),
]

# Categorical columns to be one-hot encoded
CATEGORICAL_COLS: List[str] = [
    "medical_specialty_group",
    "max_glu_serum",
    "A1Cresult",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
]

# Exact 30 features selected by Recursive Feature Elimination (RFE)
SELECTED_30_FEATURES: List[str] = [
    "age",
    "time_in_hospital_log",
    "num_lab_procedures_log",
    "num_procedures",
    "num_medications_log",
    "number_outpatient",
    "number_emergency",
    "number_inpatient_log",
    "number_diagnoses",
    "service_utilization_log",
    "numchange",
    "encounter_count",
    "meds_per_diag",
    "hospital_per_age",
    "num_medications|time_in_hospital",
    "num_medications|number_diagnoses",
    "gender",
    "change",
    "diabetesMed",
    "metformin",
    "repaglinide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "insulin",
    "medical_specialty_group_InternalMedicine",
    "medical_specialty_group_Other",
    "max_glu_serum_0",
]
