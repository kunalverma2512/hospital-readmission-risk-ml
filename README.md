# 30-Day Hospital Readmission Risk Prediction with Clinical Calibration

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-red)](https://xgboost.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Diabetes%20(130--Hospitals)-blue)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

A machine learning pipeline for predicting 30-day hospital readmission risk among diabetic patients using 10 years (1999–2008) of electronic health records from 130 US hospitals (UCI Machine Learning Repository).

Unlike standard academic pipelines that evaluate hospital visits as isolated independent rows, this project focuses on **three clinical reality requirements**:
1. **Patient-Level Longitudinal Aggregation:** Prevents patient-level data leakage between train and test splits by grouping 101,766 visits into 70,443 unique patient records.
2. **Clinical Feature Engineering & RFE:** ICD-9 diagnostic clustering, polypharmacy burden, and Recursive Feature Elimination (top 30 predictors).
3. **Cost-Sensitive Decision Threshold Tuning:** Optimizing decision cutoffs to guarantee **$\ge 70\%$ clinical recall** for high-risk patients under severe class imbalance (~12% positive class).

---

## 📊 Benchmark Evaluation Results

Evaluated on an independent 20% holdout test partition of **14,089 unseen patients**:

| Metric | Decision Tree Baseline | Final Tuned XGBoost Pipeline |
| :--- | :---: | :---: |
| **ROC-AUC** | 0.6841 | **0.8591** |
| **PR-AUC** | 0.3120 | **0.5516** |
| **Clinical Recall (Sensitivity)** | 58.00% | **70.04%** |
| **Clinical Specificity** | 74.20% | **87.83%** |
| **Precision (PPV)** | 34.00% | **45.08%** |
| **F1-Score** | 0.4280 | **0.5485** |

### Confusion Matrix on Holdout Test Set (Cutoff $\tau = 0.315$)

```text
                           Predicted Negative (<30d)    Predicted Positive (<30d)
Actual Negative (0)                 10,829                        1,501 (87.8% Specificity)
Actual Positive (1)                    527                        1,232 (70.0% Recall)
```

> **Why Threshold Tuning Matters in Healthcare:**  
> In hospital readmission prevention, a **False Negative** (discharging a patient who deteriorates and gets readmitted without post-care intervention) is far more dangerous than a **False Positive** (scheduling an extra follow-up nurse call). Standard ML models with a naive `0.50` threshold only catch ~30–40% of readmissions. Calibrating the decision threshold to **`0.315`** ensures that over **70% of high-risk patients are flagged** while maintaining a high 87.8% specificity.

---

## 🔬 Core Modeling & Feature Engineering Details

### 1. Patient-Level Grouping (Zero Data Leakage)
The raw dataset contains **101,766 encounters** across **70,443 unique patients**. Evaluating encounters as independent rows is a common pitfall because the same patient's baseline information leaks across splits. Our pipeline:
- Filters out 1,645 encounters where patients were expired/deceased (`discharge_disposition_id == 11`).
- Groups longitudinal trajectories by `patient_nbr` (cumulative hospital encounters, medication changes, and lab volumes).
- Yields **70,443 unique patient profiles** with a realistic **12.54% 30-day readmission rate**.

### 2. Clinical Feature Engineering
- **ICD-9 Disease Clustering:** Primary, secondary, and tertiary diagnosis codes (`diag_1`, `diag_2`, `diag_3`) are clustered into high-impact clinical categories: *Diabetes* (250.xx), *Circulatory* (390–459, 410, 428), *Respiratory* (460–519, 786), and *Other*.
- **Polypharmacy & Length of Stay Ratios:** $\text{meds\_per\_diag} = \frac{\text{num\_medications}}{\text{number\_diagnoses}}$ and $\text{hospital\_per\_age} = \frac{\text{time\_in\_hospital}}{\text{age\_tier}}$.
- **Composite Service Utilization Index:** Sum of outpatient, emergency, and previous inpatient admissions, stabilized with $\log(1 + x)$.

### 3. Feature Selection & Class Imbalance
- **Recursive Feature Elimination (RFE):** Selects the top 30 most informative clinical variables from the candidate pool.
- **ADASYN Oversampling:** Rebalances the training distribution along challenging decision boundary regions.

### 4. Top 5 Clinical Predictors
Feature importances extracted from the final gradient boosted pipeline:
1. **Prior Inpatient Admissions (`number_inpatient_log`):** Strongest indicator of recurring hospital dependence.
2. **Total Service Utilization (`service_utilization_log`):** Emergency and outpatient history flags patient frailty.
3. **Inpatient Length of Stay (`time_in_hospital_log`):** Extended stay correlates with acute severity.
4. **Medication Adjustments (`change` & `numchange`):** Dosage changes during stay flag unstable glycemic control.
5. **Polypharmacy Load (`num_medications_log`):** High medication count reflects multi-morbidity.

---

## 🚀 Quickstart & Usage

The repository includes both the pre-processed dataset and the serialized model bundle (`models/hospital_readmission_model.pkl`), so evaluation runs instantly out of the box.

### 1. Installation
Python 3.10 or 3.11 is recommended.
```bash
# Clone the repository
git clone https://github.com/kunalverma2512/hospital-readmission-risk-ml.git
cd hospital-readmission-risk-ml

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Holdout Test Evaluation (Takes ~5 seconds)
Evaluate the persisted pipeline on the 14,089 holdout patient test set:
```bash
python evaluate.py
```
To test alternative decision thresholds:
```bash
python evaluate.py --threshold 0.50
```

### 3. Retrain Pipeline from Scratch (Optional, ~2 minutes)
```bash
python train.py --model xgb --tune-threshold
```

---

## 📜 License
This project is licensed under the MIT License.
