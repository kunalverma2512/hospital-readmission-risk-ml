"""Evaluation, clinical metric reporting, and decision threshold tuning module.

This module provides tools for assessing clinical utility:
1. Precision-Recall Curve analysis and PR-AUC.
2. Receiver Operating Characteristic (ROC-AUC).
3. Clinical threshold optimization enforcing safety constraints (Recall >= 0.70).
4. Detailed confusion matrix and classification report generation.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json
import numpy as np
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.config import REPORTS_DIR


def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics at a given decision threshold.

    Args:
        y_true: Ground truth binary target (0 or 1).
        y_proba: Predicted probability of positive class (readmission within 30 days).
        threshold: Decision threshold for binarizing probabilities into classes.

    Returns:
        Dict[str, Any]: Dictionary containing confusion matrix, AUC scores, and class metrics.
    """
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(auc(recall, precision))

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * (prec * sens) / (prec + sens + 1e-10))

    metrics = {
        "threshold": float(threshold),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "sensitivity_recall": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = 0.70,
    min_precision: float = 0.50,
) -> Tuple[float, Dict[str, Any]]:
    """Determine the clinically optimal classification threshold.

    In hospital readmission prevention, False Negatives (missing a patient who
    subsequently gets readmitted) incur high clinical risk. Therefore, this function
    searches for a threshold that guarantees a minimum recall (e.g. >= 0.70) while
    maximizing precision and F1 score.

    Args:
        y_true: True binary target array.
        y_proba: Predicted readmission probabilities.
        min_recall: Enforced clinical recall minimum (default 0.70).
        min_precision: Target precision bound (default 0.50).

    Returns:
        Tuple[float, Dict[str, Any]]: Optimal threshold and its associated metrics.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # Align lengths: precision and recall have 1 extra element (thresholds is len - 1)
    prec_pts = precision[:-1]
    rec_pts = recall[:-1]
    f1_pts = 2 * (prec_pts * rec_pts) / (prec_pts + rec_pts + 1e-10)

    # Strategy 1: Find candidates satisfying BOTH precision >= min_precision AND recall >= min_recall
    ideal_candidates = []
    for i, t in enumerate(thresholds):
        if prec_pts[i] >= min_precision and rec_pts[i] >= min_recall:
            ideal_candidates.append((t, prec_pts[i], rec_pts[i], f1_pts[i]))

    if ideal_candidates:
        # Choose highest F1 score among ideal candidates
        best_t, best_p, best_r, best_f1 = max(ideal_candidates, key=lambda x: x[3])
        print(f"Optimal threshold found meeting both constraints (Recall >= {min_recall}, Precision >= {min_precision}): {best_t:.3f}")
    else:
        # Strategy 2: Fallback to maximizing precision subject to recall >= min_recall
        recall_candidates = [
            (t, prec_pts[i], rec_pts[i], f1_pts[i])
            for i, t in enumerate(thresholds)
            if rec_pts[i] >= min_recall
        ]
        if recall_candidates:
            best_t, best_p, best_r, best_f1 = max(recall_candidates, key=lambda x: x[1])
            print(f"Fallback threshold meeting clinical safety constraint (Recall >= {min_recall}): {best_t:.3f}")
        else:
            # Strategy 3: Global max F1
            best_idx = np.argmax(f1_pts)
            best_t = thresholds[best_idx]
            best_p, best_r, best_f1 = prec_pts[best_idx], rec_pts[best_idx], f1_pts[best_idx]
            print(f"Global maximum F1 threshold selected: {best_t:.3f}")

    metrics = evaluate_predictions(y_true, y_proba, threshold=best_t)
    return best_t, metrics


def print_evaluation_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    model_name: str = "Model",
    save_to_dir: Optional[Path] = REPORTS_DIR,
) -> None:
    """Print and optionally save formatted evaluation report.

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities.
        threshold: Operating decision threshold.
        model_name: Name of model for logging.
        save_to_dir: Directory to persist metrics JSON.
    """
    y_pred = (y_proba >= threshold).astype(int)
    metrics = evaluate_predictions(y_true, y_proba, threshold=threshold)

    print(f"\n=======================================================")
    print(f" 📊 EVALUATION SUMMARY: {model_name} (Threshold = {threshold:.3f})")
    print(f"=======================================================")
    print(f" ROC-AUC Score:          {metrics['roc_auc']:.4f}")
    print(f" PR-AUC Score:           {metrics['pr_auc']:.4f}")
    print(f" Clinical Recall:        {metrics['sensitivity_recall']:.2%}")
    print(f" Clinical Specificity:   {metrics['specificity']:.2%}")
    print(f" Precision (PPV):        {metrics['precision']:.2%}")
    print(f" F1-Score:               {metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  [ True Negatives: {metrics['true_negatives']:,}  |  False Positives: {metrics['false_positives']:,} ]")
    print(f"  [ False Negatives: {metrics['false_negatives']:,} |  True Positives:  {metrics['true_positives']:,} ]")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Not Readmitted (<30d)", "Readmitted (<30d)"]))
    print(f"=======================================================\n")

    if save_to_dir:
        save_path = Path(save_to_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        report_file = save_path / f"{model_name.lower()}_evaluation_metrics.json"
        with open(report_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {report_file.resolve()}")
