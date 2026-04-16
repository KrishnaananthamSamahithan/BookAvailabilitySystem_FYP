"""Classification and calibration metrics module.

Extended with:
  - Precision-recall AUC per class
  - Zero-support class detection and warnings
  - Threshold sweep for optimal operating point
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, labels: List[int]) -> float:
    one_hot = np.eye(len(labels))[y_true]
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true_binary: np.ndarray, y_prob_binary: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        mask = (y_prob_binary >= bins[idx]) & (
            y_prob_binary < bins[idx + 1] if idx < n_bins - 1 else y_prob_binary <= bins[idx + 1]
        )
        if not np.any(mask):
            continue
        avg_conf = np.mean(y_prob_binary[mask])
        avg_acc = np.mean(y_true_binary[mask])
        ece += (np.sum(mask) / len(y_prob_binary)) * abs(avg_conf - avg_acc)
    return float(ece)


def classwise_ece(y_true: np.ndarray, y_prob: np.ndarray, labels: List[str]) -> Dict[str, float]:
    metrics = {}
    for idx, label in enumerate(labels):
        binary_true = (y_true == idx).astype(int)
        metrics[label] = expected_calibration_error(binary_true, y_prob[:, idx])
    return metrics


def reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, labels: List[str], n_bins: int = 10
) -> Dict[str, List[Dict[str, float]]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    output: Dict[str, List[Dict[str, float]]] = {}
    for idx, label in enumerate(labels):
        binary_true = (y_true == idx).astype(int)
        class_rows = []
        for bin_index in range(n_bins):
            lower = bins[bin_index]
            upper = bins[bin_index + 1]
            mask = (y_prob[:, idx] >= lower) & (
                y_prob[:, idx] < upper if bin_index < n_bins - 1 else y_prob[:, idx] <= upper
            )
            if not np.any(mask):
                continue
            class_rows.append(
                {
                    "bin_lower": float(lower),
                    "bin_upper": float(upper),
                    "avg_confidence": float(np.mean(y_prob[mask, idx])),
                    "avg_accuracy": float(np.mean(binary_true[mask])),
                    "count": int(np.sum(mask)),
                }
            )
        output[label] = class_rows
    return output


def roc_auc_per_class(y_true: np.ndarray, y_prob: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """Compute one-vs-rest ROC-AUC for each class."""
    result: Dict[str, float] = {}
    for idx, label in enumerate(labels):
        binary_true = (y_true == idx).astype(int)
        if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
            result[label] = float("nan")
        else:
            try:
                result[label] = float(roc_auc_score(binary_true, y_prob[:, idx]))
            except Exception:
                result[label] = float("nan")
    present = [v for v in result.values() if not np.isnan(v)]
    result["macro_avg"] = float(np.mean(present)) if present else float("nan")
    return result


def precision_recall_auc_per_class(y_true: np.ndarray, y_prob: np.ndarray, labels: List[str]) -> Dict[str, object]:
    """Compute precision-recall AUC (average precision) per class and store curve points."""
    result: Dict[str, object] = {}
    for idx, label in enumerate(labels):
        binary_true = (y_true == idx).astype(int)
        pos_count = int(binary_true.sum())
        if pos_count == 0:
            result[label] = {
                "average_precision": float("nan"),
                "support": 0,
                "warning": "zero_support_class",
                "curve_precision": [],
                "curve_recall": [],
                "curve_thresholds": [],
            }
            continue
        try:
            ap = float(average_precision_score(binary_true, y_prob[:, idx]))
            precision, recall, thresholds = precision_recall_curve(binary_true, y_prob[:, idx])
            # Downsample curve to max 50 points for JSON storage
            step = max(1, len(precision) // 50)
            result[label] = {
                "average_precision": round(ap, 4),
                "support": pos_count,
                "curve_precision": [round(float(v), 4) for v in precision[::step]],
                "curve_recall": [round(float(v), 4) for v in recall[::step]],
                "curve_thresholds": [round(float(v), 4) for v in thresholds[::step]],
            }
        except Exception as exc:
            result[label] = {"average_precision": float("nan"), "error": str(exc), "support": pos_count}
    present_ap = [v["average_precision"] for v in result.values() if isinstance(v, dict) and not np.isnan(v.get("average_precision", float("nan")))]
    result["macro_avg_ap"] = float(np.mean(present_ap)) if present_ap else float("nan")
    return result


def check_zero_support_classes(y_true: np.ndarray, labels: List[str]) -> List[Dict[str, object]]:
    """Detect and report classes with zero support in the evaluation set.

    In multi-class problems, zero-support classes inflate the denominator of
    macro metrics without contributing any signal. This must be explicitly flagged.
    """
    issues = []
    for idx, label in enumerate(labels):
        count = int(np.sum(y_true == idx))
        if count == 0:
            issues.append({
                "label": label,
                "support": 0,
                "warning": (
                    f"Class '{label}' has zero support in the evaluation set. "
                    "F1-score will be 0.0 and macro averages are penalized. "
                    "This may indicate a data collection issue or a very rare class. "
                    "Consider reporting macro_f1_present_classes as the primary metric."
                ),
                "recommendation": "Use macro_f1_present_classes (excludes zero-support classes) as headline metric.",
            })
    return issues


def optimal_threshold_per_class(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str],
    metric: str = "f1",
) -> Dict[str, object]:
    """Find optimal decision threshold for each class using precision-recall curve.

    For each class, sweeps thresholds and selects the one that maximises the
    specified metric (F1 by default). This supports business threshold tuning
    (e.g., minimise unbookable false positives).
    """
    result: Dict[str, object] = {}
    for idx, label in enumerate(labels):
        binary_true = (y_true == idx).astype(int)
        if binary_true.sum() == 0:
            result[label] = {"optimal_threshold": float("nan"), "best_f1": float("nan"), "support": 0}
            continue
        try:
            precision, recall, thresholds = precision_recall_curve(binary_true, y_prob[:, idx])
            # F1 at each threshold
            with np.errstate(divide="ignore", invalid="ignore"):
                f1_scores = np.where(
                    (precision + recall) == 0, 0, 2 * precision * recall / (precision + recall)
                )
            # thresholds has len = len(precision) - 1
            best_idx = int(np.argmax(f1_scores[:-1]))
            result[label] = {
                "optimal_threshold": round(float(thresholds[best_idx]), 4),
                "best_f1": round(float(f1_scores[best_idx]), 4),
                "precision_at_optimal": round(float(precision[best_idx]), 4),
                "recall_at_optimal": round(float(recall[best_idx]), 4),
                "support": int(binary_true.sum()),
            }
        except Exception as exc:
            result[label] = {"optimal_threshold": float("nan"), "error": str(exc)}
    return result


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: List[str]) -> Dict[str, object]:
    predicted = np.argmax(y_prob, axis=1)
    numeric_labels = list(range(len(labels)))
    present_numeric_labels = sorted(int(v) for v in np.unique(y_true))
    if not present_numeric_labels:
        present_numeric_labels = numeric_labels
    present_label_names = [labels[index] for index in present_numeric_labels]

    # Zero-support check
    zero_support_issues = check_zero_support_classes(y_true, labels)

    metrics = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_f1_present_classes": float(
            f1_score(y_true, predicted, labels=present_numeric_labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob, labels=numeric_labels)),
        "multiclass_brier": multiclass_brier_score(y_true, y_prob, numeric_labels),
        "roc_auc_per_class": roc_auc_per_class(y_true, y_prob, labels),
        "precision_recall_auc": precision_recall_auc_per_class(y_true, y_prob, labels),
        "optimal_thresholds": optimal_threshold_per_class(y_true, y_prob, labels),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=numeric_labels).tolist(),
        "classification_report": classification_report(
            y_true,
            predicted,
            labels=numeric_labels,
            target_names=labels,
            zero_division=0,
            output_dict=True,
        ),
        "classwise_ece": classwise_ece(y_true, y_prob, labels),
        "reliability_table": reliability_table(y_true, y_prob, labels),
        "present_labels": present_label_names,
        "zero_support_labels": [label for idx, label in enumerate(labels) if idx not in present_numeric_labels],
        "zero_support_warnings": zero_support_issues,
        "class_support": {
            label: int(np.sum(y_true == idx))
            for idx, label in enumerate(labels)
        },
    }
    metrics["ece_macro"] = float(np.mean(list(metrics["classwise_ece"].values())))
    auc_values = [v for v in metrics["roc_auc_per_class"].values() if not np.isnan(v) and v != metrics["roc_auc_per_class"].get("macro_avg")]
    metrics["roc_auc_macro"] = float(np.mean(auc_values)) if auc_values else float("nan")
    return metrics
