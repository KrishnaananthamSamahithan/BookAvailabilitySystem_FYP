from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, labels: List[int]) -> float:
    one_hot = np.eye(len(labels))[y_true]
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true_binary: np.ndarray, y_prob_binary: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        mask = (y_prob_binary >= bins[idx]) & (y_prob_binary < bins[idx + 1] if idx < n_bins - 1 else y_prob_binary <= bins[idx + 1])
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


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, labels: List[str], n_bins: int = 10) -> Dict[str, List[Dict[str, float]]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    output: Dict[str, List[Dict[str, float]]] = {}
    for idx, label in enumerate(labels):
        binary_true = (y_true == idx).astype(int)
        class_rows = []
        for bin_index in range(n_bins):
            lower = bins[bin_index]
            upper = bins[bin_index + 1]
            mask = (y_prob[:, idx] >= lower) & (y_prob[:, idx] < upper if bin_index < n_bins - 1 else y_prob[:, idx] <= upper)
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


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: List[str]) -> Dict[str, object]:
    predicted = np.argmax(y_prob, axis=1)
    numeric_labels = list(range(len(labels)))
    metrics = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob, labels=numeric_labels)),
        "multiclass_brier": multiclass_brier_score(y_true, y_prob, numeric_labels),
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
    }
    metrics["ece_macro"] = float(np.mean(list(metrics["classwise_ece"].values())))
    return metrics
