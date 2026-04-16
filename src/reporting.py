from typing import Dict, List

import numpy as np
import pandas as pd

from src.labels import FINAL_LABELS
from src.metrics import classwise_ece


def build_schema_resolution_report(schema_resolution, prediction_time_logical_name: str = "prediction_time") -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for logical_name, resolved in schema_resolution.required.items():
        rows.append(
            {
                "logical_field": logical_name,
                "resolved_raw_column": resolved,
                "matched_alias": schema_resolution.matched_aliases.get(logical_name),
                "is_required": True,
                "selected_as_prediction_time": logical_name == prediction_time_logical_name,
            }
        )
    for logical_name, resolved in schema_resolution.optional.items():
        rows.append(
            {
                "logical_field": logical_name,
                "resolved_raw_column": resolved,
                "matched_alias": schema_resolution.matched_aliases.get(logical_name),
                "is_required": False,
                "selected_as_prediction_time": False,
            }
        )
    return rows


def subgroup_metrics(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_column: str,
    labels: List[str] | None = None,
    top_n: int = 20,
    min_rows: int = 100,
) -> List[Dict[str, object]]:
    labels = labels or FINAL_LABELS
    if group_column not in frame.columns:
        return []

    enriched = frame[[group_column]].copy()
    enriched["y_true"] = y_true
    enriched["predicted"] = np.argmax(y_prob, axis=1)
    counts = enriched[group_column].value_counts().head(top_n)

    rows: List[Dict[str, object]] = []
    for group_value, count in counts.items():
        if count < min_rows:
            continue
        mask = enriched[group_column] == group_value
        group_true = y_true[mask.to_numpy()]
        group_prob = y_prob[mask.to_numpy()]
        predicted = np.argmax(group_prob, axis=1)

        balanced_accuracy = _balanced_accuracy(group_true, predicted)
        macro_f1 = _macro_f1(group_true, predicted, len(labels))
        logloss = _safe_log_loss(group_true, group_prob)
        ece = classwise_ece(group_true, group_prob, labels)
        rows.append(
            {
                "group_column": group_column,
                "group_value": str(group_value),
                "row_count": int(count),
                "macro_f1": float(macro_f1),
                "balanced_accuracy": float(balanced_accuracy),
                "log_loss": float(logloss),
                "classwise_ece": ece,
                "ece_macro": float(np.mean(list(ece.values()))),
            }
        )
    return rows


def calibration_parity_report(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subgroup_columns: List[str],
    labels: List[str] | None = None,
) -> Dict[str, List[Dict[str, object]]]:
    labels = labels or FINAL_LABELS
    report: Dict[str, List[Dict[str, object]]] = {}
    for column in subgroup_columns:
        report[column] = subgroup_metrics(frame, y_true, y_prob, column, labels=labels)
    return report


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    per_class = []
    for label in np.unique(y_true):
        mask = y_true == label
        if np.sum(mask) == 0:
            continue
        per_class.append(float(np.mean(y_pred[mask] == y_true[mask])))
    return float(np.mean(per_class)) if per_class else 0.0


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    scores = []
    for label in range(n_classes):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def _safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    one_hot = np.eye(y_prob.shape[1])[y_true]
    clipped = np.clip(y_prob, 1e-12, 1.0)
    return float(-np.mean(np.sum(one_hot * np.log(clipped), axis=1)))
