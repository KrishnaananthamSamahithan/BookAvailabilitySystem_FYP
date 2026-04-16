"""Concept drift detection module.

Uses Population Stability Index (PSI) for feature drift and
Kolmogorov-Smirnov test for distribution shift in predictions.
Called both offline (train.py) and online (app.py) to flag when
the model may need retraining.

Fixes applied:
  - OnlineDriftMonitor now uses collections.deque(maxlen=...) instead of list.pop(0)
    for O(1) append and automatic truncation.
  - Added threading.Lock to OnlineDriftMonitor.record() and check_drift()
    for thread-safe concurrent Flask requests.
"""

import threading
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# PSI threshold conventions (standard industry values)
PSI_STABLE = 0.10        # < 0.10 → no significant change
PSI_MONITOR = 0.25       # 0.10–0.25 → some shift, monitor
PSI_RETRAIN = 0.25       # > 0.25 → significant shift, consider retraining


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute Population Stability Index between reference and current distributions.

    PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))

    Args:
        reference: 1D array from the training period.
        current: 1D array from the scoring period.
        n_bins: Number of quantile-based bins.
        epsilon: Small constant to avoid log(0).

    Returns:
        PSI score (float). Higher = more drift.
    """
    reference = np.array(reference, dtype=float)
    current = np.array(current, dtype=float)

    # Drop NaNs
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    # Use quantile-based bins from reference distribution
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    # Ensure strictly increasing edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / max(len(reference), 1) + epsilon
    cur_pct = cur_counts / max(len(current), 1) + epsilon

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def ks_test(
    reference: np.ndarray,
    current: np.ndarray,
) -> Dict[str, float]:
    """Kolmogorov-Smirnov two-sample test for distribution shift."""
    from scipy import stats
    reference = np.array(reference, dtype=float)
    current = np.array(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "is_drifted": False}
    stat, p_value = stats.ks_2samp(reference, current)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_drifted": bool(p_value < 0.05),
    }


def run_feature_drift_analysis(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Dict[str, object]:
    """Run PSI + KS drift analysis for all features."""
    results: Dict[str, object] = {"numeric": {}, "categorical": {}, "summary": {}}
    drifted_features: List[str] = []

    # Numeric features — PSI + KS
    for feat in numeric_features:
        if feat not in train_frame.columns or feat not in test_frame.columns:
            continue
        ref_vals = train_frame[feat].dropna().to_numpy(dtype=float)
        cur_vals = test_frame[feat].dropna().to_numpy(dtype=float)
        psi = compute_psi(ref_vals, cur_vals)
        ks = ks_test(ref_vals, cur_vals)
        drift_flag = (not np.isnan(psi) and psi > PSI_MONITOR) or ks.get("is_drifted", False)
        if drift_flag:
            drifted_features.append(feat)
        results["numeric"][feat] = {
            "psi": round(float(psi), 4) if not np.isnan(psi) else None,
            "psi_status": _psi_label(psi),
            "ks_statistic": round(ks["statistic"], 4),
            "ks_p_value": round(ks["p_value"], 4),
            "is_drifted": drift_flag,
        }

    # Categorical features — PSI on frequency distributions
    for feat in categorical_features:
        if feat not in train_frame.columns or feat not in test_frame.columns:
            continue
        ref_counts = train_frame[feat].value_counts(normalize=True)
        cur_counts = test_frame[feat].value_counts(normalize=True)
        all_cats = set(ref_counts.index) | set(cur_counts.index)
        epsilon = 1e-6
        ref_pct = np.array([ref_counts.get(c, 0) + epsilon for c in all_cats])
        cur_pct = np.array([cur_counts.get(c, 0) + epsilon for c in all_cats])
        ref_pct /= ref_pct.sum()
        cur_pct /= cur_pct.sum()
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        drift_flag = psi > PSI_MONITOR
        if drift_flag:
            drifted_features.append(feat)
        results["categorical"][feat] = {
            "psi": round(psi, 4),
            "psi_status": _psi_label(psi),
            "unique_train": int(len(ref_counts)),
            "unique_current": int(len(cur_counts)),
            "new_categories": int(len(set(cur_counts.index) - set(ref_counts.index))),
            "is_drifted": drift_flag,
        }

    total_checked = len(results["numeric"]) + len(results["categorical"])
    results["summary"] = {
        "total_features_checked": total_checked,
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "drift_rate": round(len(drifted_features) / max(total_checked, 1), 4),
        "overall_status": _overall_drift_status(len(drifted_features), total_checked),
        "recommendation": _drift_recommendation(len(drifted_features), total_checked),
        "psi_thresholds": {
            "stable": f"< {PSI_STABLE}",
            "monitor": f"{PSI_STABLE}–{PSI_MONITOR}",
            "retrain": f"> {PSI_RETRAIN}",
        },
    }
    return results


def run_prediction_drift_analysis(
    reference_proba: np.ndarray,
    current_proba: np.ndarray,
    labels: List[str],
) -> Dict[str, object]:
    """Detect drift in the model's output probability distribution."""
    results: Dict[str, object] = {}
    drifted = []
    for idx, label in enumerate(labels):
        ref_vals = reference_proba[:, idx]
        cur_vals = current_proba[:, idx]
        psi = compute_psi(ref_vals, cur_vals)
        ks = ks_test(ref_vals, cur_vals)
        drift_flag = (not np.isnan(psi) and psi > PSI_MONITOR) or ks.get("is_drifted", False)
        if drift_flag:
            drifted.append(label)
        results[label] = {
            "psi": round(float(psi), 4) if not np.isnan(psi) else None,
            "psi_status": _psi_label(psi),
            "ks_statistic": round(ks["statistic"], 4),
            "ks_p_value": round(ks["p_value"], 4),
            "is_drifted": drift_flag,
        }
    results["summary"] = {
        "drifted_classes": drifted,
        "overall_status": _overall_drift_status(len(drifted), len(labels)),
    }
    return results


class OnlineDriftMonitor:
    """Lightweight in-memory drift monitor for the Flask serving layer.

    Accumulates a rolling window of predictions and compares the
    distribution against the training reference distribution.

    Thread-safe: uses threading.Lock around all buffer operations.
    Time-efficient: uses collections.deque(maxlen=window_size) for O(1)
    append and automatic oldest-item eviction.
    """

    def __init__(
        self,
        reference_proba: np.ndarray,
        labels: List[str],
        window_size: int = 500,
    ):
        self.reference_proba = reference_proba
        self.labels = labels
        self.window_size = window_size
        # FIX: Use deque(maxlen=...) instead of list.pop(0)
        # deque is O(1) append+popleft vs O(n) list.pop(0)
        # maxlen enforces the window automatically without explicit eviction
        self._buffer: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()  # FIX: thread-safe for concurrent Flask requests

    def record(self, proba_vector: np.ndarray) -> None:
        """Record a new prediction probability vector (thread-safe)."""
        with self._lock:
            self._buffer.append(proba_vector.copy())

    def check_drift(self) -> Dict[str, object]:
        """Check drift against reference using the current buffer (thread-safe)."""
        with self._lock:
            buffer_copy = list(self._buffer)
        if len(buffer_copy) < 50:
            return {
                "status": "insufficient_data",
                "buffer_size": len(buffer_copy),
                "required": 50,
            }
        current_proba = np.array(buffer_copy)
        result = run_prediction_drift_analysis(
            self.reference_proba, current_proba, self.labels
        )
        result["buffer_size"] = len(buffer_copy)
        return result

    def reset(self) -> None:
        """Clear the buffer (e.g., after model retraining)."""
        with self._lock:
            self._buffer.clear()


def _psi_label(psi: float) -> str:
    if np.isnan(psi):
        return "unknown"
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_MONITOR:
        return "monitor"
    return "drift_detected"


def _overall_drift_status(n_drifted: int, n_total: int) -> str:
    if n_total == 0:
        return "unknown"
    rate = n_drifted / n_total
    if rate == 0:
        return "stable"
    if rate < 0.2:
        return "minor_drift"
    if rate < 0.5:
        return "moderate_drift"
    return "significant_drift"


def _drift_recommendation(n_drifted: int, n_total: int) -> str:
    if n_total == 0:
        return "No features checked."
    rate = n_drifted / n_total
    if rate == 0:
        return "No action needed. Model is operating on a stable distribution."
    if rate < 0.2:
        return "Minor drift detected. Monitor closely but retraining not yet required."
    if rate < 0.5:
        return "Moderate drift detected. Schedule retraining within the next cycle."
    return "Significant drift detected. Retrain the model as soon as possible."
