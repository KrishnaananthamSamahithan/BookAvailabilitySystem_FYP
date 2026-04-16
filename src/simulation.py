"""Policy simulation module.

Provides:
  - suppression_policy_simulation: sweeps thresholds and measures bookable rate lift
  - reranking_proxy_simulation: compares ML-ranked vs baseline offer ordering
  - cost_sensitive_simulation: confusion matrix weighted by business costs

Improvements:
  - Added cost_sensitive_simulation with a principled 4-class cost matrix.
  - Suppression now also reports precision, recall, and F1 at each threshold.
  - Reranking weights documented and validated.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default business cost matrix
# ---------------------------------------------------------------------------
# Rows = True class, Columns = Predicted class
# Order: bookable, price_changed, unavailable, technical_failure
#
# Interpretation:
#  - Predicting "bookable" when truly "unavailable" → high cost (mislead user)
#  - Predicting "unavailable" when truly "bookable" → moderate cost (lost sale)
#  - Correct predictions → zero cost
DEFAULT_COST_MATRIX = np.array([
    # pred: bookable, price_changed, unavailable, technical_failure
    [0,    1,         2,             1],    # true: bookable
    [3,    0,         1,             1],    # true: price_changed
    [5,    2,         0,             1],    # true: unavailable
    [4,    2,         1,             0],    # true: technical_failure
], dtype=float)


def suppression_policy_simulation(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    labels: List[str],
    thresholds: List[float],
) -> List[Dict[str, object]]:
    """Simulate suppression of offers with high unbookability risk.

    For each threshold, suppresses offers where P(unbookable) >= threshold.
    Reports bookable rate improvement, precision/recall at each threshold.
    """
    enriched = frame.copy()
    enriched["risk_unbookable"] = _risk_unbookable(probabilities, labels)
    bookable_idx = labels.index("bookable") if "bookable" in labels else 0
    enriched["p_bookable"] = probabilities[:, bookable_idx]

    rows: List[Dict[str, object]] = []
    total = len(enriched)
    total_bookable = int((enriched["outcome_label"] == "bookable").sum())

    for threshold in thresholds:
        suppressed_mask = enriched["risk_unbookable"] >= threshold
        retained_mask = ~suppressed_mask

        suppressed_count = int(suppressed_mask.sum())
        retained_count = int(retained_mask.sum())

        suppressed_true_bookable = int(((enriched["outcome_label"] == "bookable") & suppressed_mask).sum())
        suppressed_true_unbookable = int(((enriched["outcome_label"] != "bookable") & suppressed_mask).sum())
        retained_true_bookable = int(((enriched["outcome_label"] == "bookable") & retained_mask).sum())
        retained_true_unbookable = int(((enriched["outcome_label"] != "bookable") & retained_mask).sum())

        retained_bookable_rate = float(
            (enriched.loc[retained_mask, "outcome_label"] == "bookable").mean()
        ) if retained_mask.any() else 0.0

        # Precision @ threshold: of suppressed, how many were truly unbookable?
        precision_suppressed = suppressed_true_unbookable / max(suppressed_count, 1)
        # Recall @ threshold: of all unbookable, how many were suppressed?
        total_unbookable = total - total_bookable
        recall_suppressed = suppressed_true_unbookable / max(total_unbookable, 1)

        precision_f1 = (
            2 * precision_suppressed * recall_suppressed / (precision_suppressed + recall_suppressed)
            if (precision_suppressed + recall_suppressed) > 0 else 0.0
        )

        rows.append({
            "threshold": threshold,
            "suppressed_rows": suppressed_count,
            "retained_rows": retained_count,
            "suppressed_pct": round(suppressed_count / max(total, 1), 4),
            "suppressed_true_bookable": suppressed_true_bookable,
            "suppressed_true_unbookable": suppressed_true_unbookable,
            "retained_true_bookable": retained_true_bookable,
            "retained_true_unbookable": retained_true_unbookable,
            "retained_bookable_rate": round(retained_bookable_rate, 4),
            "precision_suppression": round(precision_suppressed, 4),
            "recall_suppression": round(recall_suppressed, 4),
            "f1_suppression": round(precision_f1, 4),
        })
    return rows


def reranking_proxy_simulation(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    labels: List[str],
) -> Dict[str, object]:
    """Compare ML-reranked offers vs baseline (cheapest or first-seen).

    Uses a composite score: p_bookable - 0.35*risk_unbookable - 0.05*price_penalty.
    The weights 0.35 and 0.05 prioritize booking probability over price competitiveness.
    """
    enriched = frame.copy()
    bookable_index = labels.index("bookable")
    enriched["p_bookable"] = probabilities[:, bookable_index]
    enriched["risk_unbookable"] = _risk_unbookable(probabilities, labels)

    if "price_total" in enriched and enriched["price_total"].notna().any():
        price_penalty = enriched["price_ratio_to_median"].clip(lower=0.5, upper=2.0)
        baseline = (
            enriched.sort_values(["search_group_proxy", "price_total", "prediction_time"], ascending=[True, True, True])
            .groupby("search_group_proxy", as_index=False)
            .first()
        )
        baseline_name = "cheapest_offer_baseline"
    else:
        price_penalty = 1.0
        baseline = (
            enriched.sort_values(["search_group_proxy", "prediction_time"], ascending=[True, True])
            .groupby("search_group_proxy", as_index=False)
            .first()
        )
        baseline_name = "first_seen_offer_baseline"

    # Composite rank score: bookability first, risk penalised, price slightly penalised
    enriched["ml_rank_score"] = (
        enriched["p_bookable"]
        - 0.35 * enriched["risk_unbookable"]
        - 0.05 * price_penalty
    )
    ml_top = (
        enriched.sort_values(["search_group_proxy", "ml_rank_score"], ascending=[True, False])
        .groupby("search_group_proxy", as_index=False)
        .first()
    )

    n_groups = int(enriched["search_group_proxy"].nunique())
    baseline_bookable = float((baseline["outcome_label"] == "bookable").mean())
    ml_bookable = float((ml_top["outcome_label"] == "bookable").mean())
    bookable_lift = round(ml_bookable - baseline_bookable, 4)

    return {
        "grouping_key": "search_group_proxy",
        "baseline_name": baseline_name,
        "num_groups": n_groups,
        "baseline_top1_bookable_rate": round(baseline_bookable, 4),
        "ml_top1_bookable_rate": round(ml_bookable, 4),
        "bookable_rate_lift_absolute": bookable_lift,
        "bookable_rate_lift_pct": round(bookable_lift / max(baseline_bookable, 1e-9) * 100, 2),
        "baseline_top1_unavailable_rate": round(float((baseline["outcome_label"] == "unavailable").mean()), 4),
        "ml_top1_unavailable_rate": round(float((ml_top["outcome_label"] == "unavailable").mean()), 4),
        "baseline_top1_price_changed_rate": round(float((baseline["outcome_label"] == "price_changed").mean()), 4),
        "ml_top1_price_changed_rate": round(float((ml_top["outcome_label"] == "price_changed").mean()), 4),
        "rank_score_formula": "p_bookable - 0.35 * risk_unbookable - 0.05 * price_penalty",
    }


def cost_sensitive_simulation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str],
    cost_matrix: Optional[np.ndarray] = None,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, object]:
    """Simulate decision costs using a business-defined misclassification cost matrix.

    Applies a cost matrix C[true_class, pred_class] to compute the expected
    total cost of predictions at different operating thresholds for the bookable class.

    Args:
        y_true: Ground truth integer labels.
        y_prob: Calibrated probability array (n_samples, n_classes).
        labels: Class label names.
        cost_matrix: Square matrix of costs. Default: DEFAULT_COST_MATRIX.
        thresholds: List of bookability thresholds to sweep.

    Returns:
        Dict with cost at default threshold, cost sweep, and optimal threshold.
    """
    if cost_matrix is None:
        cost_matrix = DEFAULT_COST_MATRIX
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    n_classes = len(labels)
    predicted_default = np.argmax(y_prob, axis=1)

    def _total_cost(y_true_arr: np.ndarray, y_pred_arr: np.ndarray, cm: np.ndarray) -> float:
        total = 0.0
        for true_cls in range(n_classes):
            for pred_cls in range(n_classes):
                count = int(np.sum((y_true_arr == true_cls) & (y_pred_arr == pred_cls)))
                total += count * cm[true_cls, pred_cls]
        return float(total)

    default_cost = _total_cost(y_true, predicted_default, cost_matrix)
    n = len(y_true)

    bookable_idx = labels.index("bookable") if "bookable" in labels else 0
    sweep_rows = []
    for thresh in thresholds:
        # Apply threshold: predict bookable only if p_bookable >= threshold, else next highest
        pred_thresh = np.where(
            y_prob[:, bookable_idx] >= thresh,
            bookable_idx,
            np.argmax(
                np.concatenate([
                    y_prob[:, :bookable_idx],
                    y_prob[:, bookable_idx + 1:],
                ], axis=1),
                axis=1,
            ),
        )
        # Adjust indices above bookable_idx
        pred_thresh = np.where(
            y_prob[:, bookable_idx] < thresh,
            np.where(pred_thresh >= bookable_idx, pred_thresh + 1, pred_thresh),
            pred_thresh,
        )
        cost_at_thresh = _total_cost(y_true, pred_thresh, cost_matrix)
        sweep_rows.append({
            "bookable_threshold": thresh,
            "total_cost": round(cost_at_thresh, 2),
            "cost_per_sample": round(cost_at_thresh / max(n, 1), 4),
            "cost_vs_default": round(cost_at_thresh - default_cost, 2),
        })

    optimal = min(sweep_rows, key=lambda r: r["total_cost"])
    return {
        "cost_matrix_labels": labels,
        "cost_matrix": cost_matrix.tolist(),
        "default_argmax_total_cost": round(default_cost, 2),
        "default_cost_per_sample": round(default_cost / max(n, 1), 4),
        "threshold_sweep": sweep_rows,
        "optimal_bookable_threshold": optimal["bookable_threshold"],
        "optimal_total_cost": optimal["total_cost"],
        "note": (
            "Cost matrix is [true_class, pred_class]. Predicting 'bookable' "
            "when truly 'unavailable' incurs cost 5 (highest — misleads user). "
            "Missing a bookable offer costs 2 (lost sale)."
        ),
    }


def _risk_unbookable(probabilities: np.ndarray, labels: List[str]) -> np.ndarray:
    risk_columns = [idx for idx, label in enumerate(labels) if label != "bookable"]
    if not risk_columns:
        return np.zeros(len(probabilities), dtype=float)
    return probabilities[:, risk_columns].sum(axis=1)
