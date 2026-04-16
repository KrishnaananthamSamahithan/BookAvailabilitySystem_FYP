from typing import Dict, List

import numpy as np
import pandas as pd


def suppression_policy_simulation(frame: pd.DataFrame, probabilities: np.ndarray, labels: List[str], thresholds: List[float]) -> List[Dict[str, object]]:
    enriched = frame.copy()
    enriched["risk_unbookable"] = _risk_unbookable(probabilities, labels)

    rows: List[Dict[str, object]] = []
    for threshold in thresholds:
        suppressed_mask = enriched["risk_unbookable"] >= threshold
        retained_mask = ~suppressed_mask
        rows.append(
            {
                "threshold": threshold,
                "suppressed_rows": int(suppressed_mask.sum()),
                "retained_rows": int(retained_mask.sum()),
                "suppressed_true_bookable": int(((enriched["outcome_label"] == "bookable") & suppressed_mask).sum()),
                "suppressed_true_unbookable": int(((enriched["outcome_label"] != "bookable") & suppressed_mask).sum()),
                "retained_true_bookable": int(((enriched["outcome_label"] == "bookable") & retained_mask).sum()),
                "retained_true_unbookable": int(((enriched["outcome_label"] != "bookable") & retained_mask).sum()),
                "retained_bookable_rate": float((enriched.loc[retained_mask, "outcome_label"] == "bookable").mean()) if retained_mask.any() else 0.0,
            }
        )
    return rows


def reranking_proxy_simulation(frame: pd.DataFrame, probabilities: np.ndarray, labels: List[str]) -> Dict[str, object]:
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

    enriched["ml_rank_score"] = enriched["p_bookable"] - 0.35 * enriched["risk_unbookable"] - 0.05 * price_penalty
    ml_top = (
        enriched.sort_values(["search_group_proxy", "ml_rank_score"], ascending=[True, False])
        .groupby("search_group_proxy", as_index=False)
        .first()
    )

    return {
        "grouping_key": "search_group_proxy",
        "baseline_name": baseline_name,
        "num_groups": int(enriched["search_group_proxy"].nunique()),
        "baseline_top1_bookable_rate": float((baseline["outcome_label"] == "bookable").mean()),
        "ml_top1_bookable_rate": float((ml_top["outcome_label"] == "bookable").mean()),
        "baseline_top1_unavailable_rate": float((baseline["outcome_label"] == "unavailable").mean()),
        "ml_top1_unavailable_rate": float((ml_top["outcome_label"] == "unavailable").mean()),
        "baseline_top1_technical_failure_rate": float((baseline["outcome_label"] == "technical_failure").mean()),
        "ml_top1_technical_failure_rate": float((ml_top["outcome_label"] == "technical_failure").mean()),
    }


def _risk_unbookable(probabilities: np.ndarray, labels: List[str]) -> np.ndarray:
    risk_columns = [
        labels.index("price_changed"),
        labels.index("unavailable"),
        labels.index("technical_failure"),
    ]
    return probabilities[:, risk_columns].sum(axis=1)
