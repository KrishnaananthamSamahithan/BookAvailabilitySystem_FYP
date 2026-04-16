"""Exploratory Data Analysis module.

Produces a comprehensive EDA report saved as JSON + generates
matplotlib reliability diagrams as PNG files. Called once at the
start of the training pipeline before any model fitting.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import save_json


def run_eda(processed: pd.DataFrame, reports_dir: Path) -> Dict[str, object]:
    """Run full EDA on the preprocessed frame and save a JSON report.

    Args:
        processed: The output of preprocess_raw_frame() — already cleaned.
        reports_dir: Directory to write eda_report.json.

    Returns:
        Dictionary with all EDA findings (also persisted to disk).
    """
    report: Dict[str, object] = {}

    # -------------------------------------------------------------------
    # 1. Dataset shape
    # -------------------------------------------------------------------
    report["dataset_shape"] = {"rows": int(len(processed)), "columns": int(len(processed.columns))}

    # -------------------------------------------------------------------
    # 2. Label distribution & imbalance
    # -------------------------------------------------------------------
    label_counts = processed["outcome_label"].value_counts(dropna=False).to_dict()
    label_counts = {str(k): int(v) for k, v in label_counts.items()}
    total = sum(label_counts.values())
    label_proportions = {k: round(v / total, 4) for k, v in label_counts.items()}
    max_count = max(label_counts.values()) if label_counts else 1
    min_count = min(label_counts.values()) if label_counts else 1
    imbalance_ratio = round(max_count / max(min_count, 1), 2)

    report["label_distribution"] = {
        "counts": label_counts,
        "proportions": label_proportions,
        "imbalance_ratio_max_to_min": imbalance_ratio,
        "dominant_class": max(label_counts, key=label_counts.get),
    }

    # -------------------------------------------------------------------
    # 3. Missing value rates
    # -------------------------------------------------------------------
    missing_rates = processed.isna().mean().round(4)
    report["missing_rates"] = {col: float(rate) for col, rate in missing_rates.items() if rate > 0}
    report["columns_with_any_missing"] = int((missing_rates > 0).sum())
    report["columns_fully_present"] = int((missing_rates == 0).sum())

    # -------------------------------------------------------------------
    # 4. Numeric feature summaries
    # -------------------------------------------------------------------
    numeric_cols = [
        "days_to_departure", "price_total", "passenger_count",
        "stop_count", "adults", "children", "infants",
    ]
    numeric_summaries: Dict[str, object] = {}
    for col in numeric_cols:
        if col not in processed.columns:
            continue
        s = processed[col].dropna()
        if len(s) == 0:
            continue
        numeric_summaries[col] = {
            "count": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "missing_rate": float(processed[col].isna().mean()),
        }
    report["numeric_summaries"] = numeric_summaries

    # -------------------------------------------------------------------
    # 5. Cardinality of categorical features
    # -------------------------------------------------------------------
    cat_cols = [
        "origin_airport", "destination_airport", "airline_code", "cabin_class",
        "trip_type", "meta_engine", "provider_key", "referrer_domain",
        "market", "locale", "device_os",
    ]
    cardinality: Dict[str, object] = {}
    for col in cat_cols:
        if col not in processed.columns:
            continue
        vc = processed[col].value_counts()
        cardinality[col] = {
            "unique_values": int(vc.shape[0]),
            "top_5": {str(k): int(v) for k, v in vc.head(5).items()},
            "coverage_top_5_pct": round(float(vc.head(5).sum() / len(processed) * 100), 2),
        }
    report["categorical_cardinality"] = cardinality

    # -------------------------------------------------------------------
    # 6. Temporal density — records per calendar month
    # -------------------------------------------------------------------
    if "prediction_time" in processed.columns and processed["prediction_time"].notna().any():
        monthly = (
            processed.dropna(subset=["prediction_time"])
            .groupby(processed["prediction_time"].dt.to_period("M"))
            .size()
        )
        report["records_per_month"] = {str(k): int(v) for k, v in monthly.items()}
        report["date_range"] = {
            "earliest": str(processed["prediction_time"].min()),
            "latest": str(processed["prediction_time"].max()),
            "total_days": int(
                (processed["prediction_time"].max() - processed["prediction_time"].min()).days
            ) if processed["prediction_time"].notna().any() else 0,
        }
    else:
        report["records_per_month"] = {}
        report["date_range"] = {}

    # -------------------------------------------------------------------
    # 7. Days-to-departure distribution (by label)
    # -------------------------------------------------------------------
    dtd_stats: Dict[str, object] = {}
    for label in ["bookable", "price_changed", "unavailable", "technical_failure"]:
        subset = processed.loc[processed["outcome_label"] == label, "days_to_departure"].dropna()
        if len(subset) == 0:
            continue
        dtd_stats[label] = {
            "median_days": float(subset.median()),
            "mean_days": float(subset.mean()),
            "pct_same_day": float((subset == 0).mean()),
            "pct_short_lead": float((subset <= 7).mean()),
            "pct_long_lead": float((subset >= 60).mean()),
        }
    report["days_to_departure_by_label"] = dtd_stats

    # -------------------------------------------------------------------
    # 8. Top routes and airlines
    # -------------------------------------------------------------------
    if "route" in processed.columns:
        top_routes = processed["route"].value_counts().head(10)
        report["top_10_routes"] = {str(k): int(v) for k, v in top_routes.items()}

    if "airline_code" in processed.columns:
        top_airlines = processed["airline_code"].value_counts().head(10)
        report["top_10_airlines"] = {str(k): int(v) for k, v in top_airlines.items()}

    # -------------------------------------------------------------------
    # 9. Label breakdown by meta engine
    # -------------------------------------------------------------------
    if "meta_engine" in processed.columns:
        engine_label = (
            processed.groupby(["meta_engine", "outcome_label"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )
        report["label_by_meta_engine"] = {
            str(k): {str(c): int(v) for c, v in row.items()}
            for k, row in engine_label.items()
        }

    # -------------------------------------------------------------------
    # 10. Hourly booking pattern
    # -------------------------------------------------------------------
    if "search_hour" in processed.columns:
        hourly = processed.groupby("search_hour")["outcome_label"].apply(
            lambda s: (s == "bookable").mean()
        )
        report["bookable_rate_by_hour"] = {
            int(k): round(float(v), 4) for k, v in hourly.items()
        }

    # -------------------------------------------------------------------
    # 11. Lead-time bucket distribution
    # -------------------------------------------------------------------
    if "days_to_departure_bucket" in processed.columns:
        bucket_dist = processed["days_to_departure_bucket"].value_counts().to_dict()
        report["lead_time_bucket_distribution"] = {str(k): int(v) for k, v in bucket_dist.items()}

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    save_json(reports_dir / "eda_report.json", report)
    return report


def generate_reliability_diagrams(
    reliability_table: Dict[str, List[Dict[str, float]]],
    labels: List[str],
    output_path: Path,
    title_prefix: str = "",
) -> bool:
    """Generate reliability (calibration) diagrams for each class.

    Returns True if saved successfully, False if matplotlib unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_classes = len(labels)
        fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), squeeze=False)
        fig.suptitle(f"{title_prefix}Reliability Diagrams (Calibration Curves)", fontsize=12, fontweight="bold")

        for col_idx, label in enumerate(labels):
            ax = axes[0][col_idx]
            bins = reliability_table.get(label, [])
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration", alpha=0.7)
            if bins:
                confidences = [b["avg_confidence"] for b in bins]
                accuracies = [b["avg_accuracy"] for b in bins]
                counts = [b["count"] for b in bins]
                ax.plot(confidences, accuracies, "b-o", markersize=5, linewidth=1.5, label="Model")
                # Shade gap from diagonal
                ax.fill_between(confidences, confidences, accuracies, alpha=0.15, color="red", label="Gap")
                # Size points by sample count
                total_pts = sum(counts)
                for conf, acc, cnt in zip(confidences, accuracies, counts):
                    ax.scatter(conf, acc, s=max(10, cnt / total_pts * 500), color="blue", alpha=0.5, zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Mean predicted probability", fontsize=9)
            ax.set_ylabel("Fraction of positives", fontsize=9)
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        return False
