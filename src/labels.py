from typing import Tuple

import pandas as pd


FINAL_LABELS = ["bookable", "price_changed", "unavailable", "technical_failure"]


def canonicalize_status(value) -> str:
    if pd.isna(value):
        return "ambiguous"

    status = str(value).strip().lower()
    if not status:
        return "ambiguous"

    if any(token in status for token in ["booked", "success", "bookable"]):
        return "bookable"
    if any(token in status for token in ["price mismatch", "price changed", "fare changed", "price mismatch/fare changed"]):
        return "price_changed"
    if any(token in status for token in ["not available", "unavailable", "sold out"]):
        return "unavailable"
    if any(token in status for token in ["technical failure", "timeout", "redirect error", "session expired", "failed", "error"]):
        return "technical_failure"
    if any(token in status for token in ["not booked", "abandoned", "unknown", "cancelled"]):
        return "ambiguous"
    return "ambiguous"


def label_confidence(label: str) -> float:
    return 1.0 if label in FINAL_LABELS else 0.5


def build_label_frame(status_series: pd.Series) -> pd.DataFrame:
    labels = status_series.apply(canonicalize_status)
    return pd.DataFrame(
        {
            "raw_status": status_series.astype("string"),
            "outcome_label": labels,
            "label_confidence": labels.map(label_confidence),
            "is_ambiguous": labels.eq("ambiguous"),
        }
    )


def build_label_diagnostics(status_series: pd.Series) -> dict:
    raw_status = status_series.fillna("<<MISSING>>").astype("string")
    canonical = raw_status.apply(canonicalize_status)
    raw_counts = raw_status.value_counts(dropna=False).to_dict()
    canonical_counts = canonical.value_counts(dropna=False).to_dict()
    return {
        "raw_status_frequencies": {str(k): int(v) for k, v in raw_counts.items()},
        "canonical_label_counts": {str(k): int(v) for k, v in canonical_counts.items()},
        "unmapped_to_ambiguous_count": int((canonical == "ambiguous").sum()),
    }


def encode_target(series: pd.Series) -> Tuple[pd.Series, dict]:
    mapping = {label: index for index, label in enumerate(FINAL_LABELS)}
    return series.map(mapping), mapping
