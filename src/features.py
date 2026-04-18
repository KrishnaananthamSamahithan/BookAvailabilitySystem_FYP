from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureBundle:
    frame: pd.DataFrame
    feature_groups: Dict[str, List[str]]
    categorical_features: List[str]
    numeric_features: List[str]
    inference_reference: Dict[str, object]
    feature_availability: List[Dict[str, object]]


def build_feature_bundle(df: pd.DataFrame, include_labels: bool = True) -> FeatureBundle:
    frame = build_base_frame(df)

    if include_labels and "outcome_label" in frame.columns:
        frame = add_history_features(frame)
        if frame["price_total"].notna().any():
            frame = add_price_features(frame)
        else:
            frame = add_price_defaults(frame)
    else:
        frame = add_snapshot_defaults(frame)

    return _package_feature_bundle(frame)


def _package_feature_bundle(frame: pd.DataFrame) -> FeatureBundle:
    frame = finalize_price_features(frame.copy())

    categorical_features = [
        "trip_type",
        "cabin_class",
        "origin_airport",
        "destination_airport",
        "airline_code",
        "route",
        "airline_route",
        "provider_key",
        "provider_route",
        "provider_airline",
        "meta_engine",
        "referrer_domain",
        "market",
        "locale",
        "device_os",
        "price_currency",
        "days_to_departure_bucket",
    ]

    numeric_features = [
        "days_to_departure",
        "search_hour",
        "search_dayofweek",
        "search_month",
        "departure_month",
        "is_weekend_search",
        "passenger_count",
        "adults",
        "children",
        "infants",
        "stop_count",
        "is_multi_stop",
        "is_short_lead",
        "is_long_lead",
        "route_search_density",
        "provider_offer_density",
        "missing_price",
        "unknown_provider",
        "price_total",
        "route_day_min_price",
        "route_day_median_price",
        "price_gap_to_min",
        "price_ratio_to_median",
        "price_rank_in_route_day",
        "price_rank_pct_in_route_day",
        "offers_in_route_day",
        "provider_prior_count",
        "provider_prior_rate_bookable",
        "provider_prior_rate_price_changed",
        "provider_prior_rate_unavailable",
        "provider_prior_rate_technical_failure",
        "route_prior_count",
        "route_prior_rate_bookable",
        "route_prior_rate_price_changed",
        "route_prior_rate_unavailable",
        "provider_route_prior_rate_bookable",
        "provider_route_prior_rate_price_changed",
        "provider_route_prior_rate_unavailable",
        "airline_route_prior_rate_bookable",
        "airline_route_prior_rate_price_changed",
        "airline_route_prior_rate_unavailable",
        "provider_instability_score",
        "route_instability_score",
        "price_change_pressure",
        "technical_failure_pressure",
        "unavailability_pressure",
        "provider_minutes_since_prev",
        "route_minutes_since_prev",
        "provider_route_minutes_since_prev",
    ]

    for column in categorical_features:
        frame[column] = frame[column].astype("string").fillna("Unknown")
    for column in numeric_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    feature_groups = {
        "topology_basic": [
            "trip_type",
            "cabin_class",
            "origin_airport",
            "destination_airport",
            "airline_code",
            "route",
            "airline_route",
            "provider_key",
            "meta_engine",
            "referrer_domain",
            "market",
            "locale",
            "device_os",
            "price_currency",
            "provider_airline",
            "passenger_count",
            "stop_count",
            "is_multi_stop",
            "unknown_provider",
        ],
        "temporal": [
            "days_to_departure",
            "search_hour",
            "search_dayofweek",
            "search_month",
            "departure_month",
            "is_weekend_search",
            "days_to_departure_bucket",
            "is_short_lead",
            "is_long_lead",
            "route_search_density",
            "provider_offer_density",
        ],
        "price": [
            "price_total",
            "route_day_min_price",
            "route_day_median_price",
            "price_gap_to_min",
            "price_ratio_to_median",
            "price_rank_in_route_day",
            "price_rank_pct_in_route_day",
            "offers_in_route_day",
            "missing_price",
        ],
        "reliability_history": [
            "provider_prior_count",
            "provider_prior_rate_bookable",
            "provider_prior_rate_price_changed",
            "provider_prior_rate_unavailable",
            "provider_prior_rate_technical_failure",
            "route_prior_count",
            "route_prior_rate_bookable",
            "route_prior_rate_price_changed",
            "route_prior_rate_unavailable",
            "provider_route_prior_rate_bookable",
            "provider_route_prior_rate_price_changed",
            "provider_route_prior_rate_unavailable",
            "airline_route_prior_rate_bookable",
            "airline_route_prior_rate_price_changed",
            "airline_route_prior_rate_unavailable",
            "provider_instability_score",
            "route_instability_score",
            "price_change_pressure",
            "technical_failure_pressure",
            "unavailability_pressure",
            "provider_minutes_since_prev",
            "route_minutes_since_prev",
            "provider_route_minutes_since_prev",
        ],
    }
    feature_groups["full_valid_model"] = (
        feature_groups["topology_basic"]
        + feature_groups["temporal"]
        + feature_groups["price"]
        + feature_groups["reliability_history"]
    )

    inference_reference = build_inference_reference(frame)
    feature_availability = build_feature_availability(frame, feature_groups)
    return FeatureBundle(
        frame=frame,
        feature_groups=feature_groups,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        inference_reference=inference_reference,
        feature_availability=feature_availability,
    )


def build_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy().sort_values("prediction_time").reset_index(drop=True)
    frame["route"] = frame["origin_airport"] + "_" + frame["destination_airport"]
    frame["airline_route"] = frame["airline_code"] + "_" + frame["route"]
    frame["provider_route"] = frame["provider_key"] + "_" + frame["route"]
    frame["provider_airline"] = frame["provider_key"] + "_" + frame["airline_code"]
    frame["route_day"] = frame["route"] + "_" + frame["departure_date"].dt.strftime("%Y-%m-%d")
    frame["search_group_proxy"] = (
        frame["route"] + "_"
        + frame["trip_type"].astype("string") + "_"
        + frame["cabin_class"].astype("string") + "_"
        + frame["passenger_count"].astype("string") + "_"
        + frame["prediction_time"].dt.strftime("%Y-%m-%d-%H")
    )
    frame["search_hour"] = frame["prediction_time"].dt.hour
    frame["search_dayofweek"] = frame["prediction_time"].dt.dayofweek
    frame["search_month"] = frame["prediction_time"].dt.month
    frame["departure_month"] = frame["departure_date"].dt.month
    frame["is_weekend_search"] = frame["search_dayofweek"].isin([5, 6]).astype(int)
    frame["days_to_departure_bucket"] = pd.cut(
        frame["days_to_departure"],
        bins=[-1, 1, 3, 7, 14, 30, 90, 365, np.inf],
        labels=["same_day", "1_3_days", "3_7_days", "7_14_days", "14_30_days", "30_90_days", "90_365_days", "365_plus_days"],
    ).astype("string")
    frame["is_short_lead"] = frame["days_to_departure"].le(7).astype(int)
    frame["is_long_lead"] = frame["days_to_departure"].ge(60).astype(int)
    frame["missing_price"] = frame["price_total"].isna().astype(int)

    # FIX: compute density from prior counts (not hardcoded zero)
    # _count_per_time uses cumcount + elapsed days — forward-looking safe within training
    frame["route_search_density"] = _count_per_time(frame, "route")
    frame["provider_offer_density"] = _count_per_time(frame, "provider_key")

    return frame


def build_snapshot_feature_bundle(target_df: pd.DataFrame, history_df: pd.DataFrame) -> FeatureBundle:
    target_frame = build_base_frame(target_df)
    target_frame = apply_history_snapshot(target_frame, history_df)
    return _package_feature_bundle(target_frame)


def add_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    prior_group = frame.groupby("route_day", sort=False)["price_total"]

    frame["offers_in_route_day"] = prior_group.cumcount()
    frame["route_day_min_price"] = prior_group.transform(lambda s: s.shift(1).expanding().min())
    frame["route_day_median_price"] = prior_group.transform(lambda s: s.shift(1).expanding().median())

    global_median = float(frame["price_total"].median()) if frame["price_total"].notna().any() else 0.0
    frame["route_day_min_price"] = frame["route_day_min_price"].fillna(frame["price_total"]).fillna(global_median)
    frame["route_day_median_price"] = frame["route_day_median_price"].fillna(global_median)

    frame["price_gap_to_min"] = (frame["price_total"] - frame["route_day_min_price"]).clip(lower=0)
    denom = frame["route_day_median_price"].replace(0, np.nan)
    frame["price_ratio_to_median"] = (frame["price_total"] / denom).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    frame["price_rank_in_route_day"] = prior_group.transform(_prior_rank_against_current_value).fillna(0.0)
    rank_denom = frame["offers_in_route_day"].replace(0, np.nan)
    frame["price_rank_pct_in_route_day"] = (
        frame["price_rank_in_route_day"] / rank_denom
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def finalize_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["price_total"] = frame["price_total"].fillna(0.0)
    frame["route_day_min_price"] = frame["route_day_min_price"].fillna(frame["price_total"])
    frame["route_day_median_price"] = frame["route_day_median_price"].fillna(frame["price_total"].replace(0, np.nan)).fillna(0.0)
    frame["price_gap_to_min"] = (frame["price_total"] - frame["route_day_min_price"]).clip(lower=0)
    denom = frame["route_day_median_price"].replace(0, np.nan)
    frame["price_ratio_to_median"] = (frame["price_total"] / denom).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    frame["price_rank_in_route_day"] = frame["price_rank_in_route_day"].fillna(0.0)
    rank_denom = frame["offers_in_route_day"].replace(0, np.nan)
    frame["price_rank_pct_in_route_day"] = (
        frame["price_rank_in_route_day"] / rank_denom
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame["offers_in_route_day"] = frame["offers_in_route_day"].fillna(0.0)
    return frame


def add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["is_bookable"] = frame["outcome_label"].eq("bookable").astype(int)
    frame["is_price_changed"] = frame["outcome_label"].eq("price_changed").astype(int)
    frame["is_unavailable"] = frame["outcome_label"].eq("unavailable").astype(int)
    frame["is_technical_failure"] = frame["outcome_label"].eq("technical_failure").astype(int)

    frame["provider_prior_count"] = frame.groupby("provider_key").cumcount()
    frame["route_prior_count"] = frame.groupby("route").cumcount()

    frame["provider_prior_rate_bookable"] = _prior_rate(frame, "provider_key", "is_bookable")
    frame["provider_prior_rate_price_changed"] = _prior_rate(frame, "provider_key", "is_price_changed")
    frame["provider_prior_rate_unavailable"] = _prior_rate(frame, "provider_key", "is_unavailable")
    frame["provider_prior_rate_technical_failure"] = _prior_rate(frame, "provider_key", "is_technical_failure")
    frame["route_prior_rate_bookable"] = _prior_rate(frame, "route", "is_bookable")
    frame["route_prior_rate_price_changed"] = _prior_rate(frame, "route", "is_price_changed")
    frame["route_prior_rate_unavailable"] = _prior_rate(frame, "route", "is_unavailable")
    frame["provider_route_prior_rate_bookable"] = _prior_rate(frame, "provider_route", "is_bookable")
    frame["provider_route_prior_rate_price_changed"] = _prior_rate(frame, "provider_route", "is_price_changed")
    frame["provider_route_prior_rate_unavailable"] = _prior_rate(frame, "provider_route", "is_unavailable")
    frame["airline_route_prior_rate_bookable"] = _prior_rate(frame, "airline_route", "is_bookable")
    frame["airline_route_prior_rate_price_changed"] = _prior_rate(frame, "airline_route", "is_price_changed")
    frame["airline_route_prior_rate_unavailable"] = _prior_rate(frame, "airline_route", "is_unavailable")

    frame["provider_minutes_since_prev"] = _minutes_since_previous(frame, "provider_key")
    frame["route_minutes_since_prev"] = _minutes_since_previous(frame, "route")
    frame["provider_route_minutes_since_prev"] = _minutes_since_previous(frame, "provider_route")

    global_bookable = float(frame["is_bookable"].mean())
    global_price_changed = float(frame["is_price_changed"].mean())
    global_unavailable = float(frame["is_unavailable"].mean())
    global_technical_failure = float(frame["is_technical_failure"].mean())
    time_fill = {
        "provider_minutes_since_prev": 60.0,
        "route_minutes_since_prev": 60.0,
        "provider_route_minutes_since_prev": 60.0,
    }
    frame["provider_prior_rate_bookable"] = frame["provider_prior_rate_bookable"].fillna(global_bookable)
    frame["provider_prior_rate_price_changed"] = frame["provider_prior_rate_price_changed"].fillna(global_price_changed)
    frame["provider_prior_rate_unavailable"] = frame["provider_prior_rate_unavailable"].fillna(global_unavailable)
    frame["provider_prior_rate_technical_failure"] = frame["provider_prior_rate_technical_failure"].fillna(global_technical_failure)
    frame["route_prior_rate_bookable"] = frame["route_prior_rate_bookable"].fillna(global_bookable)
    frame["route_prior_rate_price_changed"] = frame["route_prior_rate_price_changed"].fillna(global_price_changed)
    frame["route_prior_rate_unavailable"] = frame["route_prior_rate_unavailable"].fillna(global_unavailable)
    frame["provider_route_prior_rate_bookable"] = frame["provider_route_prior_rate_bookable"].fillna(global_bookable)
    frame["provider_route_prior_rate_price_changed"] = frame["provider_route_prior_rate_price_changed"].fillna(global_price_changed)
    frame["provider_route_prior_rate_unavailable"] = frame["provider_route_prior_rate_unavailable"].fillna(global_unavailable)
    frame["airline_route_prior_rate_bookable"] = frame["airline_route_prior_rate_bookable"].fillna(global_bookable)
    frame["airline_route_prior_rate_price_changed"] = frame["airline_route_prior_rate_price_changed"].fillna(global_price_changed)
    frame["airline_route_prior_rate_unavailable"] = frame["airline_route_prior_rate_unavailable"].fillna(global_unavailable)
    for column, fill_value in time_fill.items():
        frame[column] = frame[column].fillna(fill_value)

    frame["provider_instability_score"] = 1.0 - frame["provider_prior_rate_bookable"]
    frame["route_instability_score"] = 1.0 - frame["route_prior_rate_bookable"]
    frame["price_change_pressure"] = (
        frame["provider_prior_rate_price_changed"]
        + frame["route_prior_rate_price_changed"]
        + frame["provider_route_prior_rate_price_changed"]
        + frame["airline_route_prior_rate_price_changed"]
    ) / 4.0
    frame["technical_failure_pressure"] = frame["provider_prior_rate_technical_failure"]
    frame["unavailability_pressure"] = (
        frame["provider_prior_rate_unavailable"]
        + frame["route_prior_rate_unavailable"]
        + frame["provider_route_prior_rate_unavailable"]
        + frame["airline_route_prior_rate_unavailable"]
    ) / 4.0

    return frame.drop(columns=["is_bookable", "is_price_changed", "is_unavailable", "is_technical_failure"])


def add_inference_history_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in [
        "provider_prior_count",
        "provider_prior_rate_bookable",
        "provider_prior_rate_price_changed",
        "provider_prior_rate_unavailable",
        "provider_prior_rate_technical_failure",
        "route_prior_count",
        "route_prior_rate_bookable",
        "route_prior_rate_price_changed",
        "route_prior_rate_unavailable",
        "provider_route_prior_rate_bookable",
        "provider_route_prior_rate_price_changed",
        "provider_route_prior_rate_unavailable",
        "airline_route_prior_rate_bookable",
        "airline_route_prior_rate_price_changed",
        "airline_route_prior_rate_unavailable",
        "provider_instability_score",
        "route_instability_score",
        "price_change_pressure",
        "technical_failure_pressure",
        "unavailability_pressure",
        "provider_minutes_since_prev",
        "route_minutes_since_prev",
        "provider_route_minutes_since_prev",
    ]:
        frame[column] = 0.0
    return frame


def add_snapshot_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in [
        "provider_prior_count",
        "provider_prior_rate_bookable",
        "provider_prior_rate_price_changed",
        "provider_prior_rate_unavailable",
        "provider_prior_rate_technical_failure",
        "route_prior_count",
        "route_prior_rate_bookable",
        "route_prior_rate_price_changed",
        "route_prior_rate_unavailable",
        "provider_route_prior_rate_bookable",
        "provider_route_prior_rate_price_changed",
        "provider_route_prior_rate_unavailable",
        "airline_route_prior_rate_bookable",
        "airline_route_prior_rate_price_changed",
        "airline_route_prior_rate_unavailable",
        "provider_instability_score",
        "route_instability_score",
        "price_change_pressure",
        "technical_failure_pressure",
        "unavailability_pressure",
        "provider_minutes_since_prev",
        "route_minutes_since_prev",
        "provider_route_minutes_since_prev",
    ]:
        frame[column] = 0.0
    return add_price_defaults(frame)


def add_price_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in [
        "route_day_min_price",
        "route_day_median_price",
        "price_gap_to_min",
        "price_ratio_to_median",
        "price_rank_in_route_day",
        "offers_in_route_day",
    ]:
        frame[column] = 0.0
    return finalize_price_features(frame)


def apply_history_snapshot(target_frame: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    history = build_base_frame(history_df)
    history = history.sort_values("prediction_time").reset_index(drop=True)
    history["provider_prior_count"] = history.groupby("provider_key").cumcount()
    history["route_prior_count"] = history.groupby("route").cumcount()

    for label_name in ["bookable", "price_changed", "unavailable", "technical_failure"]:
        history[f"is_{label_name}"] = history["outcome_label"].eq(label_name).astype(int)

    snapshot = {
        "provider_counts": history.groupby("provider_key").size().to_dict(),
        "route_counts": history.groupby("route").size().to_dict(),
        "provider_rate_bookable": history.groupby("provider_key")["is_bookable"].mean().to_dict(),
        "provider_rate_price_changed": history.groupby("provider_key")["is_price_changed"].mean().to_dict(),
        "provider_rate_unavailable": history.groupby("provider_key")["is_unavailable"].mean().to_dict(),
        "provider_rate_technical_failure": history.groupby("provider_key")["is_technical_failure"].mean().to_dict(),
        "route_rate_bookable": history.groupby("route")["is_bookable"].mean().to_dict(),
        "route_rate_price_changed": history.groupby("route")["is_price_changed"].mean().to_dict(),
        "route_rate_unavailable": history.groupby("route")["is_unavailable"].mean().to_dict(),
        "provider_route_rate_bookable": history.groupby("provider_route")["is_bookable"].mean().to_dict(),
        "provider_route_rate_price_changed": history.groupby("provider_route")["is_price_changed"].mean().to_dict(),
        "provider_route_rate_unavailable": history.groupby("provider_route")["is_unavailable"].mean().to_dict(),
        "airline_route_rate_bookable": history.groupby("airline_route")["is_bookable"].mean().to_dict(),
        "airline_route_rate_price_changed": history.groupby("airline_route")["is_price_changed"].mean().to_dict(),
        "airline_route_rate_unavailable": history.groupby("airline_route")["is_unavailable"].mean().to_dict(),
        "provider_last_time": history.groupby("provider_key")["prediction_time"].max().to_dict(),
        "route_last_time": history.groupby("route")["prediction_time"].max().to_dict(),
        "provider_route_last_time": history.groupby("provider_route")["prediction_time"].max().to_dict(),
        "provider_first_time": history.groupby("provider_key")["prediction_time"].min().to_dict(),
        "route_first_time": history.groupby("route")["prediction_time"].min().to_dict(),
        "route_day_min_price": history.groupby("route_day")["price_total"].min().to_dict() if history["price_total"].notna().any() else {},
        "route_day_median_price": history.groupby("route_day")["price_total"].median().to_dict() if history["price_total"].notna().any() else {},
        "route_day_offer_count": history.groupby("route_day").size().to_dict(),
        "route_day_price_values": (
            history.groupby("route_day")["price_total"].apply(lambda s: s.dropna().tolist()).to_dict()
            if history["price_total"].notna().any()
            else {}
        ),
        "global_rates": {
            "bookable": float(history["is_bookable"].mean()) if len(history) else 0.0,
            "price_changed": float(history["is_price_changed"].mean()) if len(history) else 0.0,
            "unavailable": float(history["is_unavailable"].mean()) if len(history) else 0.0,
            "technical_failure": float(history["is_technical_failure"].mean()) if len(history) else 0.0,
        },
    }

    target = target_frame.copy()
    target["provider_prior_count"] = target["provider_key"].map(snapshot["provider_counts"]).fillna(0.0)
    target["route_prior_count"] = target["route"].map(snapshot["route_counts"]).fillna(0.0)
    target["provider_prior_rate_bookable"] = target["provider_key"].map(snapshot["provider_rate_bookable"]).fillna(snapshot["global_rates"]["bookable"])
    target["provider_prior_rate_price_changed"] = target["provider_key"].map(snapshot["provider_rate_price_changed"]).fillna(snapshot["global_rates"]["price_changed"])
    target["provider_prior_rate_unavailable"] = target["provider_key"].map(snapshot["provider_rate_unavailable"]).fillna(snapshot["global_rates"]["unavailable"])
    target["provider_prior_rate_technical_failure"] = target["provider_key"].map(snapshot["provider_rate_technical_failure"]).fillna(snapshot["global_rates"]["technical_failure"])
    target["route_prior_rate_bookable"] = target["route"].map(snapshot["route_rate_bookable"]).fillna(snapshot["global_rates"]["bookable"])
    target["route_prior_rate_price_changed"] = target["route"].map(snapshot["route_rate_price_changed"]).fillna(snapshot["global_rates"]["price_changed"])
    target["route_prior_rate_unavailable"] = target["route"].map(snapshot["route_rate_unavailable"]).fillna(snapshot["global_rates"]["unavailable"])
    target["provider_route_prior_rate_bookable"] = target["provider_route"].map(snapshot["provider_route_rate_bookable"]).fillna(snapshot["global_rates"]["bookable"])
    target["provider_route_prior_rate_price_changed"] = target["provider_route"].map(snapshot["provider_route_rate_price_changed"]).fillna(snapshot["global_rates"]["price_changed"])
    target["provider_route_prior_rate_unavailable"] = target["provider_route"].map(snapshot["provider_route_rate_unavailable"]).fillna(snapshot["global_rates"]["unavailable"])
    target["airline_route_prior_rate_bookable"] = target["airline_route"].map(snapshot["airline_route_rate_bookable"]).fillna(snapshot["global_rates"]["bookable"])
    target["airline_route_prior_rate_price_changed"] = target["airline_route"].map(snapshot["airline_route_rate_price_changed"]).fillna(snapshot["global_rates"]["price_changed"])
    target["airline_route_prior_rate_unavailable"] = target["airline_route"].map(snapshot["airline_route_rate_unavailable"]).fillna(snapshot["global_rates"]["unavailable"])
    target["provider_minutes_since_prev"] = _minutes_since_snapshot(target, "provider_key", snapshot["provider_last_time"])
    target["route_minutes_since_prev"] = _minutes_since_snapshot(target, "route", snapshot["route_last_time"])
    target["provider_route_minutes_since_prev"] = _minutes_since_snapshot(target, "provider_route", snapshot["provider_route_last_time"])
    target["route_search_density"] = _density_from_snapshot(target, "route", snapshot["route_counts"], snapshot["route_first_time"])
    target["provider_offer_density"] = _density_from_snapshot(target, "provider_key", snapshot["provider_counts"], snapshot["provider_first_time"])
    target["route_day_min_price"] = target["route_day"].map(snapshot["route_day_min_price"]).fillna(target["price_total"])
    target["route_day_median_price"] = target["route_day"].map(snapshot["route_day_median_price"]).fillna(target["price_total"])
    target["offers_in_route_day"] = target["route_day"].map(snapshot["route_day_offer_count"]).fillna(0.0)
    target["price_rank_in_route_day"] = _snapshot_price_rank(
        target["route_day"],
        target["price_total"],
        snapshot["route_day_price_values"],
    )
    target["provider_instability_score"] = 1.0 - target["provider_prior_rate_bookable"]
    target["route_instability_score"] = 1.0 - target["route_prior_rate_bookable"]
    target["price_change_pressure"] = (
        target["provider_prior_rate_price_changed"]
        + target["route_prior_rate_price_changed"]
        + target["provider_route_prior_rate_price_changed"]
        + target["airline_route_prior_rate_price_changed"]
    ) / 4.0
    target["technical_failure_pressure"] = target["provider_prior_rate_technical_failure"]
    target["unavailability_pressure"] = (
        target["provider_prior_rate_unavailable"]
        + target["route_prior_rate_unavailable"]
        + target["provider_route_prior_rate_unavailable"]
        + target["airline_route_prior_rate_unavailable"]
    ) / 4.0
    return finalize_price_features(target)


def build_inference_reference(frame: pd.DataFrame) -> Dict[str, object]:
    history_columns = [
        "provider_prior_count",
        "provider_prior_rate_bookable",
        "provider_prior_rate_price_changed",
        "provider_prior_rate_unavailable",
        "provider_prior_rate_technical_failure",
        "route_prior_count",
        "route_prior_rate_bookable",
        "route_prior_rate_price_changed",
        "route_prior_rate_unavailable",
        "provider_route_prior_rate_bookable",
        "provider_route_prior_rate_price_changed",
        "provider_route_prior_rate_unavailable",
        "airline_route_prior_rate_bookable",
        "airline_route_prior_rate_price_changed",
        "airline_route_prior_rate_unavailable",
        "provider_instability_score",
        "route_instability_score",
        "price_change_pressure",
        "technical_failure_pressure",
        "unavailability_pressure",
        "provider_minutes_since_prev",
        "route_minutes_since_prev",
        "provider_route_minutes_since_prev",
    ]
    defaults = {column: float(frame[column].median()) for column in history_columns}
    return {
        "history_defaults": defaults,
        "global_price_median": float(frame["price_total"].median()) if "price_total" in frame else 0.0,
        "inference_mode_note": (
            "History-dependent features use saved batch defaults during offline inference. "
            "This is an offline approximation, not a live online feature store."
        ),
    }


def build_feature_availability(frame: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> List[Dict[str, object]]:
    family_by_feature: Dict[str, str] = {}
    for family_name, columns in feature_groups.items():
        if family_name == "full_valid_model":
            continue
        for column in columns:
            family_by_feature[column] = family_name

    history_features = set(feature_groups["reliability_history"])
    final_model_features = set(feature_groups["full_valid_model"])
    rows = []
    for feature_name in sorted(final_model_features):
        rows.append(
            {
                "feature_name": feature_name,
                "feature_family": family_by_feature.get(feature_name, "unknown"),
                "available_at_prediction_time": True,
                "derived_from_prior_only_history": feature_name in history_features,
                "used_in_final_model": True,
                "non_null_fraction": float(frame[feature_name].notna().mean()) if feature_name in frame else 0.0,
            }
        )
    return rows


def run_feature_selection(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_names: List[str],
    categorical_features: List[str],
    k: int = 35,
) -> Dict[str, object]:
    """Mutual information based feature selection. Works on mixed-type features."""
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import OrdinalEncoder

    X_encoded = X_train.copy()
    cat_cols = [c for c in categorical_features if c in X_encoded.columns]
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_encoded[cat_cols] = enc.fit_transform(X_encoded[cat_cols].astype(str))

    mi_scores = mutual_info_classif(
        X_encoded[feature_names].fillna(0),
        y_train,
        discrete_features=[f in cat_cols for f in feature_names],
        random_state=42,
    )

    scored = sorted(zip(feature_names, mi_scores.tolist()), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in scored[:k]]
    return {
        "method": "mutual_information",
        "k_selected": k,
        "all_scores": {f: float(s) for f, s in scored},
        "selected_features": selected,
        "top_10": [{"feature": f, "mi_score": float(s)} for f, s in scored[:10]],
    }


def _prior_rate(frame: pd.DataFrame, key: str, indicator: str) -> pd.Series:
    group = frame.groupby(key)[indicator]
    cumulative = group.cumsum() - frame[indicator]
    prior_count = frame.groupby(key).cumcount().replace(0, np.nan)
    return cumulative / prior_count


def _prior_rank_against_current_value(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    output = np.zeros(len(values), dtype=float)
    history: List[float] = []
    for idx, current_value in enumerate(values):
        if np.isnan(current_value) or not history:
            output[idx] = 0.0
        else:
            output[idx] = float(np.sum(np.asarray(history) <= current_value))
        if not np.isnan(current_value):
            history.append(float(current_value))
    return pd.Series(output, index=series.index)


def _minutes_since_previous(frame: pd.DataFrame, key: str) -> pd.Series:
    previous = frame.groupby(key)["prediction_time"].shift(1)
    return (frame["prediction_time"] - previous).dt.total_seconds() / 60.0


def _count_per_time(frame: pd.DataFrame, key: str) -> pd.Series:
    """Compute prior-safe events-per-day density for each row."""
    cumcount = frame.groupby(key).cumcount()  # 0-based count of prior rows
    first_time = frame.groupby(key)["prediction_time"].transform("min")
    elapsed_days = (frame["prediction_time"] - first_time).dt.total_seconds() / 86400.0
    return (cumcount + 1) / elapsed_days.clip(lower=1.0)


def _minutes_since_snapshot(target: pd.DataFrame, key: str, last_time_map: Dict[str, pd.Timestamp]) -> pd.Series:
    previous_time = target[key].map(last_time_map)
    minutes = (target["prediction_time"] - pd.to_datetime(previous_time)).dt.total_seconds() / 60.0
    return minutes.fillna(60.0).clip(lower=0.0)


def _density_from_snapshot(target: pd.DataFrame, key: str, count_map: Dict[str, float], first_time_map: Dict[str, pd.Timestamp]) -> pd.Series:
    counts = target[key].map(count_map).fillna(0.0) + 1.0
    first_times = pd.to_datetime(target[key].map(first_time_map))
    elapsed_days = (target["prediction_time"] - first_times).dt.total_seconds() / 86400.0
    return counts / elapsed_days.fillna(1.0).clip(lower=1.0)


def _snapshot_price_rank(
    route_day_series: pd.Series,
    price_series: pd.Series,
    route_day_price_values: Dict[str, List[float]],
) -> pd.Series:
    ranks = []
    for route_day, current_price in zip(route_day_series.astype("string"), pd.to_numeric(price_series, errors="coerce")):
        history_prices = route_day_price_values.get(str(route_day), [])
        if np.isnan(current_price) or not history_prices:
            ranks.append(0.0)
            continue
        history_array = np.asarray(history_prices, dtype=float)
        history_array = history_array[~np.isnan(history_array)]
        if len(history_array) == 0:
            ranks.append(0.0)
            continue
        ranks.append(float(np.sum(history_array <= current_price)))
    return pd.Series(ranks, index=route_day_series.index, dtype=float)
