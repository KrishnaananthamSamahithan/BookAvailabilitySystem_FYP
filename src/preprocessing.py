from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.labels import build_label_diagnostics, build_label_frame
from src.reporting import build_schema_resolution_report
from src.schema import SchemaResolution, resolve_schema
from src.utils import (
    extract_domain,
    first_present,
    normalize_text,
    parse_query_params,
    safe_to_datetime,
    safe_to_numeric,
    save_csv,
    save_json,
)


@dataclass
class PreprocessingResult:
    frame: pd.DataFrame
    audit: Dict[str, object]
    schema: SchemaResolution


def preprocess_raw_file(config: ProjectConfig) -> PreprocessingResult:
    config.ensure_directories()
    raw = pd.read_csv(config.raw_data_path)
    result = preprocess_raw_frame(raw, config)
    save_csv(config.processed_data_path, result.frame)
    save_json(config.reports_dir / "preprocessing_audit.json", result.audit)
    save_json(
        config.reports_dir / "schema_resolution.json",
        {
            "rows": build_schema_resolution_report(result.schema),
            "missing_required": result.schema.missing_required,
            "prediction_time_assumption": config.prediction_time_assumption,
            "selected_prediction_time_raw_column": result.schema.required["prediction_time"],
            "selected_prediction_time_alias": result.schema.matched_aliases.get("prediction_time"),
        },
    )
    save_json(config.reports_dir / "label_diagnostics.json", result.audit["label_diagnostics"])
    save_json(
        config.reports_dir / "prediction_time_metadata.json",
        {
            "selected_raw_column": result.schema.required["prediction_time"],
            "matched_alias": result.schema.matched_aliases.get("prediction_time"),
            "assumption": config.prediction_time_assumption,
        },
    )
    return result


def preprocess_raw_frame(raw: pd.DataFrame, config: ProjectConfig) -> PreprocessingResult:
    schema = resolve_schema(
        list(raw.columns),
        config.column_aliases,
        config.required_logical_fields,
        config.optional_logical_fields,
    )
    if not schema.is_valid:
        raise ValueError(f"Missing required logical fields: {schema.missing_required}")

    audit = {
        "rows_initial": int(len(raw)),
        "rows_bad_timestamp": 0,
        "rows_bad_depdate": 0,
        "rows_negative_days_to_departure": 0,
        "rows_missing_required_postmap": 0,
        "rows_duplicates_removed": 0,
        "rows_unmapped_labels": 0,
        "rows_final": 0,
        "prediction_time_assumption": config.prediction_time_assumption,
    }
    audit["label_diagnostics"] = build_label_diagnostics(raw[schema.required["status"]])

    frame = pd.DataFrame()
    frame["prediction_time"] = safe_to_datetime(raw[schema.required["prediction_time"]])
    frame["departure_date"] = safe_to_datetime(raw[schema.required["departure_date"]])
    frame["origin_airport"] = normalize_text(raw[schema.required["origin"]]).str.upper()
    frame["destination_airport"] = normalize_text(raw[schema.required["destination"]]).str.upper()
    frame["airline_code"] = normalize_text(raw[schema.required["airline"]]).str.upper()
    frame["cabin_class"] = normalize_text(raw[schema.required["cabin"]]).str.upper()

    trip_col = _optional_series(raw, schema, "trip_type")
    frame["trip_type"] = normalize_text(trip_col if trip_col is not None else pd.Series(["Unknown"] * len(raw)))

    previous_page = _optional_series(raw, schema, "previous_page")
    landing_page = _optional_series(raw, schema, "landing_page")
    user_agent = _optional_series(raw, schema, "user_agent")
    inserted_on = _optional_series(raw, schema, "inserted_on")
    fdtag = _optional_series(raw, schema, "fdtag")
    flight_segments = _optional_series(raw, schema, "flight_segments")
    provider_id = _optional_series(raw, schema, "provider_id")
    price = _optional_series(raw, schema, "price")
    market = _optional_series(raw, schema, "market")
    locale = _optional_series(raw, schema, "locale")
    device = _optional_series(raw, schema, "device")
    adults = _optional_series(raw, schema, "adults")
    children = _optional_series(raw, schema, "children")
    infants = _optional_series(raw, schema, "infants")
    stops = _optional_series(raw, schema, "stops")

    frame["outcome_time"] = safe_to_datetime(inserted_on) if inserted_on is not None else pd.NaT
    frame["referrer_domain"] = (previous_page if previous_page is not None else pd.Series([None] * len(raw))).apply(extract_domain)
    frame["landing_domain"] = (landing_page if landing_page is not None else pd.Series([None] * len(raw))).apply(extract_domain)
    frame["meta_engine"] = frame["referrer_domain"].map(map_meta_engine).fillna("Direct")

    provider_key = provider_id.astype("string") if provider_id is not None else None
    if provider_key is None or provider_key.isna().all():
        provider_key = frame["landing_domain"].replace({"Unknown": pd.NA}).fillna(frame["meta_engine"])
    frame["provider_key"] = normalize_text(provider_key)

    frame["market"] = normalize_text(market if market is not None else pd.Series(["Unknown"] * len(raw)))
    frame["locale"] = normalize_text(locale if locale is not None else _extract_locale_series(user_agent, len(raw)))
    frame["device_os"] = normalize_text(device if device is not None else _extract_device_os_series(user_agent, len(raw)))

    passenger_frame = _build_passenger_frame(landing_page, adults, children, infants, len(raw))
    frame = pd.concat([frame, passenger_frame], axis=1)

    frame["stop_count"] = _build_stop_count(stops, fdtag, flight_segments, len(raw))
    frame["price_total"] = safe_to_numeric(price) if price is not None else np.nan

    label_frame = build_label_frame(raw[schema.required["status"]])
    frame = pd.concat([frame, label_frame], axis=1)

    audit["rows_bad_timestamp"] = int(frame["prediction_time"].isna().sum())
    audit["rows_bad_depdate"] = int(frame["departure_date"].isna().sum())
    audit["rows_unmapped_labels"] = int(frame["outcome_label"].eq("ambiguous").sum())

    frame = frame.dropna(subset=["prediction_time", "departure_date"]).copy()

    frame["days_to_departure"] = (frame["departure_date"] - frame["prediction_time"]).dt.days
    negative_mask = frame["days_to_departure"] < 0
    audit["rows_negative_days_to_departure"] = int(negative_mask.sum())
    frame = frame.loc[~negative_mask].copy()

    frame["prediction_time_rounded"] = frame["prediction_time"].dt.floor("min")
    frame["route"] = frame["origin_airport"] + "_" + frame["destination_airport"]

    required_postmap = [
        "prediction_time",
        "departure_date",
        "origin_airport",
        "destination_airport",
        "airline_code",
        "cabin_class",
        "outcome_label",
    ]
    missing_required_mask = frame[required_postmap].isna().any(axis=1)
    audit["rows_missing_required_postmap"] = int(missing_required_mask.sum())
    frame = frame.loc[~missing_required_mask].copy()

    dedup_key = [
        "prediction_time_rounded",
        "provider_key",
        "route",
        "airline_code",
        "cabin_class",
        "price_total",
    ]
    before_dedup = len(frame)
    frame = frame.sort_values(["prediction_time", "departure_date"]).drop_duplicates(subset=dedup_key, keep="first")
    audit["rows_duplicates_removed"] = int(before_dedup - len(frame))

    frame["unknown_provider"] = frame["provider_key"].eq("Unknown").astype(int)
    frame["missing_price"] = frame["price_total"].isna().astype(int)
    frame["stop_count"] = frame["stop_count"].fillna(0).clip(lower=0).astype(int)
    frame["is_multi_stop"] = frame["stop_count"].ge(2).astype(int)

    frame = frame.sort_values("prediction_time").reset_index(drop=True)
    audit["rows_final"] = int(len(frame))
    return PreprocessingResult(frame=frame, audit=audit, schema=schema)


def map_meta_engine(domain: str) -> str:
    text = (domain or "").lower()
    if "skyscanner" in text:
        return "Skyscanner"
    if "google" in text:
        return "Google"
    if "kayak" in text:
        return "Kayak"
    if "momondo" in text:
        return "Momondo"
    if "tripadvisor" in text:
        return "TripAdvisor"
    if text in {"", "unknown"}:
        return "Direct"
    return "Other"


def _optional_series(raw: pd.DataFrame, schema: SchemaResolution, logical_name: str) -> Optional[pd.Series]:
    column = schema.optional.get(logical_name)
    return raw[column] if column is not None else None


def _extract_device_os_series(user_agent: Optional[pd.Series], length: int) -> pd.Series:
    if user_agent is None:
        return pd.Series(["Unknown"] * length)

    def parse(agent: object) -> str:
        text = str(agent).lower()
        if "android" in text:
            return "Android"
        if "iphone" in text or "ipad" in text or "ios" in text:
            return "iOS"
        if "windows" in text:
            return "Windows"
        if "mac os" in text or "macos" in text:
            return "MacOS"
        if "linux" in text:
            return "Linux"
        return "Other"

    return user_agent.apply(parse)


def _extract_locale_series(user_agent: Optional[pd.Series], length: int) -> pd.Series:
    if user_agent is None:
        return pd.Series(["Unknown"] * length)

    def parse(agent: object) -> str:
        text = str(agent)
        marker = "Accept-Language="
        if marker not in text:
            return "Unknown"
        suffix = text.split(marker, 1)[1]
        locale = suffix.split("&", 1)[0]
        return locale.replace("%2c", ",").replace("%3b", ";")[:32] or "Unknown"

    return user_agent.apply(parse)


def _build_passenger_frame(
    landing_page: Optional[pd.Series],
    adults: Optional[pd.Series],
    children: Optional[pd.Series],
    infants: Optional[pd.Series],
    length: int,
) -> pd.DataFrame:
    if landing_page is not None:
        params = landing_page.apply(parse_query_params)
        adult_values = params.apply(lambda q: first_present(q, ["totaladults", "adults"], "1"))
        child_values = params.apply(lambda q: first_present(q, ["totalchilds", "children"], "0"))
        infant_values = params.apply(lambda q: first_present(q, ["totalinfants", "infants"], "0"))
    else:
        adult_values = adults if adults is not None else pd.Series(["1"] * length)
        child_values = children if children is not None else pd.Series(["0"] * length)
        infant_values = infants if infants is not None else pd.Series(["0"] * length)

    adults_num = safe_to_numeric(adult_values).fillna(1).clip(lower=0)
    children_num = safe_to_numeric(child_values).fillna(0).clip(lower=0)
    infants_num = safe_to_numeric(infant_values).fillna(0).clip(lower=0)

    return pd.DataFrame(
        {
            "adults": adults_num.astype(int),
            "children": children_num.astype(int),
            "infants": infants_num.astype(int),
            "passenger_count": (adults_num + children_num + infants_num).astype(int),
        }
    )


def _build_stop_count(
    stops: Optional[pd.Series],
    fdtag: Optional[pd.Series],
    flight_segments: Optional[pd.Series],
    length: int,
) -> pd.Series:
    if stops is not None:
        numeric = safe_to_numeric(stops)
        if numeric.notna().any():
            return numeric.fillna(0)

    if fdtag is not None:
        segments = fdtag.astype("string").fillna("").str.count("~") + 1
    elif flight_segments is not None:
        segments = flight_segments.astype("string").fillna("").str.count(",") + 1
    else:
        segments = pd.Series([1] * length)

    return (segments - 1).clip(lower=0)
