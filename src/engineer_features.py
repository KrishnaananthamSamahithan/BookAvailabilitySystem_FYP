import os
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd

from src.labels import canonicalize_status

RAW_INPUT_FILE = "data/raw/tbl_SearchTracking_Merged.csv"
OUTPUT_FILE = "data/processed/training_data.csv"


def extract_device_os(agent):
    if pd.isna(agent):
        return "Unknown"
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


def get_meta_engine(url):
    if pd.isna(url):
        return "Direct"
    try:
        domain = urlparse(str(url)).netloc.lower()
    except Exception:
        return "Other"
    if "skyscanner" in domain:
        return "Skyscanner"
    if "google" in domain:
        return "Google"
    if "kayak" in domain:
        return "Kayak"
    if "tripadvisor" in domain:
        return "TripAdvisor"
    if "momondo" in domain:
        return "Momondo"
    if "wego" in domain:
        return "Wego"
    if "carltonleisure" in domain or domain == "":
        return "Direct"
    return "Other"


def _parse_query_value(url, key):
    if pd.isna(url):
        return None
    try:
        parsed = parse_qs(urlparse(str(url)).query, keep_blank_values=True)
    except Exception:
        return None
    values = parsed.get(key)
    if not values:
        return None
    value = values[0]
    return value if value != "" else None


def _extract_domain(url):
    if pd.isna(url):
        return "Unknown"
    try:
        domain = urlparse(str(url)).netloc.lower()
        return domain or "Unknown"
    except Exception:
        return "Unknown"


def _simplify_referrer(url):
    domain = _extract_domain(url)
    if "google" in domain:
        return "google"
    if "skyscanner" in domain:
        return "skyscanner"
    if "wego" in domain:
        return "wego"
    if "kayak" in domain:
        return "kayak"
    if "tripadvisor" in domain:
        return "tripadvisor"
    if domain in {"unknown", "www.carltonleisure.com", "carltonleisure.com"}:
        return "direct_internal"
    return "other"


def _count_fdtag_legs(fdtag):
    if pd.isna(fdtag):
        return 1
    return str(fdtag).count("~") + 1


def _count_fdtag_carriers(fdtag):
    if pd.isna(fdtag):
        return 0
    carriers = {
        token.split("/")[0]
        for token in str(fdtag).split("~")
        if "/" in token and token.split("/")[0]
    }
    return len(carriers)


def _fdtag_codeshare_flag(fdtag):
    return int(_count_fdtag_carriers(fdtag) > 1)


def _flightsegment_leg_count(value):
    if pd.isna(value):
        return 0
    return len([segment for segment in str(value).split(",") if segment.strip()])


def _flightsegment_openjaw_flag(value):
    if pd.isna(value):
        return 0
    segments = [segment.strip() for segment in str(value).split(",") if segment.strip()]
    if len(segments) < 2:
        return 0
    first = segments[0].split("-")
    last = segments[-1].split("-")
    if len(first) != 2 or len(last) != 2:
        return 0
    return int(first[0] != last[1])


def _extract_accept_language(agent):
    if pd.isna(agent):
        return "Unknown"
    text = str(agent)
    marker = "Accept-Language="
    if marker not in text:
        return "Unknown"
    suffix = text.split(marker, 1)[1].split("&", 1)[0]
    return unquote(suffix).split(",")[0].split(";")[0][:32] or "Unknown"


def _extract_cf_ipcountry(agent):
    if pd.isna(agent):
        return "Unknown"
    text = str(agent)
    marker = "cf-ipcountry="
    if marker not in text:
        return "Unknown"
    return text.split(marker, 1)[1].split("&", 1)[0][:8] or "Unknown"


def _extract_browser_name(agent):
    if pd.isna(agent):
        return "Unknown"
    text = str(agent)
    marker = "Name = "
    if marker in text:
        return text.split(marker, 1)[1].split("~", 1)[0][:32] or "Unknown"
    lowered = text.lower()
    if "chrome" in lowered:
        return "Chrome"
    if "safari" in lowered:
        return "Safari"
    if "firefox" in lowered:
        return "Firefox"
    return "Unknown"


def _extract_browser_major(agent):
    if pd.isna(agent):
        return "Unknown"
    text = str(agent)
    marker = "Major Version = "
    if marker in text:
        return text.split(marker, 1)[1].split("~", 1)[0][:8] or "Unknown"
    return "Unknown"


def _extract_platform(agent):
    if pd.isna(agent):
        return "Unknown"
    text = str(agent)
    marker = "Platform = "
    if marker in text:
        return text.split(marker, 1)[1].split("~", 1)[0][:32] or "Unknown"
    return "Unknown"


def build_base_training_frame(df):
    frame = pd.DataFrame()
    frame["prediction_time"] = pd.to_datetime(df["LandingTime"], format="mixed", errors="coerce")
    frame["inserted_on"] = pd.to_datetime(df["InsertedOn"], format="mixed", errors="coerce")
    frame["departure_date"] = pd.to_datetime(df["DepDate"], format="mixed", errors="coerce")
    frame["arrival_date"] = pd.to_datetime(df["ArrDate"], format="mixed", errors="coerce")

    frame["trip_type"] = df["SearchType"].fillna("Unknown").astype(str)
    frame["origin_airport"] = df["Origin"].fillna("Unknown").astype(str).str.upper()
    frame["destination_airport"] = df["Destination"].fillna("Unknown").astype(str).str.upper()
    frame["airline_code"] = df["Airline"].fillna("Unknown").astype(str).str.upper()
    frame["cabin_class"] = df["Class"].fillna("Unknown").astype(str).str.upper()
    frame["meta_engine"] = df["PreviousPage"].apply(get_meta_engine)

    frame["status"] = df["Status"].astype("string")
    frame["canonical_status"] = frame["status"].apply(canonicalize_status)
    frame["outcome_label"] = frame["canonical_status"]
    frame["label_confidence"] = frame["outcome_label"].isin(
        ["bookable", "price_changed", "unavailable", "technical_failure"]
    ).astype(float).replace({0.0: 0.5, 1.0: 1.0})

    frame["route"] = frame["origin_airport"] + "_" + frame["destination_airport"]
    frame["days_to_departure"] = (frame["departure_date"] - frame["prediction_time"]).dt.days
    frame["search_hour"] = frame["prediction_time"].dt.hour
    frame["search_day"] = frame["prediction_time"].dt.dayofweek
    frame["dep_month"] = frame["departure_date"].dt.month
    frame["is_weekend"] = frame["search_day"].isin([5, 6]).astype(int)
    frame["device_os"] = df["UserAgent"].apply(extract_device_os)
    frame["itinerary_segments"] = df["FdTag"].apply(_count_fdtag_legs)

    # Real cache age from event timestamps, not simulated values.
    frame["cache_age_hours"] = (
        (frame["inserted_on"] - frame["prediction_time"]).dt.total_seconds() / 3600.0
    )

    frame["trip_duration_days"] = (frame["arrival_date"] - frame["departure_date"]).dt.days
    frame["trip_duration_bucket"] = pd.cut(
        frame["trip_duration_days"],
        bins=[-1, 0, 3, 7, 14, 30, 60, 365],
        labels=["same_day", "1_3", "4_7", "8_14", "15_30", "31_60", "61_plus"],
    ).astype(str)

    frame["direct_flights_requested"] = df["LandingPage"].map(
        lambda url: _parse_query_value(url, "DirectFlights") or "Unknown"
    )
    frame["date_flexible_requested"] = df["LandingPage"].map(
        lambda url: _parse_query_value(url, "DateFlexible") or "Unknown"
    )
    frame["landing_ref_code"] = df["LandingPage"].map(
        lambda url: _parse_query_value(url, "ref") or "Unknown"
    )
    frame["has_gd_param"] = df["LandingPage"].map(lambda url: int(_parse_query_value(url, "gd") is not None))
    frame["has_lid_param"] = df["LandingPage"].map(lambda url: int(_parse_query_value(url, "lid") is not None))
    frame["has_skyscanner_redirectid"] = df["LandingPage"].map(
        lambda url: int(_parse_query_value(url, "skyscanner_redirectid") is not None)
    )
    frame["skyscanner_id_present"] = df["SkyscannerID"].notna().astype(int)

    frame["previous_page_domain"] = df["PreviousPage"].apply(_extract_domain)
    frame["previous_page_group"] = df["PreviousPage"].apply(_simplify_referrer)
    previous_page_lower = df["PreviousPage"].fillna("").astype(str).str.lower()
    frame["previous_page_has_google"] = previous_page_lower.str.contains("google").astype(int)
    frame["previous_page_has_skyscanner"] = previous_page_lower.str.contains("skyscanner").astype(int)
    frame["previous_page_has_gclid"] = previous_page_lower.str.contains("gclid=").astype(int)

    frame["cf_ipcountry"] = df["UserAgent"].apply(_extract_cf_ipcountry)
    frame["accept_language_primary"] = df["UserAgent"].apply(_extract_accept_language)
    frame["browser_name"] = df["UserAgent"].apply(_extract_browser_name)
    frame["browser_major"] = df["UserAgent"].apply(_extract_browser_major)
    frame["browser_platform"] = df["UserAgent"].apply(_extract_platform)

    frame["fdtag_leg_count"] = df["FdTag"].apply(_count_fdtag_legs)
    frame["fdtag_carrier_count"] = df["FdTag"].apply(_count_fdtag_carriers)
    frame["fdtag_codeshare_flag"] = df["FdTag"].apply(_fdtag_codeshare_flag)
    frame["flightsegments_leg_count"] = df["FlightSegments"].apply(_flightsegment_leg_count)
    frame["flightsegments_openjaw_flag"] = df["FlightSegments"].apply(_flightsegment_openjaw_flag)

    frame["route_airline"] = frame["route"] + "_" + frame["airline_code"]
    return frame


def apply_quality_filters(df):
    df = df.dropna(subset=["prediction_time", "departure_date", "trip_type", "origin_airport", "destination_airport"])
    df = df[df["days_to_departure"].ge(0)].copy()
    df = df[df["cache_age_hours"].between(0, 24, inclusive="both")].copy()
    return df.reset_index(drop=True)


def engineer_rolling_features(df):
    print("Computing leakage-safe rolling histories...")
    df = df.sort_values("prediction_time").reset_index(drop=True)

    df["is_bookable"] = (df["outcome_label"] == "bookable").astype(int)
    df["is_technical_fail"] = (df["outcome_label"] == "technical_failure").astype(int)

    global_success = float(df["is_bookable"].mean())
    global_tech_fail = float(df["is_technical_fail"].mean())

    def rolling_mean_prior(group, source_col, default_value):
        ordered = group.sort_values("prediction_time").set_index("prediction_time")[source_col]
        values = ordered.shift(1).rolling("7D", min_periods=5).mean().fillna(default_value)
        return values.reset_index(drop=True)

    df["airline_success_rate_7d"] = (
        df[["airline_code", "prediction_time", "is_bookable"]]
        .groupby("airline_code", group_keys=False)
        .apply(lambda g: rolling_mean_prior(g, "is_bookable", global_success))
        .to_numpy()
    )
    df["airline_tech_fail_rate_7d"] = (
        df[["airline_code", "prediction_time", "is_technical_fail"]]
        .groupby("airline_code", group_keys=False)
        .apply(lambda g: rolling_mean_prior(g, "is_technical_fail", global_tech_fail))
        .to_numpy()
    )
    df["route_airline_success_rate_7d"] = (
        df[["route_airline", "prediction_time", "is_bookable"]]
        .groupby("route_airline", group_keys=False)
        .apply(lambda g: rolling_mean_prior(g, "is_bookable", global_success))
        .to_numpy()
    )

    def rolling_count_prior(group):
        ordered = group.sort_values("prediction_time").set_index("prediction_time")["is_bookable"]
        values = ordered.shift(1).rolling("7D", min_periods=1).count().fillna(0.0)
        return values.reset_index(drop=True)

    df["route_queries_7d"] = (
        df[["route", "prediction_time", "is_bookable"]]
        .groupby("route", group_keys=False)
        .apply(rolling_count_prior)
        .to_numpy()
    )
    df["route_airline_queries_7d"] = (
        df[["route_airline", "prediction_time", "is_bookable"]]
        .groupby("route_airline", group_keys=False)
        .apply(rolling_count_prior)
        .to_numpy()
    )
    denominator = df["route_queries_7d"].replace(0, np.nan)
    df["airline_route_share_7d"] = (df["route_airline_queries_7d"] / denominator).fillna(0.0)

    return df.drop(
        columns=["is_bookable", "is_technical_fail", "route_queries_7d", "route_airline_queries_7d"]
    )


def main():
    if not os.path.exists(RAW_INPUT_FILE):
        print(f"Error: {RAW_INPUT_FILE} not found.")
        return

    print(f"Loading raw data from {RAW_INPUT_FILE}...")
    raw_df = pd.read_csv(RAW_INPUT_FILE)
    print(f"Raw shape: {raw_df.shape}")

    df = build_base_training_frame(raw_df)
    df = apply_quality_filters(df)
    df = engineer_rolling_features(df)

    keep_columns = [
        "prediction_time",
        "trip_type",
        "origin_airport",
        "destination_airport",
        "airline_code",
        "cabin_class",
        "meta_engine",
        "days_to_departure",
        "search_hour",
        "search_day",
        "dep_month",
        "is_weekend",
        "device_os",
        "itinerary_segments",
        "canonical_status",
        "label_confidence",
        "outcome_label",
        "cache_age_hours",
        "trip_duration_days",
        "trip_duration_bucket",
        "direct_flights_requested",
        "date_flexible_requested",
        "landing_ref_code",
        "has_gd_param",
        "has_lid_param",
        "has_skyscanner_redirectid",
        "skyscanner_id_present",
        "previous_page_domain",
        "previous_page_group",
        "previous_page_has_google",
        "previous_page_has_skyscanner",
        "previous_page_has_gclid",
        "cf_ipcountry",
        "accept_language_primary",
        "browser_name",
        "browser_major",
        "browser_platform",
        "fdtag_leg_count",
        "fdtag_carrier_count",
        "fdtag_codeshare_flag",
        "flightsegments_leg_count",
        "flightsegments_openjaw_flag",
        "airline_success_rate_7d",
        "airline_tech_fail_rate_7d",
        "route_airline_success_rate_7d",
        "airline_route_share_7d",
    ]
    keep_columns = [column for column in keep_columns if column in df.columns]
    df = df[keep_columns].copy()

    print(f"Engineered shape: {df.shape}")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved real raw-derived feature set to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
