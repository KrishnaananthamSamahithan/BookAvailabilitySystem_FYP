import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, classification_report, log_loss, f1_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from xgboost import XGBClassifier

INPUT_CANDIDATES = [
    "data/processed/training_data.csv",
    "data/processed/processed_flight_data_full.csv",
]
MODEL_BUNDLE_FILE = "models/production_model_bundle.joblib"
CATBOOST_MODEL_FILE = "models/catboost_production.cbm"
METRICS_FILE = "metrics/metrics.json"
MODEL_COMPARISON_FILE = "metrics/model_comparison.json"
ERROR_ANALYSIS_FILE = "metrics/price_changed_error_analysis.json"

outcome_map = {
    "bookable": 0,
    "price_changed": 1,
    "unavailable": 2,
    "technical_failure": 3,
}


def choose_input_file():
    for candidate in INPUT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"No input file found in {INPUT_CANDIDATES}")


def load_dataset():
    input_path = choose_input_file()
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}.")
    return df, str(input_path)


def engineer_features(df):
    print("Engineering features...")
    df = df.copy()

    df["prediction_time"] = pd.to_datetime(df["prediction_time"], format="mixed", errors="coerce")
    df = df.dropna(subset=["prediction_time"]).sort_values("prediction_time").reset_index(drop=True)

    df["route"] = df["origin_airport"].astype(str) + "_" + df["destination_airport"].astype(str)
    df["route_airline"] = df["route"] + "__" + df["airline_code"].astype(str)
    df["missing_airline"] = df["airline_code"].eq("Unknown").astype(int)
    df["search_month"] = df["prediction_time"].dt.month.astype(int)
    df["search_dayofweek"] = df["prediction_time"].dt.dayofweek.astype(int)
    df["search_weekofyear"] = df["prediction_time"].dt.isocalendar().week.astype(int)
    df["days_to_departure_bucket"] = pd.cut(
        df["days_to_departure"],
        bins=[-1, 3, 7, 14, 30, 60, 120, 365],
        labels=["0_3", "4_7", "8_14", "15_30", "31_60", "61_120", "121_plus"],
    ).astype(str)
    df["is_short_lead"] = df["days_to_departure"].le(14).astype(int)
    df["is_last_minute"] = df["days_to_departure"].le(3).astype(int)
    df["is_long_haul_booking_window"] = df["days_to_departure"].ge(120).astype(int)

    if "itinerary_segments" in df.columns:
        df["itinerary_segments_bucket"] = pd.cut(
            df["itinerary_segments"],
            bins=[-1, 1, 2, 3, 99],
            labels=["1", "2", "3", "4_plus"],
        ).astype(str)

    if "cache_age_hours" in df.columns:
        df["cache_age_bucket"] = pd.cut(
            df["cache_age_hours"],
            bins=[-1, 1, 2, 6, 24, 1000],
            labels=["subhour", "1_2h", "2_6h", "6_24h", "24h_plus"],
        ).astype(str)

    if "price_usd_imputed" in df.columns:
        df["log_price_usd"] = np.log1p(df["price_usd_imputed"].clip(lower=0))
        df["price_bucket"] = pd.qcut(
            df["price_usd_imputed"].rank(method="first"),
            q=5,
            labels=["very_low", "low", "mid", "high", "very_high"],
        ).astype(str)

    if "price_gap_to_min" in df.columns:
        denominator = np.where(df.get("price_usd_imputed", pd.Series(np.ones(len(df)))).to_numpy() <= 0, 1.0, df.get("price_usd_imputed", pd.Series(np.ones(len(df)))).to_numpy())
        df["price_gap_ratio"] = df["price_gap_to_min"].to_numpy() / denominator
        df["price_gap_bucket"] = pd.qcut(
            df["price_gap_to_min"].rank(method="first"),
            q=5,
            labels=["very_low", "low", "mid", "high", "very_high"],
        ).astype(str)
        if "cache_age_hours" in df.columns:
            df["cache_gap_interaction"] = df["cache_age_hours"] * df["price_gap_ratio"]

    base_rate_labels = ["bookable", "price_changed", "unavailable"]
    global_class_rates = {
        label: float((df["outcome_label"] == label).mean())
        for label in base_rate_labels
    }

    _add_prior_count_feature(df, "meta_engine", "provider_prior_count")
    _add_prior_count_feature(df, "route", "route_prior_count")
    _add_prior_count_feature(df, "airline_code", "airline_prior_count")
    _add_prior_count_feature(df, "route_airline", "route_airline_prior_count")

    for label in base_rate_labels:
        indicator = (df["outcome_label"] == label).astype(int)
        df[f"is_{label}"] = indicator
        _add_prior_rate_feature(
            df, "meta_engine", f"is_{label}", f"provider_{label}_rate", global_class_rates[label]
        )
        _add_prior_rate_feature(
            df, "route", f"is_{label}", f"route_{label}_rate", global_class_rates[label]
        )
        _add_prior_rate_feature(
            df, "airline_code", f"is_{label}", f"airline_{label}_rate", global_class_rates[label]
        )
        _add_prior_rate_feature(
            df, "route_airline", f"is_{label}", f"route_airline_{label}_rate", global_class_rates[label]
        )
        df.drop(columns=[f"is_{label}"], inplace=True)

    df["provider_instability_rate"] = 1.0 - df["provider_bookable_rate"]
    df["route_instability_rate"] = 1.0 - df["route_bookable_rate"]
    df["price_change_pressure"] = (
        df["provider_price_changed_rate"] + df["route_price_changed_rate"] + df["airline_price_changed_rate"]
    ) / 3.0
    df["unavailable_pressure"] = (
        df["provider_unavailable_rate"] + df["route_unavailable_rate"] + df["airline_unavailable_rate"]
    ) / 3.0

    if "route_airline_success_rate_7d" in df.columns and "airline_success_rate_7d" in df.columns:
        df["route_airline_edge_vs_airline"] = (
            df["route_airline_success_rate_7d"] - df["airline_success_rate_7d"]
        )
    if "price_gap_ratio" in df.columns and "price_change_pressure" in df.columns:
        df["price_pressure_interaction"] = df["price_gap_ratio"] * df["price_change_pressure"]
    if "airline_route_share_7d" in df.columns and "route_airline_success_rate_7d" in df.columns:
        df["route_share_success_interaction"] = (
            df["airline_route_share_7d"] * df["route_airline_success_rate_7d"]
        )

    return df


def _add_prior_count_feature(df, group_col, output_col):
    df[output_col] = df.groupby(group_col).cumcount().astype(int)


def _add_prior_rate_feature(df, group_col, indicator_col, output_col, default_value):
    grouped = df.groupby(group_col)[indicator_col]
    prior_positive = grouped.cumsum() - df[indicator_col]
    prior_count = grouped.cumcount()
    df[output_col] = np.where(prior_count > 0, prior_positive / prior_count, default_value)


def temporal_split(df, train_frac=0.7, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    print(f"Temporal Split -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def get_feature_lists(df):
    categorical_candidates = [
        "trip_type",
        "cabin_class",
        "meta_engine",
        "origin_airport",
        "destination_airport",
        "airline_code",
        "device_os",
        "route",
        "route_airline",
        "days_to_departure_bucket",
        "itinerary_segments_bucket",
        "cache_age_bucket",
        "price_bucket",
        "price_gap_bucket",
    ]
    numeric_candidates = [
        "days_to_departure",
        "search_hour",
        "search_day",
        "search_month",
        "search_dayofweek",
        "search_weekofyear",
        "dep_month",
        "is_weekend",
        "itinerary_segments",
        "missing_airline",
        "is_short_lead",
        "is_last_minute",
        "is_long_haul_booking_window",
        "cache_age_hours",
        "price_usd_imputed",
        "price_gap_to_min",
        "price_gap_ratio",
        "log_price_usd",
        "cache_gap_interaction",
        "provider_prior_count",
        "route_prior_count",
        "airline_prior_count",
        "route_airline_prior_count",
        "provider_bookable_rate",
        "provider_price_changed_rate",
        "provider_unavailable_rate",
        "route_bookable_rate",
        "route_price_changed_rate",
        "route_unavailable_rate",
        "airline_bookable_rate",
        "airline_price_changed_rate",
        "airline_unavailable_rate",
        "route_airline_bookable_rate",
        "route_airline_price_changed_rate",
        "route_airline_unavailable_rate",
        "provider_instability_rate",
        "route_instability_rate",
        "price_change_pressure",
        "unavailable_pressure",
        "airline_success_rate_7d",
        "airline_tech_fail_rate_7d",
        "route_airline_success_rate_7d",
        "airline_route_share_7d",
        "route_airline_edge_vs_airline",
        "price_pressure_interaction",
        "route_share_success_interaction",
    ]

    categorical_features = [col for col in categorical_candidates if col in df.columns]
    numeric_features = [col for col in numeric_candidates if col in df.columns]
    return categorical_features, numeric_features


def prepare_datasets(train, val, test, categorical_features, numeric_features):
    feature_columns = categorical_features + numeric_features

    for frame in (train, val, test):
        for col in categorical_features:
            frame[col] = frame[col].fillna("Unknown").astype(str)
        for col in numeric_features:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    X_train = train[feature_columns].copy()
    X_val = val[feature_columns].copy()
    X_test = test[feature_columns].copy()

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if categorical_features:
        encoder.fit(X_train[categorical_features])
        X_train_encoded = X_train.copy()
        X_val_encoded = X_val.copy()
        X_test_encoded = X_test.copy()
        X_train_encoded[categorical_features] = encoder.transform(X_train[categorical_features])
        X_val_encoded[categorical_features] = encoder.transform(X_val[categorical_features])
        X_test_encoded[categorical_features] = encoder.transform(X_test[categorical_features])
    else:
        X_train_encoded, X_val_encoded, X_test_encoded = X_train.copy(), X_val.copy(), X_test.copy()

    return {
        "feature_columns": feature_columns,
        "raw": {"train": X_train, "val": X_val, "test": X_test},
        "encoded": {"train": X_train_encoded, "val": X_val_encoded, "test": X_test_encoded},
        "categorical": categorical_features,
        "numeric": numeric_features,
    }


def selection_score(y_true, preds, probs):
    macro = f1_score(y_true, preds, average="macro", zero_division=0)
    acc = accuracy_score(y_true, preds)
    ll = log_loss(y_true, probs, labels=sorted(pd.Series(y_true).unique()))
    return {
        "macro_f1": float(macro),
        "accuracy": float(acc),
        "log_loss": float(ll),
        "selection_score": float(macro + (0.15 * acc) - (0.05 * ll)),
    }


def train_catboost(dataset_bundle, y_train, y_val):
    trials = [
        {"iterations": 180, "learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 5},
    ]
    class_ids = sorted(pd.Series(y_train).unique())
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array(class_ids), y=y_train)

    results = []
    for params in trials:
        print(f"CatBoost trial: {params}")
        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            verbose=50,
            random_seed=42,
            class_weights=list(class_weights),
            **params,
        )
        model.fit(
            dataset_bundle["raw"]["train"],
            y_train,
            cat_features=dataset_bundle["categorical"],
            eval_set=(dataset_bundle["raw"]["val"], y_val),
            early_stopping_rounds=40,
            use_best_model=True,
        )
        val_preds = model.predict(dataset_bundle["raw"]["val"]).astype(int).ravel()
        val_probs = model.predict_proba(dataset_bundle["raw"]["val"])
        metrics = selection_score(y_val, val_preds, val_probs)
        results.append(
            {
                "model_name": "catboost",
                "params": params,
                "model": model,
                **metrics,
            }
        )
    return results


def train_xgboost(dataset_bundle, y_train, y_val):
    trials = [
        {"n_estimators": 180, "learning_rate": 0.06, "max_depth": 8, "subsample": 0.9, "colsample_bytree": 0.8},
    ]
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    results = []
    for params in trials:
        print(f"XGBoost trial: {params}")
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(sorted(pd.Series(y_train).unique())),
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=42,
            min_child_weight=2,
            reg_lambda=1.5,
            **params,
        )
        model.fit(
            dataset_bundle["encoded"]["train"],
            y_train,
            sample_weight=sample_weight,
            eval_set=[(dataset_bundle["encoded"]["val"], y_val)],
            verbose=False,
        )
        val_probs = model.predict_proba(dataset_bundle["encoded"]["val"])
        val_preds = np.asarray(val_probs).argmax(axis=1)
        metrics = selection_score(y_val, val_preds, val_probs)
        results.append(
            {
                "model_name": "xgboost",
                "params": params,
                "model": model,
                **metrics,
            }
        )
    return results


def train_lightgbm(dataset_bundle, y_train, y_val):
    trials = [
        {"n_estimators": 180, "learning_rate": 0.06, "num_leaves": 63, "max_depth": -1, "min_child_samples": 30},
    ]
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    results = []
    for params in trials:
        print(f"LightGBM trial: {params}")
        model = LGBMClassifier(
            objective="multiclass",
            n_jobs=-1,
            random_state=42,
            **params,
        )
        model.fit(
            dataset_bundle["encoded"]["train"],
            y_train,
            sample_weight=sample_weight,
            eval_set=[(dataset_bundle["encoded"]["val"], y_val)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(40, verbose=False), log_evaluation(0)],
        )
        val_probs = model.predict_proba(dataset_bundle["encoded"]["val"])
        val_preds = np.asarray(val_probs).argmax(axis=1)
        metrics = selection_score(y_val, val_preds, val_probs)
        results.append(
            {
                "model_name": "lightgbm",
                "params": params,
                "model": model,
                **metrics,
            }
        )
    return results


def evaluate_model(best_result, dataset_bundle, y_test):
    model_name = best_result["model_name"]
    if model_name == "catboost":
        X_test = dataset_bundle["raw"]["test"]
    else:
        X_test = dataset_bundle["encoded"]["test"]

    probs = best_result["model"].predict_proba(X_test)
    preds = np.asarray(probs).argmax(axis=1)
    unique_targets = sorted(pd.Series(y_test).unique())
    target_names = [inverse_outcome_map()[target] for target in unique_targets]
    report_dict = classification_report(
        y_test,
        preds,
        labels=unique_targets,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    print("\n--- Test Classification Report ---")
    print(
        classification_report(
            y_test,
            preds,
            labels=unique_targets,
            target_names=target_names,
            zero_division=0,
        )
    )
    metrics = selection_score(y_test, preds, probs)
    metrics["weighted_f1"] = float(f1_score(y_test, preds, average="weighted", zero_division=0))
    metrics["classification_report"] = report_dict
    return preds, probs, metrics


def inverse_outcome_map():
    return {value: key for key, value in outcome_map.items()}


def build_price_changed_error_analysis(test_frame, y_test, preds):
    frame = test_frame.copy()
    frame["y_true"] = y_test.to_numpy()
    frame["y_pred"] = preds
    price_changed_id = outcome_map["price_changed"]
    price_changed_rows = frame[frame["y_true"] == price_changed_id].copy()
    false_negatives = price_changed_rows[price_changed_rows["y_pred"] != price_changed_id].copy()

    analysis = {
        "price_changed_support": int(len(price_changed_rows)),
        "price_changed_false_negatives": int(len(false_negatives)),
        "price_changed_recall": float((price_changed_rows["y_pred"] == price_changed_id).mean()) if len(price_changed_rows) else 0.0,
        "top_false_negative_slices": {},
    }

    slice_columns = [
        "meta_engine",
        "trip_type",
        "days_to_departure_bucket",
        "price_gap_bucket",
        "cache_age_bucket",
        "itinerary_segments_bucket",
        "route",
        "airline_code",
    ]
    for column in slice_columns:
        if column not in frame.columns:
            continue
        grouped = (
            false_negatives[column]
            .fillna("Unknown")
            .value_counts()
            .head(10)
            .rename_axis(column)
            .reset_index(name="false_negative_count")
        )
        support = (
            price_changed_rows[column]
            .fillna("Unknown")
            .value_counts()
            .rename_axis(column)
            .reset_index(name="price_changed_support")
        )
        merged = grouped.merge(support, on=column, how="left")
        merged["false_negative_rate_within_slice"] = (
            merged["false_negative_count"] / merged["price_changed_support"].clip(lower=1)
        )
        analysis["top_false_negative_slices"][column] = merged.to_dict(orient="records")

    return analysis


def save_json(path_string, payload):
    path = Path(path_string)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    df, source_path = load_dataset()
    df = df[df["outcome_label"].isin(outcome_map.keys())].copy()
    df["target"] = df["outcome_label"].map(outcome_map)
    print(f"Filtered to {len(df)} confident rows (dropped ambiguous).")

    df = engineer_features(df)
    categorical_features, numeric_features = get_feature_lists(df)
    train, val, test = temporal_split(df)
    dataset_bundle = prepare_datasets(train, val, test, categorical_features, numeric_features)

    y_train = train["target"].astype(int).to_numpy()
    y_val = val["target"].astype(int).to_numpy()
    y_test = test["target"].astype(int).to_numpy()

    comparison_results = []
    comparison_results.extend(train_catboost(dataset_bundle, y_train, y_val))
    comparison_results.extend(train_xgboost(dataset_bundle, y_train, y_val))
    comparison_results.extend(train_lightgbm(dataset_bundle, y_train, y_val))

    best_result = max(comparison_results, key=lambda item: item["selection_score"])
    print(
        "Best validation setup:"
        f" model={best_result['model_name']}"
        f" params={best_result['params']}"
        f" macro_f1={best_result['macro_f1']:.4f}"
        f" accuracy={best_result['accuracy']:.4f}"
        f" log_loss={best_result['log_loss']:.4f}"
    )

    preds, probs, test_metrics = evaluate_model(best_result, dataset_bundle, y_test)
    error_analysis = build_price_changed_error_analysis(test, pd.Series(y_test), preds)

    comparison_payload = {
        "rows": [
            {
                "model_name": row["model_name"],
                "params": row["params"],
                "macro_f1": row["macro_f1"],
                "accuracy": row["accuracy"],
                "log_loss": row["log_loss"],
                "selection_score": row["selection_score"],
            }
            for row in sorted(comparison_results, key=lambda item: item["selection_score"], reverse=True)
        ]
    }

    metrics_payload = {
        "source_path": source_path,
        "validation": {
            "best_model": best_result["model_name"],
            "best_params": best_result["params"],
            "macro_f1": best_result["macro_f1"],
            "accuracy": best_result["accuracy"],
            "log_loss": best_result["log_loss"],
            "selection_score": best_result["selection_score"],
        },
        "test": test_metrics,
        "features": {
            "categorical": dataset_bundle["categorical"],
            "numerical": dataset_bundle["numeric"],
        },
    }

    bundle = {
        "model_name": best_result["model_name"],
        "model": best_result["model"],
        "feature_columns": dataset_bundle["feature_columns"],
        "categorical_features": dataset_bundle["categorical"],
        "numeric_features": dataset_bundle["numeric"],
        "encoder": None if not dataset_bundle["categorical"] else "ordinal_encoder_applied_in_saved_bundle_metadata_only",
        "params": best_result["params"],
        "source_path": source_path,
        "outcome_map": outcome_map,
    }

    # Save concrete encoder state for non-CatBoost models.
    if dataset_bundle["categorical"]:
        bundle["encoder"] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        bundle["encoder"].fit(dataset_bundle["raw"]["train"][dataset_bundle["categorical"]])

    Path(MODEL_BUNDLE_FILE).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_BUNDLE_FILE)
    if best_result["model_name"] == "catboost":
        Path(CATBOOST_MODEL_FILE).parent.mkdir(parents=True, exist_ok=True)
        best_result["model"].save_model(CATBOOST_MODEL_FILE)

    save_json(METRICS_FILE, metrics_payload)
    save_json(MODEL_COMPARISON_FILE, comparison_payload)
    save_json(ERROR_ANALYSIS_FILE, error_analysis)

    print(f"Metrics saved to {METRICS_FILE}")
    print(f"Model comparison saved to {MODEL_COMPARISON_FILE}")
    print(f"Price-changed error analysis saved to {ERROR_ANALYSIS_FILE}")
    print(f"Model bundle saved to {MODEL_BUNDLE_FILE}")
    if best_result["model_name"] == "catboost":
        print(f"CatBoost model also saved to {CATBOOST_MODEL_FILE}")


if __name__ == "__main__":
    main()
