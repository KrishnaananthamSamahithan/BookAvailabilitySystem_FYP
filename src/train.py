import json
from typing import Dict, List

import joblib
import pandas as pd

from src.calibration import align_calibrated_probabilities, compare_calibrators
from src.config import ProjectConfig
from src.evaluation import comparison_table, evaluate_predictions
from src.features import build_feature_bundle, build_snapshot_feature_bundle
from src.labels import FINAL_LABELS, encode_target
from src.models import build_candidates, compare_models, predict_proba_aligned
from src.preprocessing import preprocess_raw_file
from src.reporting import calibration_parity_report, subgroup_metrics
from src.simulation import reranking_proxy_simulation, suppression_policy_simulation
from src.split import rolling_backtest_splits, temporal_train_validation_test_split
from src.utils import save_csv, save_json


def run_training_pipeline(config: ProjectConfig | None = None) -> Dict[str, object]:
    """Run the final research-grade pipeline.

    This is the single authoritative training path for the dissertation project.
    It avoids duplicate notebook-specific logic and keeps all decisions traceable
    through saved artifacts.
    """

    config = config or ProjectConfig()
    config.ensure_directories()

    preprocessing_result = preprocess_raw_file(config)
    processed = preprocessing_result.frame.copy()

    supervised = processed.loc[processed["outcome_label"].isin(FINAL_LABELS)].copy()
    split_raw = temporal_train_validation_test_split(
        supervised,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    save_json(config.reports_dir / "split_metadata.json", split_raw.metadata)

    train_bundle = build_feature_bundle(split_raw.train, include_labels=True)
    validation_bundle = build_snapshot_feature_bundle(split_raw.validation, split_raw.train)
    history_for_test = pd.concat([split_raw.train, split_raw.validation], ignore_index=True)
    test_bundle = build_snapshot_feature_bundle(split_raw.test, history_for_test)

    split = type("FeatureSplit", (), {})()
    split.train = train_bundle.frame.copy()
    split.validation = validation_bundle.frame.copy()
    split.test = test_bundle.frame.copy()
    split.metadata = split_raw.metadata

    feature_bundle = train_bundle
    split.train["target"], target_mapping = encode_target(split.train["outcome_label"])
    split.validation["target"] = split.validation["outcome_label"].map(target_mapping)
    split.test["target"] = split.test["outcome_label"].map(target_mapping)

    full_feature_columns = feature_bundle.feature_groups["full_valid_model"]
    train_X = split.train[full_feature_columns]
    train_y = split.train["target"].to_numpy()
    validation_X = split.validation[full_feature_columns]
    validation_y = split.validation["target"].to_numpy()
    test_X = split.test[full_feature_columns]
    test_y = split.test["target"].to_numpy()

    model_results = compare_models(
        X_train=train_X,
        y_train=train_y,
        X_validation=validation_X,
        y_validation=validation_y,
        labels=FINAL_LABELS,
        categorical_features=[f for f in feature_bundle.categorical_features if f in full_feature_columns],
        numeric_features=[f for f in feature_bundle.numeric_features if f in full_feature_columns],
        alpha=config.calibration_alpha,
        random_state=config.random_state,
    )
    model_rows = _model_result_rows(model_results)
    save_csv(config.reports_dir / "model_comparison.csv", comparison_table(model_rows))
    save_json(config.reports_dir / "model_comparison.json", {"rows": model_rows})

    best_model_result = model_results[0]
    best_estimator = best_model_result.estimator

    calibration_results = compare_calibrators(best_model_result.validation_probabilities, validation_y, FINAL_LABELS)
    calibration_rows = _calibration_result_rows(calibration_results)
    save_csv(config.reports_dir / "calibration_comparison.csv", comparison_table(calibration_rows))
    save_json(config.reports_dir / "calibration_comparison.json", {"rows": calibration_rows})

    best_calibrator_result = calibration_results[0]
    raw_test_probabilities = predict_proba_aligned(best_estimator, test_X, list(range(len(FINAL_LABELS))))
    calibrated_test_probabilities = align_calibrated_probabilities(
        best_calibrator_result.calibrator,
        raw_test_probabilities,
        list(range(len(FINAL_LABELS))),
    )
    final_metrics = evaluate_predictions(test_y, calibrated_test_probabilities, FINAL_LABELS)
    save_json(config.reports_dir / "final_test_metrics.json", final_metrics)

    subgroup_reports = {
        "provider_key": subgroup_metrics(split.test, test_y, calibrated_test_probabilities, "provider_key", FINAL_LABELS),
        "route": subgroup_metrics(split.test, test_y, calibrated_test_probabilities, "route", FINAL_LABELS, top_n=20),
        "airline_code": subgroup_metrics(split.test, test_y, calibrated_test_probabilities, "airline_code", FINAL_LABELS),
        "days_to_departure_bucket": subgroup_metrics(split.test, test_y, calibrated_test_probabilities, "days_to_departure_bucket", FINAL_LABELS),
    }
    save_json(config.reports_dir / "subgroup_evaluation.json", subgroup_reports)
    save_json(
        config.reports_dir / "calibration_parity.json",
        calibration_parity_report(
            split.test,
            test_y,
            calibrated_test_probabilities,
            ["provider_key", "route", "airline_code"],
            FINAL_LABELS,
        ),
    )

    ablation_rows = run_ablation_study(
        train_frame=split.train,
        validation_frame=split.validation,
        feature_bundle=feature_bundle,
        model_name=best_model_result.name,
        random_state=config.random_state,
    )
    save_csv(config.reports_dir / "ablation_results.csv", pd.DataFrame(ablation_rows))
    save_json(config.reports_dir / "ablation_results.json", {"rows": ablation_rows})

    backtest_rows = run_rolling_backtest(
        features=split.train.assign(split_part="train").pipe(
            lambda train_frame: pd.concat(
                [
                    train_frame,
                    split.validation.assign(split_part="validation"),
                    split.test.assign(split_part="test"),
                ],
                ignore_index=True,
            )
        ),
        feature_bundle=feature_bundle,
        model_name=best_model_result.name,
        random_state=config.random_state,
        n_windows=config.rolling_backtest_windows,
    )
    save_csv(config.reports_dir / "rolling_backtest.csv", pd.DataFrame(backtest_rows))
    save_json(config.reports_dir / "rolling_backtest.json", {"rows": backtest_rows})

    policy_results = {
        "suppression_policy": suppression_policy_simulation(split.test, calibrated_test_probabilities, FINAL_LABELS, thresholds=[0.2, 0.4, 0.6, 0.8]),
        "reranking_proxy": reranking_proxy_simulation(split.test, calibrated_test_probabilities, FINAL_LABELS),
    }
    save_json(config.reports_dir / "policy_simulation_results.json", policy_results)

    save_csv(config.reports_dir / "feature_availability_audit.csv", pd.DataFrame(feature_bundle.feature_availability))
    save_json(config.reports_dir / "feature_availability_audit.json", {"rows": feature_bundle.feature_availability})

    ambiguous_experiment = run_ambiguous_handling_experiment(processed, config)
    save_json(config.reports_dir / "ambiguous_handling_experiment.json", ambiguous_experiment)

    feature_inventory = {
        "categorical_features": feature_bundle.categorical_features,
        "numeric_features": feature_bundle.numeric_features,
        "feature_groups": feature_bundle.feature_groups,
    }
    save_json(config.reports_dir / "feature_inventory.json", feature_inventory)

    model_bundle = {
        "model_name": best_model_result.name,
        "estimator": best_estimator,
        "calibrator_name": best_calibrator_result.name,
        "calibrator": best_calibrator_result.calibrator,
        "feature_columns": full_feature_columns,
        "categorical_features": [f for f in feature_bundle.categorical_features if f in full_feature_columns],
        "numeric_features": [f for f in feature_bundle.numeric_features if f in full_feature_columns],
        "target_mapping": target_mapping,
        "labels": FINAL_LABELS,
        "inference_reference": feature_bundle.inference_reference,
        "prediction_time_assumption": config.prediction_time_assumption,
        "config": config.to_dict(),
    }
    joblib.dump(model_bundle, config.models_dir / "final_model_bundle.joblib")

    summary = {
        "best_model": best_model_result.name,
        "best_model_selection_score": best_model_result.selection_score,
        "best_calibrator": best_calibrator_result.name,
        "final_test_macro_f1": final_metrics["macro_f1"],
        "final_test_log_loss": final_metrics["log_loss"],
        "final_test_balanced_accuracy": final_metrics["balanced_accuracy"],
        "final_test_weighted_f1": final_metrics["weighted_f1"],
        "processed_rows": int(len(processed)),
        "supervised_rows": int(len(supervised)),
        "artifacts_dir": str(config.artifacts_dir),
    }
    save_json(config.reports_dir / "summary.json", summary)
    return summary


def run_ablation_study(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_bundle,
    model_name: str,
    random_state: int,
) -> List[Dict[str, object]]:
    groups = ["topology_basic", "temporal", "price", "reliability_history", "full_valid_model"]
    rows = []
    for group_name in groups:
        if group_name == "full_valid_model":
            feature_columns = feature_bundle.feature_groups[group_name]
        elif group_name == "temporal":
            feature_columns = feature_bundle.feature_groups["topology_basic"] + feature_bundle.feature_groups["temporal"]
        elif group_name == "price":
            feature_columns = feature_bundle.feature_groups["topology_basic"] + feature_bundle.feature_groups["temporal"] + feature_bundle.feature_groups["price"]
        elif group_name == "reliability_history":
            feature_columns = feature_bundle.feature_groups["topology_basic"] + feature_bundle.feature_groups["temporal"] + feature_bundle.feature_groups["reliability_history"]
        else:
            feature_columns = feature_bundle.feature_groups[group_name]

        estimator = _fit_single_model(model_name, train_frame, validation_frame, feature_columns, feature_bundle, random_state)
        probabilities = predict_proba_aligned(estimator, validation_frame[feature_columns], list(range(len(FINAL_LABELS))))
        metrics = evaluate_predictions(validation_frame["target"].to_numpy(), probabilities, FINAL_LABELS)
        rows.append(
            {
                "feature_group": group_name,
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "log_loss": metrics["log_loss"],
                "multiclass_brier": metrics["multiclass_brier"],
                "ece_macro": metrics["ece_macro"],
            }
        )
    return rows


def run_rolling_backtest(
    features: pd.DataFrame,
    feature_bundle,
    model_name: str,
    random_state: int,
    n_windows: int,
) -> List[Dict[str, object]]:
    rows = []
    base_frame = features.drop(columns=["split_part"], errors="ignore").copy()
    for window in rolling_backtest_splits(base_frame, n_windows=n_windows):
        train_bundle = build_feature_bundle(window["train"], include_labels=True)
        test_bundle = build_snapshot_feature_bundle(window["test"], window["train"])
        train_bundle.frame["target"] = train_bundle.frame["outcome_label"].map({label: idx for idx, label in enumerate(FINAL_LABELS)})
        test_bundle.frame["target"] = test_bundle.frame["outcome_label"].map({label: idx for idx, label in enumerate(FINAL_LABELS)})
        feature_columns = train_bundle.feature_groups["full_valid_model"]
        estimator = _fit_single_model(model_name, train_bundle.frame, test_bundle.frame, feature_columns, train_bundle, random_state)
        probabilities = predict_proba_aligned(estimator, test_bundle.frame[feature_columns], list(range(len(FINAL_LABELS))))
        metrics = evaluate_predictions(test_bundle.frame["target"].to_numpy(), probabilities, FINAL_LABELS)
        rows.append(
            {
                "window_id": window["window_id"],
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "macro_f1": metrics["macro_f1"],
                "log_loss": metrics["log_loss"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
        )
    return rows


def run_ambiguous_handling_experiment(processed: pd.DataFrame, config: ProjectConfig) -> Dict[str, object]:
    """Compare the default ambiguous-excluded setup with a simple ambiguous-kept experiment.

    This is intentionally lightweight. The dissertation point is to make the
    treatment of ambiguous rows explicit, not to claim a fully noise-robust
    alternate labeling system.
    """

    ambiguous_rows = processed["outcome_label"].eq("ambiguous").sum()
    result = {
        "ambiguous_row_count": int(ambiguous_rows),
        "default_training_policy": "exclude_ambiguous_rows",
        "experimental_policy": "keep_ambiguous_as_fifth_class_for_diagnostic_comparison",
    }

    if ambiguous_rows == 0:
        result["status"] = "skipped_no_ambiguous_rows"
        return result

    frame = processed.copy()
    split_raw = temporal_train_validation_test_split(frame, config.train_fraction, config.validation_fraction, config.test_fraction)
    train_bundle = build_feature_bundle(split_raw.train, include_labels=True)
    validation_bundle = build_snapshot_feature_bundle(split_raw.validation, split_raw.train)
    mapping = {label: idx for idx, label in enumerate(FINAL_LABELS + ["ambiguous"])}
    train_bundle.frame["target_5class"] = train_bundle.frame["outcome_label"].map(mapping)
    validation_bundle.frame["target_5class"] = validation_bundle.frame["outcome_label"].map(mapping)
    feature_columns = train_bundle.feature_groups["full_valid_model"]
    estimator = build_candidates(
        categorical_features=[f for f in train_bundle.categorical_features if f in feature_columns],
        numeric_features=[f for f in train_bundle.numeric_features if f in feature_columns],
        random_state=config.random_state,
    )["catboost"]
    estimator.fit(
        train_bundle.frame[feature_columns],
        train_bundle.frame["target_5class"].to_numpy(),
        cat_features=[f for f in train_bundle.categorical_features if f in feature_columns],
        eval_set=(validation_bundle.frame[feature_columns], validation_bundle.frame["target_5class"].to_numpy()),
    )
    probabilities = predict_proba_aligned(estimator, validation_bundle.frame[feature_columns], list(range(5)))
    metrics = evaluate_predictions(validation_bundle.frame["target_5class"].to_numpy(), probabilities, FINAL_LABELS + ["ambiguous"])
    result["status"] = "completed"
    result["validation_macro_f1_5class"] = metrics["macro_f1"]
    result["validation_log_loss_5class"] = metrics["log_loss"]
    return result


def _fit_single_model(model_name: str, train_frame: pd.DataFrame, validation_frame: pd.DataFrame, feature_columns: List[str], feature_bundle, random_state: int):
    candidates = build_candidates(
        categorical_features=[f for f in feature_bundle.categorical_features if f in feature_columns],
        numeric_features=[f for f in feature_bundle.numeric_features if f in feature_columns],
        random_state=random_state,
    )
    estimator = candidates[model_name]
    X_train = train_frame[feature_columns]
    y_train = train_frame["target"].to_numpy()
    X_validation = validation_frame[feature_columns]
    y_validation = validation_frame["target"].to_numpy()
    if model_name == "catboost":
        estimator.fit(
            X_train,
            y_train,
            cat_features=[f for f in feature_bundle.categorical_features if f in feature_columns],
            eval_set=(X_validation, y_validation),
        )
    else:
        estimator.fit(X_train, y_train)
    return estimator


def _model_result_rows(results) -> List[Dict[str, object]]:
    rows = []
    for result in results:
        rows.append(
            {
                "name": result.name,
                "selection_score": result.selection_score,
                "accuracy": result.validation_metrics["accuracy"],
                "balanced_accuracy": result.validation_metrics["balanced_accuracy"],
                "macro_f1": result.validation_metrics["macro_f1"],
                "weighted_f1": result.validation_metrics["weighted_f1"],
                "log_loss": result.validation_metrics["log_loss"],
                "multiclass_brier": result.validation_metrics["multiclass_brier"],
                "ece_macro": result.validation_metrics["ece_macro"],
                "tuning_metadata": result.tuning_metadata,
            }
        )
    return rows


def _calibration_result_rows(results) -> List[Dict[str, object]]:
    rows = []
    for result in results:
        rows.append(
            {
                "name": result.name,
                "selection_score": result.selection_score,
                "accuracy": result.validation_metrics["accuracy"],
                "balanced_accuracy": result.validation_metrics["balanced_accuracy"],
                "macro_f1": result.validation_metrics["macro_f1"],
                "weighted_f1": result.validation_metrics["weighted_f1"],
                "log_loss": result.validation_metrics["log_loss"],
                "multiclass_brier": result.validation_metrics["multiclass_brier"],
                "ece_macro": result.validation_metrics["ece_macro"],
            }
        )
    return rows


if __name__ == "__main__":
    summary = run_training_pipeline()
    print(json.dumps(summary, indent=2))
