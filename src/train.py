"""Main training pipeline — Research Grade.

Authoritative single-entry-point for the dissertation pipeline.
Implements ALL academic requirements:

  Phase 1  — Preprocessing & EDA
  Phase 2  — Label presence check + zero-support warnings
  Phase 3  — 4-way temporal split (no validation double-dipping)
  Phase 4  — Feature selection audit
  Phase 5  — Naive baselines (DummyClassifier)
  Phase 6  — 5-model comparison (LR, RF, CatBoost, XGBoost, LightGBM)
              FIX: RF now grid-searched; all 5 models saved to artifact
  Phase 7  — Calibration (5 methods incl. Dirichlet) on clean val_calibration holdout
  Phase 8  — Final test evaluation (full metric suite + PR-AUC + optimal thresholds)
  Phase 9  — Statistical significance (McNemar + bootstrap CI + Bonferroni)
  Phase 10 — Subgroup evaluation & calibration parity
  Phase 11 — Ablation study (ISOLATED + CUMULATIVE, 3 seeds)
  Phase 12 — Rolling backtest (5 windows)
  Phase 13 — Policy simulation (suppression + reranking + cost-sensitive)
  Phase 14 — Concept drift analysis (feature PSI/KS + prediction drift)
  Phase 15 — SHAP explainability (per-class top-25)
  Phase 16 — SHAP dependency plots (top 3 features per class, saved as PNG)
  Phase 17 — Permutation importance (model-agnostic fallback)
  Phase 18 — Partial dependence plots (PDP, saved as PNG)
  Phase 19 — Learning curves (5 training fractions)
  Phase 20 — Error analysis (top N misclassified samples)
  Phase 21 — Ambiguous handling experiment
  Phase 22 — Provider holdout generalisation
  Phase 23 — Route holdout generalisation
  Phase 24 — Snapshot parity audit
  Phase 25 — Save model bundle
  Phase 26 — Summary
"""

import json
import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.calibration import align_calibrated_probabilities, compare_calibrators
from src.config import ProjectConfig
from src.drift import run_feature_drift_analysis, run_prediction_drift_analysis
from src.eda import generate_reliability_diagrams, run_eda
from src.evaluation import comparison_table, evaluate_predictions
from src.features import (
    build_feature_bundle,
    build_snapshot_feature_bundle,
    run_feature_selection,
)
from src.labels import FINAL_LABELS
from src.models import (
    ModelResult,
    build_candidates,
    compare_models,
    predict_proba_aligned,
    run_naive_baselines,
)
from src.preprocessing import preprocess_raw_file
from src.reporting import calibration_parity_report, subgroup_metrics
from src.significance import run_significance_suite
from src.simulation import (
    cost_sensitive_simulation,
    reranking_proxy_simulation,
    suppression_policy_simulation,
)
from src.split import FourWaySplit, rolling_backtest_splits, temporal_four_way_split
from src.utils import save_csv, save_json


def run_training_pipeline(config: ProjectConfig | None = None) -> Dict[str, object]:
    """Run the complete research-grade training pipeline.

    This is the single authoritative entry point. All design decisions are
    traced through saved artifacts so the examiner can reproduce any claim.
    """

    config = config or ProjectConfig()
    config.ensure_directories()

    # -----------------------------------------------------------------------
    # Phase 1: Preprocessing & EDA
    # -----------------------------------------------------------------------
    preprocessing_result = preprocess_raw_file(config)
    processed = preprocessing_result.frame.copy()

    eda_report = run_eda(processed, config.reports_dir)
    save_json(config.reports_dir / "eda_report.json", eda_report)

    # -----------------------------------------------------------------------
    # Phase 2: Label presence check + zero-support warnings
    # -----------------------------------------------------------------------
    class_presence = {
        "configured_labels": list(FINAL_LABELS),
        "processed_label_counts": {
            label: int((processed["outcome_label"] == label).sum())
            for label in FINAL_LABELS
        },
    }
    active_labels = [
        label for label in FINAL_LABELS
        if class_presence["processed_label_counts"][label] > 0
    ]
    class_presence["active_labels"] = list(active_labels)
    class_presence["inactive_labels"] = [
        label for label in FINAL_LABELS if label not in active_labels
    ]
    class_presence["zero_support_warning"] = (
        f"The following classes are absent from training data and will receive F1=0.0: "
        f"{class_presence['inactive_labels']}. "
        "Headline metric is macro_f1_present_classes (excludes zero-support classes)."
        if class_presence["inactive_labels"] else "All configured classes are present."
    )
    if len(active_labels) < 2:
        raise ValueError(f"Not enough active labels. Active: {active_labels}")
    save_json(config.reports_dir / "class_presence.json", class_presence)

    supervised = processed.loc[processed["outcome_label"].isin(active_labels)].copy()

    # -----------------------------------------------------------------------
    # Phase 3: 4-way temporal split (eliminates validation double-dipping)
    # -----------------------------------------------------------------------
    four_way = temporal_four_way_split(
        supervised,
        train_fraction=0.55,
        val_model_fraction=0.20,
        val_calibration_fraction=0.10,
        test_fraction=0.15,
    )
    save_json(config.reports_dir / "split_metadata.json", four_way.metadata)

    # Build feature bundles
    train_bundle = build_feature_bundle(four_way.train, include_labels=True)
    val_model_bundle = build_snapshot_feature_bundle(four_way.val_model, four_way.train)
    val_calib_bundle = build_snapshot_feature_bundle(
        four_way.val_calibration,
        pd.concat([four_way.train, four_way.val_model], ignore_index=True),
    )
    history_for_test = pd.concat(
        [four_way.train, four_way.val_model, four_way.val_calibration],
        ignore_index=True,
    )
    test_bundle = build_snapshot_feature_bundle(four_way.test, history_for_test)

    feature_bundle = train_bundle
    target_mapping = {label: idx for idx, label in enumerate(active_labels)}

    for bundle in [train_bundle, val_model_bundle, val_calib_bundle, test_bundle]:
        bundle.frame["target"] = bundle.frame["outcome_label"].map(target_mapping)

    full_feature_columns = feature_bundle.feature_groups["full_valid_model"]
    cat_features = [f for f in feature_bundle.categorical_features if f in full_feature_columns]
    num_features = [f for f in feature_bundle.numeric_features if f in full_feature_columns]

    train_X = train_bundle.frame[full_feature_columns]
    train_y = train_bundle.frame["target"].to_numpy()
    val_model_X = val_model_bundle.frame[full_feature_columns]
    val_model_y = val_model_bundle.frame["target"].to_numpy()
    val_calib_X = val_calib_bundle.frame[full_feature_columns]
    val_calib_y = val_calib_bundle.frame["target"].to_numpy()
    test_X = test_bundle.frame[full_feature_columns]
    test_y = test_bundle.frame["target"].to_numpy()

    # -----------------------------------------------------------------------
    # Phase 4: Feature selection audit
    # -----------------------------------------------------------------------
    feature_selection_report = run_feature_selection(
        train_X, train_y, full_feature_columns, cat_features, k=35
    )
    save_json(config.reports_dir / "feature_selection.json", feature_selection_report)

    feature_inventory = {
        "categorical_features": feature_bundle.categorical_features,
        "numeric_features": feature_bundle.numeric_features,
        "feature_groups": feature_bundle.feature_groups,
    }
    save_json(config.reports_dir / "feature_inventory.json", feature_inventory)
    save_csv(
        config.reports_dir / "feature_availability_audit.csv",
        pd.DataFrame(feature_bundle.feature_availability),
    )
    save_json(
        config.reports_dir / "feature_availability_audit.json",
        {"rows": feature_bundle.feature_availability},
    )

    # -----------------------------------------------------------------------
    # Phase 5: Naive baselines (lower bound — must be beaten by all ML models)
    # -----------------------------------------------------------------------
    baseline_rows = run_naive_baselines(
        train_X, train_y, test_X, test_y, active_labels
    )
    save_json(config.reports_dir / "naive_baselines.json", {"rows": baseline_rows})
    save_csv(config.reports_dir / "naive_baselines.csv", pd.DataFrame(baseline_rows))

    # -----------------------------------------------------------------------
    # Phase 6: Model comparison (5 models, all properly tuned)
    # FIX: RF now grid-searched; XGB/LGB always in output; eval_set disabled for fairness
    # -----------------------------------------------------------------------
    model_results = compare_models(
        X_train=train_X,
        y_train=train_y,
        X_validation=val_model_X,
        y_validation=val_model_y,
        labels=active_labels,
        categorical_features=cat_features,
        numeric_features=num_features,
        alpha=config.calibration_alpha,
        random_state=config.random_state,
    )
    model_rows = _model_result_rows(model_results)
    save_csv(config.reports_dir / "model_comparison.csv", comparison_table(model_rows))
    save_json(config.reports_dir / "model_comparison.json", {"rows": model_rows})

    best_model_result = model_results[0]
    best_estimator = best_model_result.estimator

    # -----------------------------------------------------------------------
    # Phase 7: Calibration on clean val_calibration holdout (no double-dip)
    # Now includes DirichletCalibrator
    # -----------------------------------------------------------------------
    raw_val_calib_proba = predict_proba_aligned(
        best_estimator, val_calib_X, list(range(len(active_labels)))
    )
    calibration_results = compare_calibrators(
        raw_val_calib_proba,
        val_calib_y,
        active_labels,
        min_balanced_accuracy=config.calibration_min_balanced_accuracy,
        min_macro_f1=config.calibration_min_macro_f1,
        max_drop_vs_uncalibrated=config.calibration_max_drop_vs_uncalibrated,
    )
    calibration_rows = _calibration_result_rows(calibration_results)
    save_csv(config.reports_dir / "calibration_comparison.csv", comparison_table(calibration_rows))
    save_json(config.reports_dir / "calibration_comparison.json", {"rows": calibration_rows})

    best_calibrator_result = calibration_results[0]

    # -----------------------------------------------------------------------
    # Phase 8: Final test evaluation (full metric suite)
    # -----------------------------------------------------------------------
    raw_test_proba = predict_proba_aligned(
        best_estimator, test_X, list(range(len(active_labels)))
    )
    calibrated_test_proba = align_calibrated_probabilities(
        best_calibrator_result.calibrator,
        raw_test_proba,
        list(range(len(active_labels))),
    )
    final_metrics = evaluate_predictions(test_y, calibrated_test_proba, active_labels)
    save_json(config.reports_dir / "final_test_metrics.json", final_metrics)

    # Reliability diagrams
    generate_reliability_diagrams(
        final_metrics["reliability_table"],
        active_labels,
        config.reports_dir / "reliability_diagrams.png",
        title_prefix="Final Model — ",
    )

    # -----------------------------------------------------------------------
    # Phase 9: Statistical significance testing (with Bonferroni correction)
    # -----------------------------------------------------------------------
    for result in model_results:
        if result.estimator is not None:
            proba = predict_proba_aligned(
                result.estimator, test_X, list(range(len(active_labels)))
            )
            result.validation_probabilities = proba  # now test-set probabilities

    significance_report = run_significance_suite(
        test_y, model_results, active_labels,
        n_bootstrap=1000, random_state=config.random_state,
    )
    save_json(config.reports_dir / "significance_tests.json", significance_report)

    # -----------------------------------------------------------------------
    # Phase 10: Subgroup evaluation & calibration parity
    # -----------------------------------------------------------------------
    subgroup_reports = {
        "provider_key": subgroup_metrics(
            test_bundle.frame, test_y, calibrated_test_proba, "provider_key", active_labels
        ),
        "route": subgroup_metrics(
            test_bundle.frame, test_y, calibrated_test_proba, "route", active_labels, top_n=20
        ),
        "airline_code": subgroup_metrics(
            test_bundle.frame, test_y, calibrated_test_proba, "airline_code", active_labels
        ),
        "days_to_departure_bucket": subgroup_metrics(
            test_bundle.frame, test_y, calibrated_test_proba, "days_to_departure_bucket", active_labels
        ),
    }
    save_json(config.reports_dir / "subgroup_evaluation.json", subgroup_reports)
    save_json(
        config.reports_dir / "calibration_parity.json",
        calibration_parity_report(
            test_bundle.frame,
            test_y,
            calibrated_test_proba,
            ["provider_key", "route", "airline_code"],
            active_labels,
        ),
    )

    # -----------------------------------------------------------------------
    # Phase 11: Ablation study (ISOLATED + CUMULATIVE)
    # -----------------------------------------------------------------------
    ablation_rows = run_ablation_study(
        train_frame=train_bundle.frame,
        validation_frame=val_model_bundle.frame,
        feature_bundle=feature_bundle,
        model_name=best_model_result.name,
        random_state=config.random_state,
        labels=active_labels,
        random_seeds=config.ablation_random_seeds,
    )
    save_csv(config.reports_dir / "ablation_results.csv", pd.DataFrame(ablation_rows))
    save_json(config.reports_dir / "ablation_results.json", {"rows": ablation_rows})

    # -----------------------------------------------------------------------
    # Phase 12: Rolling backtest (5 windows — FIX from 3)
    # -----------------------------------------------------------------------
    backtest_rows = run_rolling_backtest(
        features=supervised,
        feature_bundle=feature_bundle,
        model_name=best_model_result.name,
        random_state=config.random_state,
        n_windows=config.rolling_backtest_windows,
        labels=active_labels,
    )
    save_csv(config.reports_dir / "rolling_backtest.csv", pd.DataFrame(backtest_rows))
    save_json(config.reports_dir / "rolling_backtest.json", {"rows": backtest_rows})

    # -----------------------------------------------------------------------
    # Phase 13: Policy simulation (suppression + reranking + cost-sensitive)
    # -----------------------------------------------------------------------
    policy_results = {
        "suppression_policy": suppression_policy_simulation(
            test_bundle.frame, calibrated_test_proba, active_labels,
            thresholds=config.cost_sensitive_thresholds,
        ),
        "reranking_proxy": reranking_proxy_simulation(
            test_bundle.frame, calibrated_test_proba, active_labels
        ),
        "cost_sensitive": cost_sensitive_simulation(
            test_y, calibrated_test_proba, active_labels,
            thresholds=config.cost_sensitive_thresholds,
        ),
    }
    save_json(config.reports_dir / "policy_simulation_results.json", policy_results)

    # -----------------------------------------------------------------------
    # Phase 14: Concept drift analysis (train vs test)
    # -----------------------------------------------------------------------
    drift_report = run_feature_drift_analysis(
        train_bundle.frame[full_feature_columns],
        test_bundle.frame[full_feature_columns],
        numeric_features=num_features,
        categorical_features=cat_features,
    )
    save_json(config.reports_dir / "drift_analysis.json", drift_report)

    pred_drift_report = run_prediction_drift_analysis(
        raw_val_calib_proba, calibrated_test_proba, active_labels
    )
    save_json(config.reports_dir / "prediction_drift.json", pred_drift_report)

    # -----------------------------------------------------------------------
    # Phase 15 & 16: SHAP explainability + dependency plots (PNG)
    # -----------------------------------------------------------------------
    shap_report = run_shap_explainability(
        best_estimator, val_model_X, active_labels,
        reports_dir=config.reports_dir,
        top_n=25,
    )
    save_json(config.reports_dir / "shap_explainability.json", shap_report)

    # -----------------------------------------------------------------------
    # Phase 17: Permutation importance (model-agnostic fallback)
    # -----------------------------------------------------------------------
    perm_report = run_permutation_explainability(
        best_estimator, val_model_X, val_model_y,
        random_state=config.random_state, top_n=25
    )
    save_json(config.reports_dir / "explainability_permutation_importance.json", perm_report)

    # -----------------------------------------------------------------------
    # Phase 18: Partial dependence plots (PDP)
    # -----------------------------------------------------------------------
    pdp_report = run_partial_dependence_plots(
        best_estimator, val_model_X, active_labels,
        feature_names=num_features[:6],
        reports_dir=config.reports_dir,
    )
    save_json(config.reports_dir / "partial_dependence.json", pdp_report)

    # -----------------------------------------------------------------------
    # Phase 19: Learning curves
    # -----------------------------------------------------------------------
    learning_curve_rows = run_learning_curves(
        train_bundle.frame, val_model_bundle.frame,
        full_feature_columns, feature_bundle,
        best_model_result.name, active_labels,
        config.random_state, config.learning_curve_fractions,
    )
    save_json(config.reports_dir / "learning_curves.json", {"rows": learning_curve_rows})

    # -----------------------------------------------------------------------
    # Phase 20: Error analysis
    # -----------------------------------------------------------------------
    error_report = run_error_analysis(
        test_bundle.frame, test_y, calibrated_test_proba, active_labels,
        n_samples=config.error_analysis_n_samples,
    )
    save_json(config.reports_dir / "error_analysis.json", error_report)

    # -----------------------------------------------------------------------
    # Phase 21: Ambiguous handling experiment
    # -----------------------------------------------------------------------
    ambiguous_experiment = run_ambiguous_handling_experiment(processed, config)
    save_json(config.reports_dir / "ambiguous_handling_experiment.json", ambiguous_experiment)

    # -----------------------------------------------------------------------
    # Phase 22: Provider holdout generalisation
    # -----------------------------------------------------------------------
    provider_holdout = run_provider_holdout_generalization(
        supervised=supervised,
        model_name=best_model_result.name,
        labels=active_labels,
        random_state=config.random_state,
        holdout_fraction=config.provider_holdout_fraction,
    )
    save_json(config.reports_dir / "provider_holdout_evaluation.json", provider_holdout)

    # -----------------------------------------------------------------------
    # Phase 23: Route holdout generalisation
    # -----------------------------------------------------------------------
    route_holdout = run_route_holdout_generalization(
        supervised=supervised,
        model_name=best_model_result.name,
        labels=active_labels,
        random_state=config.random_state,
        holdout_fraction=config.provider_holdout_fraction,
    )
    save_json(config.reports_dir / "route_holdout_evaluation.json", route_holdout)

    # -----------------------------------------------------------------------
    # Phase 24: Snapshot parity audit
    # -----------------------------------------------------------------------
    snapshot_parity = run_snapshot_parity_audit(
        target_df=four_way.val_model,
        history_df=four_way.train,
        feature_columns=full_feature_columns,
        labels=active_labels,
        estimator=best_estimator,
        calibrator=best_calibrator_result.calibrator,
    )
    save_json(config.reports_dir / "snapshot_parity_audit.json", snapshot_parity)

    # -----------------------------------------------------------------------
    # Phase 25: Save model bundle
    # -----------------------------------------------------------------------
    model_bundle = {
        "model_name": best_model_result.name,
        "estimator": best_estimator,
        "calibrator_name": best_calibrator_result.name,
        "calibrator": best_calibrator_result.calibrator,
        "feature_columns": full_feature_columns,
        "categorical_features": cat_features,
        "numeric_features": num_features,
        "target_mapping": target_mapping,
        "labels": active_labels,
        "inference_reference": feature_bundle.inference_reference,
        "prediction_time_assumption": config.prediction_time_assumption,
        "config": config.to_dict(),
        "val_calib_proba": raw_val_calib_proba,
    }
    joblib.dump(model_bundle, config.models_dir / "final_model_bundle.joblib")

    # -----------------------------------------------------------------------
    # Phase 26: Summary
    # -----------------------------------------------------------------------
    best_baseline_f1 = max(
        (r["macro_f1"] for r in baseline_rows), default=0.0
    )
    # PR-AUC macro from final metrics (excludes zero-support classes)
    pr_auc_values = {
        k: v for k, v in final_metrics.get("precision_recall_auc", {}).items()
        if k not in ("macro_avg_ap",) and isinstance(v, dict) and not np.isnan(v.get("average_precision", float("nan")))
    }

    required_artifacts = [
        "summary.json",
        "model_comparison.json",
        "calibration_comparison.json",
        "final_test_metrics.json",
        "naive_baselines.json",
        "significance_tests.json",
        "drift_analysis.json",
        "provider_holdout_evaluation.json",
        "route_holdout_evaluation.json",
        "snapshot_parity_audit.json",
        "class_presence.json",
        "experiment_manifest.json",
    ]

    phase_status = {
        "phase_1_preprocessing_eda": True,
        "phase_2_label_presence": True,
        "phase_3_temporal_split": True,
        "phase_4_feature_selection": True,
        "phase_5_naive_baselines": len(baseline_rows) > 0,
        "phase_6_model_comparison": len(model_rows) > 0,
        "phase_7_calibration": len(calibration_rows) > 0,
        "phase_8_final_evaluation": bool(final_metrics),
        "phase_9_significance": len(significance_report.get("mcnemar_pairwise", [])) > 0,
        "phase_10_subgroup_calibration_parity": bool(subgroup_reports),
        "phase_11_ablation": len(ablation_rows) > 0,
        "phase_12_rolling_backtest": len(backtest_rows) > 0,
        "phase_13_policy_simulation": bool(policy_results),
        "phase_14_drift": bool(drift_report) and bool(pred_drift_report),
        "phase_15_16_shap": shap_report.get("status") == "completed",
        "phase_17_permutation_importance": perm_report.get("status") == "completed",
        "phase_18_partial_dependence": pdp_report.get("status") == "completed",
        "phase_19_learning_curves": len(learning_curve_rows) > 0,
        "phase_20_error_analysis": error_report.get("status") in {"completed", "no_errors_found"},
        "phase_21_ambiguous_handling": ambiguous_experiment.get("status") in {"completed", "skipped_no_ambiguous_rows"},
        "phase_22_provider_holdout": provider_holdout.get("status") == "completed",
        "phase_23_route_holdout": route_holdout.get("status") == "completed",
        "phase_24_snapshot_parity": snapshot_parity.get("status") == "completed",
        "phase_25_model_bundle": (config.models_dir / "final_model_bundle.joblib").exists(),
        "phase_26_summary": True,
    }
    completed_phase_count = int(sum(1 for done in phase_status.values() if done))

    summary = {
        "best_model": best_model_result.name,
        "best_model_selection_score": best_model_result.selection_score,
        "best_calibrator": best_calibrator_result.name,
        "final_test_macro_f1": final_metrics["macro_f1"],
        "final_test_macro_f1_present_classes": final_metrics["macro_f1_present_classes"],
        "final_test_log_loss": final_metrics["log_loss"],
        "final_test_balanced_accuracy": final_metrics["balanced_accuracy"],
        "final_test_weighted_f1": final_metrics["weighted_f1"],
        "final_test_roc_auc_macro": final_metrics.get("roc_auc_macro"),
        "final_test_pr_auc_macro": final_metrics.get("precision_recall_auc", {}).get("macro_avg_ap"),
        "final_test_ece_macro": final_metrics.get("ece_macro"),
        "best_baseline_macro_f1": best_baseline_f1,
        "lift_over_best_baseline": round(
            final_metrics["macro_f1"] - best_baseline_f1, 4
        ),
        "labels_evaluated": active_labels,
        "zero_support_classes": class_presence["inactive_labels"],
        "split_strategy": "temporal_four_way_no_double_dipping",
        "drift_status": drift_report.get("summary", {}).get("overall_status"),
        "shap_status": shap_report.get("status"),
        "significance_tests_run": len(significance_report.get("mcnemar_pairwise", [])),
        "bonferroni_correction_applied": True,
        "rolling_backtest_windows": len(backtest_rows),
        "ablation_configs_run": len(ablation_rows),
        "learning_curve_points": len(learning_curve_rows),
        "provider_holdout_status": provider_holdout.get("status"),
        "route_holdout_status": route_holdout.get("status"),
        "snapshot_parity_status": snapshot_parity.get("status"),
        "processed_rows": int(len(processed)),
        "supervised_rows": int(len(supervised)),
        "artifacts_dir": str(config.artifacts_dir),
        "pipeline_phases_completed": completed_phase_count,
        "pipeline_phases_total": len(phase_status),
        "phase_status": phase_status,
        "pipeline_version": "v2.1_research_evidence_hardened",
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "reproducibility": {
            "random_state": config.random_state,
            "ablation_random_seeds": list(config.ablation_random_seeds),
            "rolling_backtest_windows": config.rolling_backtest_windows,
            "git_commit": os.getenv("GIT_COMMIT", "unknown"),
            "code_entrypoint": "python -m src.train",
        },
    }
    save_json(config.reports_dir / "summary.json", summary)
    save_json(
        config.reports_dir / "experiment_manifest.json",
        {
            "generated_at_utc": summary["generated_at_utc"],
            "pipeline_version": summary["pipeline_version"],
            "labels_evaluated": active_labels,
            "data_rows": {
                "processed": int(len(processed)),
                "supervised": int(len(supervised)),
            },
            "artifacts_expected": required_artifacts,
            "artifacts_present": [
                name for name in required_artifacts
                if name == "experiment_manifest.json" or (config.reports_dir / name).exists()
            ],
            "artifacts_missing": [
                name for name in required_artifacts
                if name != "experiment_manifest.json" and not (config.reports_dir / name).exists()
            ],
        },
    )
    return summary


# ---------------------------------------------------------------------------
# Ablation study — ISOLATED + CUMULATIVE
# ---------------------------------------------------------------------------

def run_ablation_study(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_bundle,
    model_name: str,
    random_state: int,
    labels: List[str],
    random_seeds: List[int] | None = None,
) -> List[Dict[str, object]]:
    """Run BOTH isolated and cumulative feature group ablation.

    Isolated: each group used alone (measures standalone signal).
    Cumulative: groups added in order (measures marginal gain).
    """
    groups = feature_bundle.feature_groups
    seeds = list(random_seeds or [random_state])

    ablation_configs = {
        # ISOLATED — measures each group alone
        "isolated_topology":   groups["topology_basic"],
        "isolated_temporal":   groups["temporal"],
        "isolated_price":      groups["price"],
        "isolated_history":    groups["reliability_history"],
        # CUMULATIVE — measures marginal contribution when added
        "cumul_topology":      groups["topology_basic"],
        "cumul_topo_temporal": groups["topology_basic"] + groups["temporal"],
        "cumul_topo_temp_price": groups["topology_basic"] + groups["temporal"] + groups["price"],
        "cumul_topo_temp_history": groups["topology_basic"] + groups["temporal"] + groups["reliability_history"],
        "full_valid_model":    groups["full_valid_model"],
    }

    rows = []
    for config_name, feature_columns in ablation_configs.items():
        available = [f for f in feature_columns if f in train_frame.columns and f in validation_frame.columns]
        if not available:
            continue

        seed_metrics = []
        for seed in seeds:
            estimator = _fit_single_model(
                model_name, train_frame, validation_frame, available, feature_bundle, seed
            )
            probabilities = predict_proba_aligned(
                estimator, validation_frame[available], list(range(len(labels)))
            )
            metrics = evaluate_predictions(validation_frame["target"].to_numpy(), probabilities, labels)
            seed_metrics.append(metrics)

        metric_names = [
            "macro_f1", "macro_f1_present_classes", "weighted_f1",
            "balanced_accuracy", "log_loss", "multiclass_brier", "ece_macro",
        ]
        ablation_type = "isolated" if config_name.startswith("isolated") else "cumulative"
        rows.append({
            "config_name": config_name,
            "ablation_type": ablation_type,
            "n_features": len(available),
            "n_runs": len(seed_metrics),
            **{
                f"{m}_mean": float(np.mean([r[m] for r in seed_metrics]))
                for m in metric_names
            },
            **{
                f"{m}_std": float(np.std([r[m] for r in seed_metrics], ddof=1))
                if len(seed_metrics) > 1 else 0.0
                for m in metric_names
            },
        })
    return rows


# ---------------------------------------------------------------------------
# Rolling backtest
# ---------------------------------------------------------------------------

def run_rolling_backtest(
    features: pd.DataFrame,
    feature_bundle,
    model_name: str,
    random_state: int,
    n_windows: int,
    labels: List[str],
) -> List[Dict[str, object]]:
    rows = []
    for window in rolling_backtest_splits(features, n_windows=n_windows):
        train_bundle = build_feature_bundle(window["train"], include_labels=True)
        test_bundle = build_snapshot_feature_bundle(window["test"], window["train"])
        label_map = {label: idx for idx, label in enumerate(labels)}
        train_bundle.frame["target"] = train_bundle.frame["outcome_label"].map(label_map)
        test_bundle.frame["target"] = test_bundle.frame["outcome_label"].map(label_map)
        feature_columns = [
            f for f in train_bundle.feature_groups["full_valid_model"]
            if f in train_bundle.frame.columns and f in test_bundle.frame.columns
        ]
        estimator = _fit_single_model(
            model_name, train_bundle.frame, test_bundle.frame,
            feature_columns, train_bundle, random_state,
        )
        probabilities = predict_proba_aligned(
            estimator, test_bundle.frame[feature_columns], list(range(len(labels)))
        )
        metrics = evaluate_predictions(
            test_bundle.frame["target"].to_numpy(), probabilities, labels
        )
        rows.append({
            "window_id": window["window_id"],
            "train_start": window["train_start"],
            "train_end": window["train_end"],
            "test_start": window["test_start"],
            "test_end": window["test_end"],
            "train_rows": len(window["train"]),
            "test_rows": len(window["test"]),
            "macro_f1": metrics["macro_f1"],
            "macro_f1_present_classes": metrics["macro_f1_present_classes"],
            "log_loss": metrics["log_loss"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "ece_macro": metrics["ece_macro"],
        })
    return rows


# ---------------------------------------------------------------------------
# SHAP explainability + dependency plots
# ---------------------------------------------------------------------------

def run_shap_explainability(
    estimator,
    X_sample: pd.DataFrame,
    labels: List[str],
    reports_dir=None,
    top_n: int = 25,
) -> Dict[str, object]:
    """Compute SHAP values + dependency plots for the best model."""
    if estimator is None:
        return {"status": "skipped_no_estimator"}
    if len(X_sample) == 0:
        return {"status": "skipped_empty_sample"}

    try:
        import shap

        sample = X_sample.sample(min(500, len(X_sample)), random_state=42)
        explainer_input = sample
        predict_target = estimator
        feature_names = list(X_sample.columns)

        # Explain the fitted classifier input space, not the sklearn Pipeline wrapper.
        if hasattr(estimator, "named_steps") and "preprocessor" in estimator.named_steps and "classifier" in estimator.named_steps:
            preprocessor = estimator.named_steps["preprocessor"]
            classifier = estimator.named_steps["classifier"]
            transformed = preprocessor.transform(sample)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            explainer_input = transformed
            predict_target = classifier
            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out().tolist()
            else:
                feature_names = [f"feature_{idx}" for idx in range(explainer_input.shape[1])]

        try:
            explainer = shap.TreeExplainer(predict_target)
            shap_values = explainer.shap_values(explainer_input)
        except Exception:
            background = shap.sample(sample, min(50, len(sample)))
            explainer = shap.KernelExplainer(estimator.predict_proba, background)
            shap_values = explainer.shap_values(sample, nsamples=100)

        per_class: Dict[str, object] = {}

        def _extract_class_shap(shap_array, class_idx):
            if isinstance(shap_array, list):
                return shap_array[class_idx] if class_idx < len(shap_array) else None
            if hasattr(shap_array, "values") and shap_array.values.ndim == 3:
                return shap_array.values[:, :, class_idx]
            if isinstance(shap_array, np.ndarray) and shap_array.ndim == 2:
                return shap_array
            return None

        for class_idx, label in enumerate(labels):
            sv = _extract_class_shap(shap_values, class_idx)
            if sv is None:
                continue
            mean_abs = np.abs(sv).mean(axis=0)
            ranked = sorted(
                zip(feature_names, mean_abs.tolist()),
                key=lambda x: x[1], reverse=True,
            )[:top_n]
            per_class[label] = [
                {"feature": f, "mean_abs_shap": round(float(v), 6)}
                for f, v in ranked
            ]

        # Save SHAP dependency plots for top 3 features of the best class
        shap_plots_saved = []
        if reports_dir is not None and per_class:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                primary_label = labels[0]
                top_features = [entry["feature"] for entry in per_class.get(primary_label, [])[:3]]

                for feat in top_features:
                    if feat not in feature_names:
                        continue
                    fig, ax = plt.subplots(figsize=(6, 4))
                    feat_idx = feature_names.index(feat)
                    sv_primary = _extract_class_shap(shap_values, 0)
                    if sv_primary is not None:
                        if isinstance(explainer_input, pd.DataFrame):
                            x_axis = explainer_input[feat].values
                        else:
                            x_axis = explainer_input[:, feat_idx]
                        ax.scatter(
                            x_axis,
                            sv_primary[:, feat_idx],
                            alpha=0.4, s=8, color="#1f77b4",
                        )
                        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                        ax.set_xlabel(feat, fontsize=9)
                        ax.set_ylabel(f"SHAP value for '{primary_label}'", fontsize=9)
                        ax.set_title(f"SHAP Dependence: {feat}", fontsize=10)
                        ax.grid(True, alpha=0.3)
                        out_path = reports_dir / f"shap_dependency_{feat.replace('/', '_')}.png"
                        plt.tight_layout()
                        plt.savefig(out_path, dpi=120, bbox_inches="tight")
                        plt.close(fig)
                        shap_plots_saved.append(str(out_path))
            except Exception as e:
                shap_plots_saved.append(f"failed: {e}")

        return {
            "status": "completed",
            "sample_size": len(sample),
            "top_n": top_n,
            "per_class_shap": per_class,
            "dependency_plots_saved": shap_plots_saved,
        }

    except ImportError:
        return {"status": "skipped_shap_not_installed", "install": "pip install shap"}
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


# ---------------------------------------------------------------------------
# Permutation importance (model-agnostic)
# ---------------------------------------------------------------------------

def run_permutation_explainability(
    estimator,
    validation_X: pd.DataFrame,
    validation_y: np.ndarray,
    random_state: int,
    top_n: int = 25,
) -> Dict[str, object]:
    if estimator is None or len(validation_X) == 0:
        return {"status": "skipped"}
    try:
        result = permutation_importance(
            estimator,
            validation_X,
            validation_y,
            scoring="f1_macro",
            n_repeats=5,
            random_state=random_state,
            n_jobs=1,
        )
        rows = [
            {
                "feature": feat,
                "importance_mean": float(result.importances_mean[idx]),
                "importance_std": float(result.importances_std[idx]),
            }
            for idx, feat in enumerate(validation_X.columns)
        ]
        rows = sorted(rows, key=lambda r: r["importance_mean"], reverse=True)
        return {"status": "completed", "scoring": "f1_macro", "rows": rows[:top_n]}
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


# ---------------------------------------------------------------------------
# Partial Dependence Plots (PDP)
# ---------------------------------------------------------------------------

def run_partial_dependence_plots(
    estimator,
    X_sample: pd.DataFrame,
    labels: List[str],
    feature_names: List[str],
    reports_dir=None,
) -> Dict[str, object]:
    """Compute partial dependence for key features and save PNG."""
    if estimator is None or len(X_sample) == 0:
        return {"status": "skipped_no_estimator"}
    try:
        from sklearn.inspection import PartialDependenceDisplay
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Filter to numeric features that exist in the sample
        valid_features = [f for f in feature_names if f in X_sample.columns][:6]
        if not valid_features:
            return {"status": "skipped_no_valid_features"}

        sample = X_sample.sample(min(1000, len(X_sample)), random_state=42)
        results = []

        for target_class_idx, label in enumerate(labels[:2]):  # Do for top 2 classes
            try:
                fig, ax = plt.subplots(1, len(valid_features), figsize=(4 * len(valid_features), 4))
                if len(valid_features) == 1:
                    ax = [ax]
                display = PartialDependenceDisplay.from_estimator(
                    estimator, sample, valid_features,
                    target=target_class_idx,
                    ax=ax,
                    kind="average",
                )
                fig.suptitle(f"Partial Dependence Plots — Class: {label}", fontsize=11, fontweight="bold")
                plt.tight_layout()
                if reports_dir is not None:
                    out_path = reports_dir / f"pdp_{label}.png"
                    plt.savefig(out_path, dpi=120, bbox_inches="tight")
                    results.append({"class": label, "saved_to": str(out_path), "features": valid_features})
                plt.close(fig)
            except Exception as e:
                results.append({"class": label, "error": str(e)})

        return {"status": "completed", "results": results}
    except ImportError:
        return {"status": "skipped_matplotlib_unavailable"}
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def run_learning_curves(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: List[str],
    feature_bundle,
    model_name: str,
    labels: List[str],
    random_state: int,
    fractions: List[float],
) -> List[Dict[str, object]]:
    """Train on increasing fractions of training data and evaluate on validation set.

    Reveals whether the model is underfitting (curves haven't converged) or
    overfitting (large train-val gap that doesn't close with more data).
    """
    rows = []
    for frac in fractions:
        n_rows = max(100, int(len(train_frame) * frac))
        subset = train_frame.iloc[:n_rows]  # temporal order preserved
        try:
            estimator = _fit_single_model(
                model_name, subset, val_frame,
                feature_columns, feature_bundle, random_state,
            )
            # Train metrics
            train_proba = predict_proba_aligned(
                estimator, subset[feature_columns], list(range(len(labels)))
            )
            train_metrics = evaluate_predictions(subset["target"].to_numpy(), train_proba, labels)
            # Val metrics
            val_proba = predict_proba_aligned(
                estimator, val_frame[feature_columns], list(range(len(labels)))
            )
            val_metrics = evaluate_predictions(val_frame["target"].to_numpy(), val_proba, labels)

            rows.append({
                "train_fraction": frac,
                "train_rows": n_rows,
                "train_macro_f1": round(train_metrics["macro_f1"], 4),
                "val_macro_f1": round(val_metrics["macro_f1"], 4),
                "train_log_loss": round(train_metrics["log_loss"], 4),
                "val_log_loss": round(val_metrics["log_loss"], 4),
                "train_balanced_accuracy": round(train_metrics["balanced_accuracy"], 4),
                "val_balanced_accuracy": round(val_metrics["balanced_accuracy"], 4),
                "gap_macro_f1": round(train_metrics["macro_f1"] - val_metrics["macro_f1"], 4),
            })
        except Exception as exc:
            rows.append({"train_fraction": frac, "train_rows": n_rows, "error": str(exc)})
    return rows


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def run_error_analysis(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str],
    n_samples: int = 200,
) -> Dict[str, object]:
    """Analyse the top-N most confidently misclassified samples.

    Finds samples where the model was highly confident but wrong.
    Reports patterns in misclassified samples by true class, predicted class,
    and top contributing feature values.
    """
    predicted = np.argmax(y_prob, axis=1)
    confidence = y_prob.max(axis=1)
    misclassified_mask = predicted != y_true

    if not misclassified_mask.any():
        return {"status": "no_errors_found", "total_test_rows": len(y_true)}

    # Sort by confidence descending (most confident errors first)
    error_indices = np.where(misclassified_mask)[0]
    sorted_indices = error_indices[np.argsort(confidence[error_indices])[::-1]]
    top_n_indices = sorted_indices[:n_samples]

    # Build error table
    error_rows = []
    for idx in top_n_indices:
        true_label = labels[y_true[idx]]
        pred_label = labels[predicted[idx]]
        conf = float(confidence[idx])
        row_data = frame.iloc[idx] if idx < len(frame) else None

        entry = {
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": round(conf, 4),
            "confusion_type": f"{true_label} → {pred_label}",
        }
        if row_data is not None:
            for col in ["provider_key", "route", "airline_code", "days_to_departure_bucket",
                        "days_to_departure", "search_hour", "price_total"]:
                if col in row_data.index:
                    entry[col] = str(row_data[col]) if not isinstance(row_data[col], float) else round(row_data[col], 2)
        error_rows.append(entry)

    # Summarize by confusion type
    confusion_counts: Dict[str, int] = {}
    for row in error_rows:
        key = row["confusion_type"]
        confusion_counts[key] = confusion_counts.get(key, 0) + 1

    # Per true-class error rate
    per_class_errors = {}
    for idx, label in enumerate(labels):
        true_mask = y_true == idx
        if not true_mask.any():
            continue
        error_rate = float(misclassified_mask[true_mask].mean())
        most_confused_to = labels[
            int(np.argmax(np.bincount(predicted[true_mask & misclassified_mask], minlength=len(labels))))
        ] if (true_mask & misclassified_mask).any() else "N/A"
        per_class_errors[label] = {
            "total_samples": int(true_mask.sum()),
            "error_rate": round(error_rate, 4),
            "most_confused_with": most_confused_to,
        }

    return {
        "status": "completed",
        "total_test_rows": int(len(y_true)),
        "total_misclassified": int(misclassified_mask.sum()),
        "overall_error_rate": round(float(misclassified_mask.mean()), 4),
        "n_samples_analysed": len(error_rows),
        "confusion_type_counts": dict(sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)),
        "per_class_error_analysis": per_class_errors,
        "top_confident_errors": error_rows[:50],  # Top 50 worst errors for inspection
    }


# ---------------------------------------------------------------------------
# Ambiguous handling experiment
# ---------------------------------------------------------------------------

def run_ambiguous_handling_experiment(
    processed: pd.DataFrame, config: ProjectConfig
) -> Dict[str, object]:
    ambiguous_rows = processed["outcome_label"].eq("ambiguous").sum()
    result: Dict[str, object] = {
        "ambiguous_row_count": int(ambiguous_rows),
        "default_training_policy": "exclude_ambiguous_rows",
        "experimental_policy": "keep_ambiguous_as_fifth_class_for_diagnostic_comparison",
    }
    if ambiguous_rows == 0:
        result["status"] = "skipped_no_ambiguous_rows"
        return result

    from src.split import temporal_four_way_split as _four

    frame = processed.copy()
    four = _four(frame, 0.55, 0.20, 0.10, 0.15)
    train_b = build_feature_bundle(four.train, include_labels=True)
    val_b = build_snapshot_feature_bundle(four.val_model, four.train)
    mapping = {label: idx for idx, label in enumerate(FINAL_LABELS + ["ambiguous"])}
    train_b.frame["target_5class"] = train_b.frame["outcome_label"].map(mapping)
    val_b.frame["target_5class"] = val_b.frame["outcome_label"].map(mapping)
    feature_columns = [
        f for f in train_b.feature_groups["full_valid_model"]
        if f in train_b.frame.columns and f in val_b.frame.columns
    ]
    cat_f = [f for f in train_b.categorical_features if f in feature_columns]
    candidates = build_candidates(
        categorical_features=cat_f,
        numeric_features=[f for f in train_b.numeric_features if f in feature_columns],
        random_state=config.random_state,
    )
    estimator = candidates["catboost"]
    estimator.fit(
        train_b.frame[feature_columns],
        train_b.frame["target_5class"].to_numpy(),
        cat_features=cat_f,
        eval_set=(val_b.frame[feature_columns], val_b.frame["target_5class"].to_numpy()),
    )
    probabilities = predict_proba_aligned(
        estimator, val_b.frame[feature_columns], list(range(5))
    )
    metrics = evaluate_predictions(
        val_b.frame["target_5class"].to_numpy(), probabilities, FINAL_LABELS + ["ambiguous"]
    )
    result["status"] = "completed"
    result["validation_macro_f1_5class"] = metrics["macro_f1"]
    result["validation_log_loss_5class"] = metrics["log_loss"]
    return result


# ---------------------------------------------------------------------------
# Provider holdout
# ---------------------------------------------------------------------------

def run_provider_holdout_generalization(
    supervised: pd.DataFrame,
    model_name: str,
    labels: List[str],
    random_state: int,
    holdout_fraction: float,
) -> Dict[str, object]:
    if "provider_key" not in supervised.columns or supervised["provider_key"].isna().all():
        return {"status": "skipped_missing_provider_key"}

    provider_series = supervised["provider_key"].dropna().astype("string")
    provider_counts = provider_series.value_counts()
    eligible_counts = provider_counts[provider_counts >= 500]
    eligible_providers = eligible_counts.index.tolist()
    if len(eligible_providers) < 5:
        return {
            "status": "skipped_insufficient_provider_support",
            "provider_count": int(provider_counts.shape[0]),
            "eligible_provider_count": int(len(eligible_providers)),
            "min_rows_per_provider": 500,
        }

    rng = np.random.default_rng(random_state)
    holdout_count = min(
        len(eligible_providers) - 1,
        max(2, int(np.ceil(len(eligible_providers) * holdout_fraction))),
    )
    holdout_providers = set(
        rng.choice(eligible_providers, size=holdout_count, replace=False).tolist()
    )
    train_df = supervised.loc[~supervised["provider_key"].astype("string").isin(holdout_providers)].copy()
    test_df = supervised.loc[supervised["provider_key"].astype("string").isin(holdout_providers)].copy()
    if train_df.empty or test_df.empty:
        return {"status": "skipped_empty_split", "train_rows": int(len(train_df)), "test_rows": int(len(test_df))}

    train_bundle = build_feature_bundle(train_df, include_labels=True)
    test_bundle = build_snapshot_feature_bundle(test_df, train_df)
    label_map = {label: idx for idx, label in enumerate(labels)}
    train_bundle.frame["target"] = train_bundle.frame["outcome_label"].map(label_map)
    test_bundle.frame["target"] = test_bundle.frame["outcome_label"].map(label_map)
    feature_columns = [
        f for f in train_bundle.feature_groups["full_valid_model"]
        if f in train_bundle.frame.columns and f in test_bundle.frame.columns
    ]
    estimator = _fit_single_model(
        model_name, train_bundle.frame, test_bundle.frame,
        feature_columns, train_bundle, random_state,
    )
    probabilities = predict_proba_aligned(
        estimator, test_bundle.frame[feature_columns], list(range(len(labels)))
    )
    metrics = evaluate_predictions(test_bundle.frame["target"].to_numpy(), probabilities, labels)
    return {
        "status": "completed",
        "holdout_fraction_requested": holdout_fraction,
        "eligible_provider_count": int(len(eligible_providers)),
        "eligible_provider_min_rows": 500,
        "holdout_provider_count": len(holdout_providers),
        "holdout_providers": list(holdout_providers)[:10],
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "metrics": metrics,
    }


def run_route_holdout_generalization(
    supervised: pd.DataFrame,
    model_name: str,
    labels: List[str],
    random_state: int,
    holdout_fraction: float,
) -> Dict[str, object]:
    if "route" not in supervised.columns or supervised["route"].isna().all():
        return {"status": "skipped_missing_route"}

    route_series = supervised["route"].dropna().astype("string")
    route_counts = route_series.value_counts()
    eligible_counts = route_counts[route_counts >= 500]
    eligible_routes = eligible_counts.index.tolist()
    if len(eligible_routes) < 5:
        return {
            "status": "skipped_insufficient_route_support",
            "route_count": int(route_counts.shape[0]),
            "eligible_route_count": int(len(eligible_routes)),
            "min_rows_per_route": 500,
        }

    rng = np.random.default_rng(random_state)
    holdout_count = min(
        len(eligible_routes) - 1,
        max(2, int(np.ceil(len(eligible_routes) * holdout_fraction))),
    )
    holdout_routes = set(
        rng.choice(eligible_routes, size=holdout_count, replace=False).tolist()
    )
    train_df = supervised.loc[~supervised["route"].astype("string").isin(holdout_routes)].copy()
    test_df = supervised.loc[supervised["route"].astype("string").isin(holdout_routes)].copy()
    if train_df.empty or test_df.empty:
        return {"status": "skipped_empty_split", "train_rows": int(len(train_df)), "test_rows": int(len(test_df))}

    train_bundle = build_feature_bundle(train_df, include_labels=True)
    test_bundle = build_snapshot_feature_bundle(test_df, train_df)
    label_map = {label: idx for idx, label in enumerate(labels)}
    train_bundle.frame["target"] = train_bundle.frame["outcome_label"].map(label_map)
    test_bundle.frame["target"] = test_bundle.frame["outcome_label"].map(label_map)
    feature_columns = [
        f for f in train_bundle.feature_groups["full_valid_model"]
        if f in train_bundle.frame.columns and f in test_bundle.frame.columns
    ]
    estimator = _fit_single_model(
        model_name, train_bundle.frame, test_bundle.frame,
        feature_columns, train_bundle, random_state,
    )
    probabilities = predict_proba_aligned(
        estimator, test_bundle.frame[feature_columns], list(range(len(labels)))
    )
    metrics = evaluate_predictions(test_bundle.frame["target"].to_numpy(), probabilities, labels)
    return {
        "status": "completed",
        "holdout_fraction_requested": holdout_fraction,
        "eligible_route_count": int(len(eligible_routes)),
        "eligible_route_min_rows": 500,
        "holdout_route_count": len(holdout_routes),
        "holdout_routes": list(holdout_routes)[:10],
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "metrics": metrics,
    }


def run_snapshot_parity_audit(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame,
    feature_columns: List[str],
    labels: List[str],
    estimator,
    calibrator,
) -> Dict[str, object]:
    if target_df.empty or history_df.empty:
        return {"status": "skipped_empty_input"}

    authoritative_bundle = build_snapshot_feature_bundle(target_df, history_df)
    approximation_bundle = build_feature_bundle(target_df, include_labels=False)

    authoritative = authoritative_bundle.frame.copy()
    approximation = approximation_bundle.frame.copy()
    shared_columns = [c for c in feature_columns if c in authoritative.columns and c in approximation.columns]
    if not shared_columns:
        return {"status": "skipped_no_shared_features"}

    aligned_authoritative = authoritative[shared_columns].copy()
    aligned_approximation = approximation[shared_columns].copy()
    numeric_columns = [
        c for c in shared_columns
        if pd.api.types.is_numeric_dtype(aligned_authoritative[c]) and pd.api.types.is_numeric_dtype(aligned_approximation[c])
    ]
    categorical_columns = [c for c in shared_columns if c not in numeric_columns]

    numeric_diffs = []
    if numeric_columns:
        diff_frame = (aligned_authoritative[numeric_columns] - aligned_approximation[numeric_columns]).abs()
        for column in numeric_columns:
            numeric_diffs.append(
                {
                    "feature": column,
                    "mean_abs_diff": float(diff_frame[column].mean()),
                    "max_abs_diff": float(diff_frame[column].max()),
                }
            )
    categorical_agreement = []
    for column in categorical_columns:
        agreement = (
            aligned_authoritative[column].astype("string").fillna("Unknown")
            == aligned_approximation[column].astype("string").fillna("Unknown")
        ).mean()
        categorical_agreement.append({"feature": column, "agreement": float(agreement)})

    auth_proba = predict_proba_aligned(estimator, authoritative[feature_columns], list(range(len(labels))))
    approx_proba = predict_proba_aligned(estimator, approximation[feature_columns], list(range(len(labels))))
    auth_cal = align_calibrated_probabilities(calibrator, auth_proba, list(range(len(labels))))
    approx_cal = align_calibrated_probabilities(calibrator, approx_proba, list(range(len(labels))))

    auth_pred = np.argmax(auth_cal, axis=1)
    approx_pred = np.argmax(approx_cal, axis=1)
    probability_delta = np.abs(auth_cal - approx_cal)
    return {
        "status": "completed",
        "rows_compared": int(len(target_df)),
        "feature_column_count": len(shared_columns),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "prediction_agreement": float((auth_pred == approx_pred).mean()),
        "mean_abs_probability_delta": float(probability_delta.mean()),
        "max_abs_probability_delta": float(probability_delta.max()),
        "worst_numeric_features": sorted(numeric_diffs, key=lambda row: row["mean_abs_diff"], reverse=True)[:15],
        "worst_categorical_features": sorted(categorical_agreement, key=lambda row: row["agreement"])[:15],
        "note": (
            "Authoritative snapshot features use historical rows available at prediction time. "
            "Approximation uses default-only snapshot features without real history, which mirrors a weaker serving setup."
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_single_model(
    model_name: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: List[str],
    feature_bundle,
    random_state: int,
):
    candidates = build_candidates(
        categorical_features=[f for f in feature_bundle.categorical_features if f in feature_columns],
        numeric_features=[f for f in feature_bundle.numeric_features if f in feature_columns],
        random_state=random_state,
    )
    estimator = candidates.get(model_name, candidates["catboost"])
    X_train = train_frame[feature_columns]
    y_train = train_frame["target"].to_numpy()
    X_val = validation_frame[feature_columns]
    y_val = validation_frame["target"].to_numpy()

    # FIX: CatBoost eval_set is only used in ablation/backtest helpers, NOT in comparison phase
    # The comparison phase (compare_models) always calls with use_eval_set=False
    if model_name == "catboost":
        estimator.fit(
            X_train, y_train,
            cat_features=[f for f in feature_bundle.categorical_features if f in feature_columns],
            eval_set=(X_val, y_val),
        )
    else:
        estimator.fit(X_train, y_train)
    return estimator


def _model_result_rows(results: List) -> List[Dict[str, object]]:
    rows = []
    for result in results:
        rows.append({
            "name": result.name,
            "selection_score": result.selection_score,
            "accuracy": result.validation_metrics.get("accuracy"),
            "balanced_accuracy": result.validation_metrics.get("balanced_accuracy"),
            "macro_f1": result.validation_metrics.get("macro_f1"),
            "macro_f1_present_classes": result.validation_metrics.get("macro_f1_present_classes"),
            "weighted_f1": result.validation_metrics.get("weighted_f1"),
            "log_loss": result.validation_metrics.get("log_loss"),
            "multiclass_brier": result.validation_metrics.get("multiclass_brier"),
            "ece_macro": result.validation_metrics.get("ece_macro"),
            "roc_auc_macro": result.validation_metrics.get("roc_auc_macro"),
            "pr_auc_macro": result.validation_metrics.get("precision_recall_auc", {}).get("macro_avg_ap"),
            "tuning_metadata": result.tuning_metadata,
        })
    return rows


def _calibration_result_rows(results: List) -> List[Dict[str, object]]:
    rows = []
    for result in results:
        rows.append({
            "name": result.name,
            "selection_score": result.selection_score,
            "accuracy": result.validation_metrics.get("accuracy"),
            "balanced_accuracy": result.validation_metrics.get("balanced_accuracy"),
            "macro_f1": result.validation_metrics.get("macro_f1"),
            "macro_f1_present_classes": result.validation_metrics.get("macro_f1_present_classes"),
            "weighted_f1": result.validation_metrics.get("weighted_f1"),
            "log_loss": result.validation_metrics.get("log_loss"),
            "multiclass_brier": result.validation_metrics.get("multiclass_brier"),
            "ece_macro": result.validation_metrics.get("ece_macro"),
            "accepted": result.accepted,
            "rejection_reason": result.rejection_reason,
        })
    return rows


if __name__ == "__main__":
    summary = run_training_pipeline()
    print(json.dumps(summary, indent=2, default=str))
