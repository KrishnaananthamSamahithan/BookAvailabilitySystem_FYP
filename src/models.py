from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from src.metrics import classification_metrics


@dataclass
class ModelResult:
    name: str
    estimator: object
    validation_probabilities: np.ndarray
    validation_metrics: Dict[str, object]
    selection_score: float
    tuning_metadata: Dict[str, object]


def compare_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    labels: List[str],
    categorical_features: List[str],
    numeric_features: List[str],
    alpha: float,
    random_state: int,
) -> List[ModelResult]:
    """Compare LR, RF, CatBoost, XGBoost, LightGBM on the validation set.

    All models go through a grid search to ensure fair comparison.
    CatBoost is NOT given the validation set as eval_set during the comparison
    phase to ensure a fair, unbiased comparison across all models.
    The eval_set is only used when re-fitting the final chosen model for production.
    """
    candidates = [
        _tune_logistic_regression(
            X_train, y_train, X_validation, y_validation,
            labels, categorical_features, numeric_features, alpha, random_state,
        ),
        _tune_random_forest(                              # FIX: now properly tuned
            X_train, y_train, X_validation, y_validation,
            labels, categorical_features, numeric_features, alpha, random_state,
        ),
        _tune_catboost(
            X_train, y_train, X_validation, y_validation,
            labels, categorical_features, alpha, random_state,
            use_eval_set=False,  # Disabled for fair comparison
        ),
        _tune_xgboost(
            X_train, y_train, X_validation, y_validation,
            labels, categorical_features, numeric_features, alpha, random_state,
        ),
        _tune_lightgbm(
            X_train, y_train, X_validation, y_validation,
            labels, categorical_features, numeric_features, alpha, random_state,
        ),
    ]
    candidates.sort(key=lambda item: item.selection_score, reverse=True)
    return candidates


def run_naive_baselines(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    labels: List[str],
) -> List[Dict[str, object]]:
    """Evaluate DummyClassifier strategies as lower-bound baselines.

    Any ML model that cannot beat these baselines provides no value.
    """
    rows = []
    for strategy in ["most_frequent", "stratified", "uniform", "prior"]:
        dummy = DummyClassifier(strategy=strategy, random_state=42)
        dummy.fit(X_train, y_train)
        raw_proba = dummy.predict_proba(X_test)

        # Align probability columns to label order
        n_labels = len(labels)
        aligned = np.zeros((len(X_test), n_labels), dtype=float)
        for src_idx, cls in enumerate(dummy.classes_):
            if int(cls) < n_labels:
                aligned[:, int(cls)] = raw_proba[:, src_idx]
        row_sums = aligned.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        aligned /= row_sums

        metrics = classification_metrics(y_test, aligned, labels)
        rows.append({
            "name": f"dummy_{strategy}",
            "strategy": strategy,
            "selection_score": float(metrics["macro_f1_present_classes"] - 0.05 * metrics["log_loss"]),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_f1_present_classes": metrics["macro_f1_present_classes"],
            "weighted_f1": metrics["weighted_f1"],
            "log_loss": metrics["log_loss"],
            "multiclass_brier": metrics["multiclass_brier"],
            "ece_macro": metrics["ece_macro"],
        })
    return rows


def build_candidates(categorical_features: List[str], numeric_features: List[str], random_state: int) -> Dict[str, object]:
    return {
        "logistic_regression": _build_logistic_pipeline(categorical_features, numeric_features, c_value=1.0, random_state=random_state),
        "random_forest": _build_random_forest_pipeline(categorical_features, numeric_features, random_state=random_state),
        "catboost": _build_catboost(depth=8, learning_rate=0.06, iterations=300, random_state=random_state),
        "xgboost": _build_xgboost_pipeline(categorical_features, numeric_features, random_state=random_state),
        "lightgbm": _build_lightgbm_pipeline(categorical_features, numeric_features, random_state=random_state),
    }


def predict_proba_aligned(estimator, X: pd.DataFrame, class_labels: List[int]) -> np.ndarray:
    raw_probabilities = estimator.predict_proba(X)
    estimator_classes = getattr(estimator, "classes_", None)
    if estimator_classes is None and hasattr(estimator, "named_steps"):
        classifier = estimator.named_steps.get("classifier")
        estimator_classes = getattr(classifier, "classes_", None)
    if estimator_classes is None:
        return raw_probabilities

    aligned = np.zeros((len(X), len(class_labels)), dtype=float)
    class_to_position = {int(label): idx for idx, label in enumerate(class_labels)}
    for source_idx, class_value in enumerate(estimator_classes):
        if int(class_value) in class_to_position:
            aligned[:, class_to_position[int(class_value)]] = raw_probabilities[:, source_idx]
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return aligned / row_sums


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def _tune_logistic_regression(
    X_train, y_train, X_validation, y_validation,
    labels, categorical_features, numeric_features, alpha, random_state,
) -> ModelResult:
    trials = []
    for c_value in [0.1, 0.5, 1.0, 2.0, 5.0]:
        estimator = _build_logistic_pipeline(categorical_features, numeric_features, c_value=c_value, random_state=random_state)
        estimator.fit(X_train, y_train)
        probabilities = predict_proba_aligned(estimator, X_validation, list(range(len(labels))))
        metrics = classification_metrics(y_validation, probabilities, labels)
        score = metrics["macro_f1_present_classes"] - alpha * metrics["log_loss"]
        trials.append((score, c_value, estimator, probabilities, metrics))

    best_score, best_c, best_estimator, best_probabilities, best_metrics = max(trials, key=lambda item: item[0])
    return ModelResult(
        name="logistic_regression",
        estimator=best_estimator,
        validation_probabilities=best_probabilities,
        validation_metrics=best_metrics,
        selection_score=float(best_score),
        tuning_metadata={"best_C": best_c, "trial_count": len(trials), "C_grid": [0.1, 0.5, 1.0, 2.0, 5.0]},
    )


def _build_logistic_pipeline(categorical_features, numeric_features, c_value, random_state) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                ]),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]),
                numeric_features,
            ),
        ]
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            solver="saga",
            class_weight="balanced",
            C=c_value,
            random_state=random_state,
        )),
    ])


# ---------------------------------------------------------------------------
# Random Forest — NOW PROPERLY TUNED (FIX: was previously using fixed defaults)
# ---------------------------------------------------------------------------

def _tune_random_forest(
    X_train, y_train, X_validation, y_validation,
    labels, categorical_features, numeric_features, alpha, random_state,
) -> ModelResult:
    """Grid-search tuning for Random Forest.

    FIX: Previously RF was fitted with fixed defaults and NOT through a grid search,
    while all other models used grid search. This unfairly advantaged RF in the
    model comparison and invalidated selection conclusions.
    """
    trials = []
    grid = [
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": 18, "min_samples_leaf": 2},
        {"n_estimators": 500, "max_depth": 18, "min_samples_leaf": 1},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 5},
    ]
    for params in grid:
        estimator = _build_random_forest_pipeline(
            categorical_features, numeric_features,
            random_state=random_state, **params,
        )
        estimator.fit(X_train, y_train)
        probabilities = predict_proba_aligned(estimator, X_validation, list(range(len(labels))))
        metrics = classification_metrics(y_validation, probabilities, labels)
        score = metrics["macro_f1_present_classes"] - alpha * metrics["log_loss"]
        trials.append((score, params, estimator, probabilities, metrics))

    best_score, best_params, best_estimator, best_probabilities, best_metrics = max(trials, key=lambda item: item[0])
    return ModelResult(
        name="random_forest",
        estimator=best_estimator,
        validation_probabilities=best_probabilities,
        validation_metrics=best_metrics,
        selection_score=float(best_score),
        tuning_metadata={
            "best_params": best_params,
            "trial_count": len(trials),
            "grid": grid,
            "note": "RF now tuned via grid search — same rigor as other models.",
        },
    )


def _build_random_forest_pipeline(
    categorical_features, numeric_features, random_state,
    n_estimators: int = 300, max_depth=18, min_samples_leaf: int = 2,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                ]),
                categorical_features,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
        ]
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )),
    ])


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

def _tune_catboost(
    X_train, y_train, X_validation, y_validation,
    labels, categorical_features, alpha, random_state,
    use_eval_set: bool = False,
) -> ModelResult:
    trials = []
    grid = [
        {"depth": 6, "learning_rate": 0.05, "iterations": 300},
        {"depth": 8, "learning_rate": 0.05, "iterations": 300},
        {"depth": 6, "learning_rate": 0.08, "iterations": 400},
        {"depth": 8, "learning_rate": 0.08, "iterations": 400},
        {"depth": 6, "learning_rate": 0.10, "iterations": 500},
        {"depth": 8, "learning_rate": 0.10, "iterations": 500},
    ]
    for params in grid:
        estimator = _build_catboost(random_state=random_state, **params)
        fit_kwargs = {"cat_features": categorical_features}
        if use_eval_set:
            fit_kwargs["eval_set"] = (X_validation, y_validation)
        estimator.fit(X_train, y_train, **fit_kwargs)
        probabilities = predict_proba_aligned(estimator, X_validation, list(range(len(labels))))
        metrics = classification_metrics(y_validation, probabilities, labels)
        score = metrics["macro_f1_present_classes"] - alpha * metrics["log_loss"]
        trials.append((score, params, estimator, probabilities, metrics))

    best_score, best_params, best_estimator, best_probabilities, best_metrics = max(trials, key=lambda item: item[0])
    return ModelResult(
        name="catboost",
        estimator=best_estimator,
        validation_probabilities=best_probabilities,
        validation_metrics=best_metrics,
        selection_score=float(best_score),
        tuning_metadata={
            "best_params": best_params,
            "trial_count": len(trials),
            "eval_set_used_during_fit": use_eval_set,
        },
    )


def _build_catboost(depth: int, learning_rate: float, iterations: int, random_state: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=random_state,
    )


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def _tune_xgboost(
    X_train, y_train, X_validation, y_validation,
    labels, categorical_features, numeric_features, alpha, random_state,
) -> ModelResult:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return _make_skipped_result("xgboost", "xgboost not installed")

    trials = []
    grid = [
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300},
        {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 300},
        {"max_depth": 6, "learning_rate": 0.10, "n_estimators": 400},
        {"max_depth": 8, "learning_rate": 0.10, "n_estimators": 400},
        {"max_depth": 6, "learning_rate": 0.08, "n_estimators": 500},
    ]
    for params in grid:
        estimator = _build_xgboost_pipeline(categorical_features, numeric_features, random_state=random_state, **params)
        estimator.fit(X_train, y_train)
        probabilities = predict_proba_aligned(estimator, X_validation, list(range(len(labels))))
        metrics = classification_metrics(y_validation, probabilities, labels)
        score = metrics["macro_f1_present_classes"] - alpha * metrics["log_loss"]
        trials.append((score, params, estimator, probabilities, metrics))

    best_score, best_params, best_estimator, best_probabilities, best_metrics = max(trials, key=lambda item: item[0])
    return ModelResult(
        name="xgboost",
        estimator=best_estimator,
        validation_probabilities=best_probabilities,
        validation_metrics=best_metrics,
        selection_score=float(best_score),
        tuning_metadata={"best_params": best_params, "trial_count": len(trials)},
    )


def _build_xgboost_pipeline(
    categorical_features, numeric_features, random_state,
    max_depth=6, learning_rate=0.05, n_estimators=300,
) -> Pipeline:
    from xgboost import XGBClassifier

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                categorical_features,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
        ]
    )
    # FIX: Removed deprecated use_label_encoder=False (removed in XGBoost 2.x)
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )),
    ])


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def _tune_lightgbm(
    X_train, y_train, X_validation, y_validation,
    labels, categorical_features, numeric_features, alpha, random_state,
) -> ModelResult:
    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        return _make_skipped_result("lightgbm", "lightgbm not installed")

    trials = []
    grid = [
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300},
        {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 300},
        {"num_leaves": 31, "learning_rate": 0.10, "n_estimators": 400},
        {"num_leaves": 63, "learning_rate": 0.10, "n_estimators": 400},
        {"num_leaves": 127, "learning_rate": 0.08, "n_estimators": 500},
    ]
    for params in grid:
        estimator = _build_lightgbm_pipeline(categorical_features, numeric_features, random_state=random_state, **params)
        estimator.fit(X_train, y_train)
        probabilities = predict_proba_aligned(estimator, X_validation, list(range(len(labels))))
        metrics = classification_metrics(y_validation, probabilities, labels)
        score = metrics["macro_f1_present_classes"] - alpha * metrics["log_loss"]
        trials.append((score, params, estimator, probabilities, metrics))

    best_score, best_params, best_estimator, best_probabilities, best_metrics = max(trials, key=lambda item: item[0])
    return ModelResult(
        name="lightgbm",
        estimator=best_estimator,
        validation_probabilities=best_probabilities,
        validation_metrics=best_metrics,
        selection_score=float(best_score),
        tuning_metadata={"best_params": best_params, "trial_count": len(trials)},
    )


def _build_lightgbm_pipeline(
    categorical_features, numeric_features, random_state,
    num_leaves=31, learning_rate=0.05, n_estimators=300,
) -> Pipeline:
    from lightgbm import LGBMClassifier

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                categorical_features,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
        ]
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LGBMClassifier(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skipped_result(name: str, reason: str) -> ModelResult:
    n_dummy = 10
    dummy_proba = np.ones((n_dummy, 4)) / 4
    dummy_y = np.zeros(n_dummy, dtype=int)
    dummy_metrics = classification_metrics(dummy_y, dummy_proba, ["a", "b", "c", "d"])
    return ModelResult(
        name=name,
        estimator=None,
        validation_probabilities=dummy_proba,
        validation_metrics=dummy_metrics,
        selection_score=-1e9,
        tuning_metadata={"skipped": True, "reason": reason},
    )
