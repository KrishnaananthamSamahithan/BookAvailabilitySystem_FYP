from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    numeric_labels = list(range(len(labels)))
    candidates = [
        _tune_logistic_regression(
            X_train,
            y_train,
            X_validation,
            y_validation,
            labels,
            categorical_features,
            numeric_features,
            alpha,
            random_state,
        ),
        _fit_random_forest(
            X_train,
            y_train,
            X_validation,
            y_validation,
            labels,
            categorical_features,
            numeric_features,
            alpha,
            random_state,
        ),
        _tune_catboost(
            X_train,
            y_train,
            X_validation,
            y_validation,
            labels,
            categorical_features,
            alpha,
            random_state,
        ),
    ]
    candidates.sort(key=lambda item: item.selection_score, reverse=True)
    return candidates


def build_candidates(categorical_features: List[str], numeric_features: List[str], random_state: int) -> Dict[str, object]:
    return {
        "logistic_regression": _build_logistic_pipeline(categorical_features, numeric_features, c_value=1.0, random_state=random_state),
        "random_forest": _build_random_forest_pipeline(categorical_features, numeric_features, random_state=random_state),
        "catboost": _build_catboost(depth=8, learning_rate=0.06, iterations=300, random_state=random_state),
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
        aligned[:, class_to_position[int(class_value)]] = raw_probabilities[:, source_idx]
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return aligned / row_sums


def _tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    labels: List[str],
    categorical_features: List[str],
    numeric_features: List[str],
    alpha: float,
    random_state: int,
) -> ModelResult:
    trials = []
    for c_value in [0.5, 1.0, 2.0]:
        estimator = _build_logistic_pipeline(categorical_features, numeric_features, c_value=c_value, random_state=random_state)
        estimator.fit(X_train, y_train)
        probabilities = predict_proba_aligned(estimator, X_validation, list(range(len(labels))))
        metrics = classification_metrics(y_validation, probabilities, labels)
        score = metrics["macro_f1"] - alpha * metrics["log_loss"]
        trials.append((score, c_value, estimator, probabilities, metrics))

    best_score, best_c, best_estimator, best_probabilities, best_metrics = max(trials, key=lambda item: item[0])
    return ModelResult(
        name="logistic_regression",
        estimator=best_estimator,
        validation_probabilities=best_probabilities,
        validation_metrics=best_metrics,
        selection_score=float(best_score),
        tuning_metadata={"best_C": best_c, "trial_count": len(trials)},
    )


def _fit_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    labels: List[str],
    categorical_features: List[str],
    numeric_features: List[str],
    alpha: float,
    random_state: int,
) -> ModelResult:
    estimator = _build_random_forest_pipeline(categorical_features, numeric_features, random_state=random_state)
    estimator.fit(X_train, y_train)
    probabilities = predict_proba_aligned(estimator, X_validation, list(range(len(labels))))
    metrics = classification_metrics(y_validation, probabilities, labels)
    score = metrics["macro_f1"] - alpha * metrics["log_loss"]
    return ModelResult(
        name="random_forest",
        estimator=estimator,
        validation_probabilities=probabilities,
        validation_metrics=metrics,
        selection_score=float(score),
        tuning_metadata={"strategy": "fixed_reasonable_defaults"},
    )


def _tune_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    labels: List[str],
    categorical_features: List[str],
    alpha: float,
    random_state: int,
) -> ModelResult:
    trials = []
    grid = [
        {"depth": 6, "learning_rate": 0.05, "iterations": 250},
        {"depth": 8, "learning_rate": 0.05, "iterations": 250},
        {"depth": 6, "learning_rate": 0.08, "iterations": 350},
        {"depth": 8, "learning_rate": 0.08, "iterations": 350},
    ]
    for params in grid:
        estimator = _build_catboost(random_state=random_state, **params)
        estimator.fit(X_train, y_train, cat_features=categorical_features, eval_set=(X_validation, y_validation))
        probabilities = predict_proba_aligned(estimator, X_validation, list(range(len(labels))))
        metrics = classification_metrics(y_validation, probabilities, labels)
        score = metrics["macro_f1"] - alpha * metrics["log_loss"]
        trials.append((score, params, estimator, probabilities, metrics))

    best_score, best_params, best_estimator, best_probabilities, best_metrics = max(trials, key=lambda item: item[0])
    return ModelResult(
        name="catboost",
        estimator=best_estimator,
        validation_probabilities=best_probabilities,
        validation_metrics=best_metrics,
        selection_score=float(best_score),
        tuning_metadata={"best_params": best_params, "trial_count": len(trials)},
    )


def _build_logistic_pipeline(categorical_features: List[str], numeric_features: List[str], c_value: float, random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="saga",
                    class_weight="balanced",
                    C=c_value,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _build_random_forest_pipeline(categorical_features: List[str], numeric_features: List[str], random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=18,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
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
