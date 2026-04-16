from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.metrics import classification_metrics


class IdentityCalibrator:
    name = "none"

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "IdentityCalibrator":
        self.classes_ = np.arange(probabilities.shape[1])
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        return probabilities


class TemperatureScalingCalibrator:
    name = "temperature_scaling"

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "TemperatureScalingCalibrator":
        self.classes_ = np.arange(probabilities.shape[1])
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))

        def objective(temp: np.ndarray) -> float:
            scaled = _softmax(logits / temp[0])
            metrics = classification_metrics(y_true, scaled, [str(i) for i in range(probabilities.shape[1])])
            return metrics["log_loss"]

        result = minimize(objective, x0=np.array([1.0]), bounds=[(0.05, 10.0)])
        self.temperature = float(result.x[0])
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
        return _softmax(logits / self.temperature)


class OVRIsotonicCalibrator:
    name = "ovr_isotonic"

    def __init__(self) -> None:
        self.models: List[IsotonicRegression] = []

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "OVRIsotonicCalibrator":
        self.classes_ = np.arange(probabilities.shape[1])
        self.models = []
        for class_index in range(probabilities.shape[1]):
            isotonic = IsotonicRegression(out_of_bounds="clip")
            binary_target = (y_true == class_index).astype(int)
            isotonic.fit(probabilities[:, class_index], binary_target)
            self.models.append(isotonic)
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated = np.zeros_like(probabilities)
        for class_index, isotonic in enumerate(self.models):
            calibrated[:, class_index] = isotonic.transform(probabilities[:, class_index])
        calibrated = np.clip(calibrated, 1e-9, 1.0)
        return calibrated / calibrated.sum(axis=1, keepdims=True)


class MultinomialLogisticCalibrator:
    name = "multinomial_logistic"

    def __init__(self) -> None:
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
        )

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "MultinomialLogisticCalibrator":
        features = np.log(np.clip(probabilities, 1e-12, 1.0))
        self.model.fit(features, y_true)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        features = np.log(np.clip(probabilities, 1e-12, 1.0))
        return self.model.predict_proba(features)


@dataclass
class CalibrationResult:
    name: str
    calibrator: object
    validation_metrics: Dict[str, object]
    selection_score: float


def compare_calibrators(
    validation_probabilities: np.ndarray,
    y_validation: np.ndarray,
    labels: List[str],
) -> List[CalibrationResult]:
    calibrators = [
        IdentityCalibrator(),
        TemperatureScalingCalibrator(),
        OVRIsotonicCalibrator(),
        MultinomialLogisticCalibrator(),
    ]
    results: List[CalibrationResult] = []

    for calibrator in calibrators:
        calibrator.fit(validation_probabilities, y_validation)
        calibrated = align_calibrated_probabilities(calibrator, validation_probabilities, list(range(len(labels))))
        metrics = classification_metrics(y_validation, calibrated, labels)
        selection_score = metrics["macro_f1"] - 0.05 * metrics["log_loss"] - 0.10 * metrics["ece_macro"]
        results.append(
            CalibrationResult(
                name=calibrator.name,
                calibrator=calibrator,
                validation_metrics=metrics,
                selection_score=float(selection_score),
            )
        )

    results.sort(key=lambda item: item.selection_score, reverse=True)
    return results


def align_calibrated_probabilities(calibrator, probabilities: np.ndarray, class_labels: List[int]) -> np.ndarray:
    raw_probabilities = calibrator.predict_proba(probabilities)
    calibrator_classes = getattr(calibrator, "classes_", None)
    if calibrator_classes is None:
        return raw_probabilities

    aligned = np.zeros((len(probabilities), len(class_labels)), dtype=float)
    class_to_position = {int(label): idx for idx, label in enumerate(class_labels)}
    for source_idx, class_value in enumerate(calibrator_classes):
        aligned[:, class_to_position[int(class_value)]] = raw_probabilities[:, source_idx]
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return aligned / row_sums


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)
