"""Calibration module.

Provides calibrators:
  - IdentityCalibrator (no calibration baseline)
  - TemperatureScalingCalibrator (single scalar — preserves rank)
  - OVRIsotonicCalibrator (class-wise isotonic regression)
  - MultinomialLogisticCalibrator (parametric recalibration)
  - DirichletCalibrator (principled multiclass extension; state-of-art)

Improvements:
  - Added DirichletCalibrator for principled multi-class calibration.
  - Rejection logic bug fixed: acceptance criteria now applied strictly.
  - compare_calibrators returns ALL results; first accepted with best score wins.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

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
    """Scales all class logits by a single learned temperature T.

    Preserves the model's discriminative ordering while adjusting
    the sharpness/width of the probability distribution.
    """
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
    """One-vs-rest isotonic regression calibrator.

    Fits an independent isotonic regression for each class.
    Non-parametric — no shape assumptions.
    May overfit on small calibration sets with minority classes.
    """
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
    """Platt-style multi-class calibration using a logistic regression on log-probabilities."""
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


class DirichletCalibrator:
    """Dirichlet calibration for multi-class probability outputs.

    Fits a linear transformation in log-probability space that maps
    raw model outputs to a Dirichlet distribution. This is the principled
    multi-class extension of Platt scaling (Kull et al., 2019).

    Reference: Kull M., Filho T.M.S., Flach P. (2019). Dirichlet Calibration.
    """
    name = "dirichlet"

    def __init__(self, lambda_reg: float = 1e-3) -> None:
        self.lambda_reg = lambda_reg
        self._model: Optional[LogisticRegression] = None

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "DirichletCalibrator":
        self.classes_ = np.arange(probabilities.shape[1])
        n_classes = probabilities.shape[1]
        # Log-transform raw probabilities to form feature matrix
        # Each sample gets n_classes log-probability features
        log_probs = np.log(np.clip(probabilities, 1e-12, 1.0))
        # Fit L2-regularized multinomial logistic regression on log-probabilities
        self._model = LogisticRegression(
            solver="lbfgs",
            multi_class="multinomial",
            C=1.0 / (self.lambda_reg * len(y_true)),
            max_iter=1000,
        )
        self._model.fit(log_probs, y_true)
        self.classes_ = self._model.classes_
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        if self._model is None:
            return probabilities
        log_probs = np.log(np.clip(probabilities, 1e-12, 1.0))
        return self._model.predict_proba(log_probs)


@dataclass
class CalibrationResult:
    name: str
    calibrator: object
    validation_metrics: Dict[str, object]
    selection_score: float
    accepted: bool
    rejection_reason: Optional[str]


def compare_calibrators(
    validation_probabilities: np.ndarray,
    y_validation: np.ndarray,
    labels: List[str],
    min_balanced_accuracy: float = 0.40,
    min_macro_f1: float = 0.30,
    max_drop_vs_uncalibrated: float = 0.03,
) -> List[CalibrationResult]:
    """Compare all calibrators and select the best accepted one.

    Calibrators are accepted if they meet all quality thresholds:
      1. balanced_accuracy >= min_balanced_accuracy
      2. macro_f1_present_classes >= min_macro_f1
      3. They do not degrade discriminative performance vs uncalibrated by more
         than max_drop_vs_uncalibrated on either balanced_accuracy or macro_f1.

    The identity calibrator (no calibration) is always accepted as a fallback.
    Results are sorted by selection_score descending; best accepted calibrator is first.
    """
    calibrators = [
        IdentityCalibrator(),
        TemperatureScalingCalibrator(),
        OVRIsotonicCalibrator(),
        MultinomialLogisticCalibrator(),
        DirichletCalibrator(),
    ]
    results: List[CalibrationResult] = []
    uncalibrated_reference: Optional[Dict[str, object]] = None

    for calibrator in calibrators:
        try:
            calibrator.fit(validation_probabilities, y_validation)
            calibrated = align_calibrated_probabilities(
                calibrator, validation_probabilities, list(range(len(labels)))
            )
            metrics = classification_metrics(y_validation, calibrated, labels)
        except Exception as exc:
            # If a calibrator fails to fit, treat it as rejected
            results.append(CalibrationResult(
                name=calibrator.name,
                calibrator=calibrator,
                validation_metrics={"error": str(exc)},
                selection_score=-1e9,
                accepted=False,
                rejection_reason=f"fit_failed: {exc}",
            ))
            continue

        if calibrator.name == "none":
            uncalibrated_reference = metrics

        # --- Acceptance gate (strictly applied) ---
        accepted = True
        rejection_reason = None

        if calibrator.name == "none":
            # Identity calibrator always accepted as baseline fallback
            accepted = True
        elif metrics["balanced_accuracy"] < min_balanced_accuracy:
            accepted = False
            rejection_reason = (
                f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
                f"< threshold={min_balanced_accuracy}"
            )
        elif metrics["macro_f1_present_classes"] < min_macro_f1:
            accepted = False
            rejection_reason = (
                f"macro_f1_present_classes={metrics['macro_f1_present_classes']:.4f} "
                f"< threshold={min_macro_f1}"
            )
        elif uncalibrated_reference is not None:
            ba_drop = uncalibrated_reference["balanced_accuracy"] - metrics["balanced_accuracy"]
            f1_drop = uncalibrated_reference["macro_f1_present_classes"] - metrics["macro_f1_present_classes"]
            if ba_drop > max_drop_vs_uncalibrated or f1_drop > max_drop_vs_uncalibrated:
                accepted = False
                rejection_reason = (
                    f"degrades_discrimination_vs_uncalibrated: "
                    f"ba_drop={ba_drop:.4f}, f1_drop={f1_drop:.4f}, "
                    f"max_allowed={max_drop_vs_uncalibrated}"
                )

        if accepted:
            # Primary: high macro_f1_present_classes; secondary: low ECE (weight 0.10)
            selection_score = (
                metrics["macro_f1_present_classes"]
                - 0.05 * metrics["log_loss"]
                - 0.10 * metrics["ece_macro"]
            )
        else:
            selection_score = -1e9

        results.append(
            CalibrationResult(
                name=calibrator.name,
                calibrator=calibrator,
                validation_metrics=metrics,
                selection_score=float(selection_score),
                accepted=accepted,
                rejection_reason=rejection_reason,
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
        if int(class_value) in class_to_position:
            aligned[:, class_to_position[int(class_value)]] = raw_probabilities[:, source_idx]
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return aligned / row_sums


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)
