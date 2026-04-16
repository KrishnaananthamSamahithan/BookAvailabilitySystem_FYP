from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    metadata: Dict[str, object]


@dataclass
class FourWaySplit:
    train: pd.DataFrame
    val_model: pd.DataFrame
    val_calibration: pd.DataFrame
    test: pd.DataFrame
    metadata: Dict[str, object]


def temporal_train_validation_test_split(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> TemporalSplit:
    if round(train_fraction + validation_fraction + test_fraction, 6) != 1.0:
        raise ValueError("Train/validation/test fractions must sum to 1.0")

    ordered = frame.sort_values("prediction_time").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * train_fraction)
    validation_end = int(n_rows * (train_fraction + validation_fraction))

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()

    metadata = {
        "n_rows": n_rows,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_start": str(train["prediction_time"].min()) if not train.empty else None,
        "train_end": str(train["prediction_time"].max()) if not train.empty else None,
        "validation_start": str(validation["prediction_time"].min()) if not validation.empty else None,
        "validation_end": str(validation["prediction_time"].max()) if not validation.empty else None,
        "test_start": str(test["prediction_time"].min()) if not test.empty else None,
        "test_end": str(test["prediction_time"].max()) if not test.empty else None,
        "class_distribution": {
            "train": train["outcome_label"].value_counts().to_dict() if "outcome_label" in train else {},
            "validation": validation["outcome_label"].value_counts().to_dict() if "outcome_label" in validation else {},
            "test": test["outcome_label"].value_counts().to_dict() if "outcome_label" in test else {},
        },
    }
    return TemporalSplit(train=train, validation=validation, test=test, metadata=metadata)


def temporal_four_way_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.55,
    val_model_fraction: float = 0.20,
    val_calibration_fraction: float = 0.10,
    test_fraction: float = 0.15,
) -> FourWaySplit:
    """Strict 4-way temporal split eliminating validation double-dipping.

    train          → used for model fitting
    val_model      → used for model selection (NOT seen during training)
    val_calibration → used for calibration fitting ONLY (clean holdout)
    test           → final unbiased evaluation ONLY
    """
    total = train_fraction + val_model_fraction + val_calibration_fraction + test_fraction
    if round(total, 6) != 1.0:
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    ordered = frame.sort_values("prediction_time").reset_index(drop=True)
    n = len(ordered)
    i1 = int(n * train_fraction)
    i2 = int(n * (train_fraction + val_model_fraction))
    i3 = int(n * (train_fraction + val_model_fraction + val_calibration_fraction))

    train = ordered.iloc[:i1].copy()
    val_model = ordered.iloc[i1:i2].copy()
    val_calibration = ordered.iloc[i2:i3].copy()
    test = ordered.iloc[i3:].copy()

    def _dist(df: pd.DataFrame) -> Dict[str, int]:
        if "outcome_label" not in df.columns or df.empty:
            return {}
        return {str(k): int(v) for k, v in df["outcome_label"].value_counts().items()}

    metadata = {
        "split_strategy": "temporal_four_way",
        "n_rows": n,
        "train_rows": len(train),
        "val_model_rows": len(val_model),
        "val_calibration_rows": len(val_calibration),
        "test_rows": len(test),
        "fractions": {
            "train": train_fraction,
            "val_model": val_model_fraction,
            "val_calibration": val_calibration_fraction,
            "test": test_fraction,
        },
        "time_ranges": {
            "train": [str(train["prediction_time"].min()), str(train["prediction_time"].max())] if not train.empty else [],
            "val_model": [str(val_model["prediction_time"].min()), str(val_model["prediction_time"].max())] if not val_model.empty else [],
            "val_calibration": [str(val_calibration["prediction_time"].min()), str(val_calibration["prediction_time"].max())] if not val_calibration.empty else [],
            "test": [str(test["prediction_time"].min()), str(test["prediction_time"].max())] if not test.empty else [],
        },
        "class_distribution": {
            "train": _dist(train),
            "val_model": _dist(val_model),
            "val_calibration": _dist(val_calibration),
            "test": _dist(test),
        },
        "double_dipping_eliminated": True,
        "note": (
            "val_model is used exclusively for model selection. "
            "val_calibration is used exclusively for calibration fitting. "
            "test is the final evaluation holdout — not used at any earlier stage."
        ),
    }
    return FourWaySplit(
        train=train,
        val_model=val_model,
        val_calibration=val_calibration,
        test=test,
        metadata=metadata,
    )


def rolling_backtest_splits(frame: pd.DataFrame, n_windows: int = 5) -> List[Dict[str, pd.DataFrame]]:
    """Generate n_windows forward-chaining rolling backtest windows.

    Each window has an expanding train set and a fixed-size test window.
    Uses 5 windows by default for more robust variance estimation.
    """
    ordered = frame.sort_values("prediction_time").reset_index(drop=True)
    n_rows = len(ordered)
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")

    step = n_rows // (n_windows + 2)
    windows: List[Dict[str, pd.DataFrame]] = []

    for idx in range(n_windows):
        train_end = step * (idx + 2)
        test_end = min(step * (idx + 3), n_rows)
        train = ordered.iloc[:train_end].copy()
        test = ordered.iloc[train_end:test_end].copy()
        if train.empty or test.empty:
            continue
        windows.append(
            {
                "window_id": idx + 1,
                "train": train,
                "test": test,
                "train_start": str(train["prediction_time"].min()),
                "train_end": str(train["prediction_time"].max()),
                "test_start": str(test["prediction_time"].min()),
                "test_end": str(test["prediction_time"].max()),
            }
        )
    return windows
