from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
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


def rolling_backtest_splits(frame: pd.DataFrame, n_windows: int = 3) -> List[Dict[str, pd.DataFrame]]:
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
