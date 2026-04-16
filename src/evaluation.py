from typing import Dict, List

import numpy as np
import pandas as pd

from src.metrics import classification_metrics


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray, labels: List[str]) -> Dict[str, object]:
    return classification_metrics(y_true, probabilities, labels)


def comparison_table(rows: List[Dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    preferred_columns = [
        "name",
        "selection_score",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_f1_present_classes",
        "weighted_f1",
        "log_loss",
        "multiclass_brier",
        "ece_macro",
    ]
    columns = [column for column in preferred_columns if column in frame.columns]
    return frame[columns].sort_values("selection_score", ascending=False).reset_index(drop=True)
