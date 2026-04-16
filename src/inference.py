import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.calibration import align_calibrated_probabilities
from src.config import ProjectConfig
from src.features import build_snapshot_feature_bundle
from src.models import predict_proba_aligned
from src.preprocessing import preprocess_raw_frame


def run_inference(input_csv: Path, output_csv: Path, history_csv: Path, bundle_path: Path | None = None) -> pd.DataFrame:
    """Run batch/offline inference with an explicit historical reference dataset.

    This entry point is intentionally honest in scope. It is not a live online
    service with a feature store. Instead, it requires a historical labeled
    dataset so that the same history-based feature logic used in validation and
    test can be reproduced during offline scoring.
    """

    config = ProjectConfig()
    bundle_path = bundle_path or (config.models_dir / "final_model_bundle.joblib")
    bundle = joblib.load(bundle_path)

    raw_input = pd.read_csv(input_csv)
    raw_history = pd.read_csv(history_csv)

    if "Status" not in raw_input.columns and "status" not in raw_input.columns:
        raw_input["Status"] = "unknown"
    if "Status" not in raw_history.columns and "status" not in raw_history.columns:
        raise ValueError("History CSV must include raw outcome labels so prior-only history features can be built safely.")

    input_preprocessed = preprocess_raw_frame(raw_input, config).frame
    history_preprocessed = preprocess_raw_frame(raw_history, config).frame

    scored_bundle = build_snapshot_feature_bundle(input_preprocessed, history_preprocessed)
    X = scored_bundle.frame[bundle["feature_columns"]]

    probabilities = predict_proba_aligned(bundle["estimator"], X, list(range(len(bundle["labels"]))))
    calibrated = align_calibrated_probabilities(bundle["calibrator"], probabilities, list(range(len(bundle["labels"]))))

    prediction_frame = pd.DataFrame(calibrated, columns=[f"p_{label}" for label in bundle["labels"]])
    prediction_frame["predicted_label"] = prediction_frame.idxmax(axis=1).str.replace("p_", "", regex=False)

    output = pd.concat(
        [
            scored_bundle.frame[
                [
                    "prediction_time",
                    "departure_date",
                    "origin_airport",
                    "destination_airport",
                    "airline_code",
                    "provider_key",
                    "meta_engine",
                    "search_group_proxy",
                ]
            ].reset_index(drop=True),
            prediction_frame,
        ],
        axis=1,
    )
    output.to_csv(output_csv, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final offline batch inference pipeline.")
    parser.add_argument("--input", required=True, help="Path to the input CSV.")
    parser.add_argument("--history", required=True, help="Path to the historical reference CSV.")
    parser.add_argument("--output", required=True, help="Path to the output predictions CSV.")
    parser.add_argument("--bundle", required=False, help="Optional path to the model bundle.")
    args = parser.parse_args()

    output = run_inference(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        history_csv=Path(args.history),
        bundle_path=Path(args.bundle) if args.bundle else None,
    )
    print(json.dumps({"rows_scored": len(output), "output_path": args.output}, indent=2))


if __name__ == "__main__":
    main()
