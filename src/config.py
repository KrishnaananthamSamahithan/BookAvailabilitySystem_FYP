from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parents[1]
else:
    ROOT_DIR = Path.cwd()


@dataclass
class ProjectConfig:
    """Central configuration used by the final research pipeline.

    The goal of this config object is to keep the pipeline transparent and easy
    to audit. The prediction timestamp assumption is especially important:
    unless a stronger real-time event timestamp exists in the raw logs, we treat
    `LandingTime` as the decision-time proxy.
    """

    root_dir: Path = ROOT_DIR
    raw_data_path: Path = ROOT_DIR / "data" / "raw" / "tbl_SearchTracking_Merged.csv"
    processed_data_path: Path = ROOT_DIR / "data" / "processed" / "final_processed_flight_data.csv"
    artifacts_dir: Path = ROOT_DIR / "artifacts"
    models_dir: Path = ROOT_DIR / "artifacts" / "models"
    reports_dir: Path = ROOT_DIR / "artifacts" / "reports"
    notebook_path: Path = ROOT_DIR / "notebooks" / "Flight_Bookability_Analysis_V2.ipynb"

    prediction_time_field_name: str = "prediction_time"
    prediction_time_assumption: str = (
        "LandingTime is used as the prediction-time proxy because it is the "
        "best available timestamp for the real-time decision moment in the "
        "current raw log export."
    )

    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    rolling_backtest_windows: int = 3

    random_state: int = 42
    calibration_alpha: float = 0.05

    final_target_labels: List[str] = field(
        default_factory=lambda: [
            "bookable",
            "price_changed",
            "unavailable",
            "technical_failure",
        ]
    )

    column_aliases: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "status": ["Status", "status", "booking_status", "result_status"],
            "origin": ["Origin", "origin", "from", "origin_airport"],
            "destination": ["Destination", "destination", "to", "destination_airport"],
            "airline": ["Airline", "airline", "carrier", "airline_code"],
            "cabin": ["Class", "class", "cabin", "cabin_class"],
            "departure_date": ["DepDate", "departure_date", "dep_date", "flight_date"],
            "prediction_time": ["LandingTime", "prediction_time", "search_time", "event_time"],
            "inserted_on": ["InsertedOn", "inserted_on", "outcome_time"],
            "trip_type": ["SearchType", "trip_type", "search_type"],
            "previous_page": ["PreviousPage", "previous_page", "referrer", "url"],
            "landing_page": ["LandingPage", "landing_page"],
            "fdtag": ["FdTag", "fdtag"],
            "user_agent": ["UserAgent", "user_agent"],
            "flight_segments": ["FlightSegments", "flight_segments"],
            "provider_id": ["Provider", "provider", "provider_id"],
            "price": ["Price", "price", "total_price", "fare"],
            "currency": ["Currency", "currency"],
            "stops": ["Stops", "stops", "stop_count"],
            "market": ["Market", "market"],
            "locale": ["Locale", "locale"],
            "device": ["Device", "device"],
            "adults": ["Adults", "adults", "adult_count"],
            "children": ["Children", "children", "child_count"],
            "infants": ["Infants", "infants", "infant_count"],
        }
    )

    required_logical_fields: List[str] = field(
        default_factory=lambda: [
            "status",
            "origin",
            "destination",
            "airline",
            "cabin",
            "departure_date",
            "prediction_time",
        ]
    )

    optional_logical_fields: List[str] = field(
        default_factory=lambda: [
            "inserted_on",
            "trip_type",
            "previous_page",
            "landing_page",
            "fdtag",
            "user_agent",
            "flight_segments",
            "provider_id",
            "price",
            "currency",
            "stops",
            "market",
            "locale",
            "device",
            "adults",
            "children",
            "infants",
        ]
    )

    def ensure_directories(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.notebook_path.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, object]:
        return {
            "root_dir": str(self.root_dir),
            "raw_data_path": str(self.raw_data_path),
            "processed_data_path": str(self.processed_data_path),
            "artifacts_dir": str(self.artifacts_dir),
            "models_dir": str(self.models_dir),
            "reports_dir": str(self.reports_dir),
            "notebook_path": str(self.notebook_path),
            "prediction_time_field_name": self.prediction_time_field_name,
            "prediction_time_assumption": self.prediction_time_assumption,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "rolling_backtest_windows": self.rolling_backtest_windows,
            "random_state": self.random_state,
            "calibration_alpha": self.calibration_alpha,
            "final_target_labels": list(self.final_target_labels),
            "column_aliases": dict(self.column_aliases),
            "required_logical_fields": list(self.required_logical_fields),
            "optional_logical_fields": list(self.optional_logical_fields),
        }
