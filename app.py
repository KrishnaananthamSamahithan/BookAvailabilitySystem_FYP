"""Real-Time Flight Bookability Prediction System - Flask Application.

This module provides:
  - A research dashboard (/) showing all training artifacts
  - A real-time single-prediction UI (/predict)
  - A REST JSON API (/api/predict) with latency measurement
  - Live drift monitoring (/api/drift)
  - Model metrics API (/api/metrics)
  - Latency statistics API (/api/latency)

The in-memory feature store (InMemoryFeatureStore) accumulates predictions
so that provider/route history features are updated with each new request,
enabling a genuine near-real-time serving experience.
"""

import json
import os
import time
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.calibration import align_calibrated_probabilities
from src.drift import OnlineDriftMonitor
from src.models import predict_proba_aligned
from src.config import ProjectConfig

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CONFIG = ProjectConfig()
BUNDLE_PATH = CONFIG.models_dir / "final_model_bundle.joblib"
REPORTS_DIR = CONFIG.reports_dir
REQUIRED_REPORTS = [
    "summary.json",
    "model_comparison.json",
    "calibration_comparison.json",
    "final_test_metrics.json",
    "naive_baselines.json",
    "significance_tests.json",
    "drift_analysis.json",
    "provider_holdout_evaluation.json",
    "class_presence.json",
    "experiment_manifest.json",
]

# ---------------------------------------------------------------------------
# In-memory feature store
# ---------------------------------------------------------------------------

class InMemoryFeatureStore:
    """Thread-safe in-memory store for real-time history-based features.

    FIX: Now tracks per-outcome counters for all 4 classes independently,
    instead of the incorrect binary `price_changed = 1 - bookable_rate`
    approximation that assumed a 2-class world.

    Accumulates labelled outcomes from live predictions so that the next
    prediction can use updated provider/route reliability rates without
    requiring a full retraining cycle.
    """

    OUTCOME_LABELS = ["bookable", "price_changed", "unavailable", "technical_failure"]
    OUTCOME_ALIASES = {
        "bookable": "bookable",
        "price_changed": "price_changed",
        "pricechange": "price_changed",
        "price_change": "price_changed",
        "unavailable": "unavailable",
        "technical_failure": "technical_failure",
        "technicalfailure": "technical_failure",
        "technical": "technical_failure",
        "failure": "technical_failure",
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._history_defaults: Dict[str, float] = {}
        # Per-provider: total count + per-outcome counts (all 4 classes)
        self._provider_counts: Dict[str, int] = {}
        self._provider_outcomes: Dict[str, Dict[str, int]] = {}   # FIX
        # Per-route: same structure
        self._route_counts: Dict[str, int] = {}
        self._route_outcomes: Dict[str, Dict[str, int]] = {}       # FIX
        # Provider-route combined
        self._prov_route_outcomes: Dict[str, Dict[str, int]] = {}  # FIX
        # Timestamps for recency features
        self._provider_last_time: Dict[str, datetime] = {}
        self._route_last_time: Dict[str, datetime] = {}
        self._prediction_log: deque = deque(maxlen=5000)
        self._pending_feedback: Dict[str, Dict[str, str]] = {}

    def _rate(self, outcome_dict: Dict[str, int], outcome: str) -> float:
        """Compute the prior rate for a given outcome, defaulting to equal prior."""
        total = sum(outcome_dict.values())
        if total == 0:
            return 0.25  # uninformed prior: equal across 4 classes
        return float(outcome_dict.get(outcome, 0)) / total

    def bootstrap_defaults(self, history_defaults: Optional[Dict[str, float]]) -> None:
        """Seed live-serving priors from training-time inference defaults."""
        if not history_defaults:
            return
        with self._lock:
            self._history_defaults = {
                str(key): float(value)
                for key, value in history_defaults.items()
                if value is not None
            }

    def get_features(
        self,
        provider_key: str,
        route: str,
        now: datetime,
    ) -> Dict[str, float]:
        """Return real-time history features for the given provider/route."""
        with self._lock:
            p_count = self._provider_counts.get(
                provider_key,
                int(round(self._history_defaults.get("provider_prior_count", 0.0))),
            )
            r_count = self._route_counts.get(
                route,
                int(round(self._history_defaults.get("route_prior_count", 0.0))),
            )

            p_outcomes = self._provider_outcomes.get(provider_key, {})
            r_outcomes = self._route_outcomes.get(route, {})
            pr_outcomes = self._prov_route_outcomes.get(f"{provider_key}_{route}", {})

            # FIX: compute each rate independently from per-outcome counters
            p_rate_bookable = self._rate(p_outcomes, "bookable") if p_outcomes else self._history_defaults.get("provider_prior_rate_bookable", 0.25)
            p_rate_price_changed = self._rate(p_outcomes, "price_changed") if p_outcomes else self._history_defaults.get("provider_prior_rate_price_changed", 0.25)
            p_rate_unavailable = self._rate(p_outcomes, "unavailable") if p_outcomes else self._history_defaults.get("provider_prior_rate_unavailable", 0.25)
            p_rate_technical = self._rate(p_outcomes, "technical_failure") if p_outcomes else self._history_defaults.get("provider_prior_rate_technical_failure", 0.25)

            r_rate_bookable = self._rate(r_outcomes, "bookable") if r_outcomes else self._history_defaults.get("route_prior_rate_bookable", 0.25)
            r_rate_price_changed = self._rate(r_outcomes, "price_changed") if r_outcomes else self._history_defaults.get("route_prior_rate_price_changed", 0.25)
            r_rate_unavailable = self._rate(r_outcomes, "unavailable") if r_outcomes else self._history_defaults.get("route_prior_rate_unavailable", 0.25)

            pr_rate_bookable = self._rate(pr_outcomes, "bookable") if pr_outcomes else self._history_defaults.get("provider_route_prior_rate_bookable", 0.25)
            pr_rate_price_changed = self._rate(pr_outcomes, "price_changed") if pr_outcomes else self._history_defaults.get("provider_route_prior_rate_price_changed", 0.25)
            pr_rate_unavailable = self._rate(pr_outcomes, "unavailable") if pr_outcomes else self._history_defaults.get("provider_route_prior_rate_unavailable", 0.25)

            p_last = self._provider_last_time.get(provider_key)
            r_last = self._route_last_time.get(route)

            p_minutes = (now - p_last).total_seconds() / 60.0 if p_last else self._history_defaults.get("provider_minutes_since_prev", 60.0)
            r_minutes = (now - r_last).total_seconds() / 60.0 if r_last else self._history_defaults.get("route_minutes_since_prev", 60.0)
            pr_minutes = min(p_minutes, r_minutes)

        provider_instability = (p_rate_price_changed + p_rate_unavailable + p_rate_technical) / 3.0
        route_instability = (r_rate_price_changed + r_rate_unavailable) / 2.0
        price_change_pressure = (
            p_rate_price_changed + r_rate_price_changed + pr_rate_price_changed
        ) / 3.0
        technical_failure_pressure = p_rate_technical
        unavailability_pressure = (
            p_rate_unavailable + r_rate_unavailable + pr_rate_unavailable
        ) / 3.0

        return {
            "provider_prior_count": float(p_count),
            "provider_prior_rate_bookable": p_rate_bookable,
            "provider_prior_rate_price_changed": p_rate_price_changed,   # FIX
            "provider_prior_rate_unavailable": p_rate_unavailable,        # FIX
            "provider_prior_rate_technical_failure": p_rate_technical,    # FIX
            "route_prior_count": float(r_count),
            "route_prior_rate_bookable": r_rate_bookable,
            "route_prior_rate_price_changed": r_rate_price_changed,
            "route_prior_rate_unavailable": r_rate_unavailable,
            "provider_route_prior_rate_bookable": pr_rate_bookable,        # FIX
            "provider_route_prior_rate_price_changed": pr_rate_price_changed,
            "provider_route_prior_rate_unavailable": pr_rate_unavailable,  # FIX
            "airline_route_prior_rate_bookable": r_rate_bookable,
            "airline_route_prior_rate_price_changed": r_rate_price_changed,
            "airline_route_prior_rate_unavailable": r_rate_unavailable,
            "provider_instability_score": provider_instability,
            "route_instability_score": route_instability,
            "price_change_pressure": price_change_pressure,
            "technical_failure_pressure": technical_failure_pressure,
            "unavailability_pressure": unavailability_pressure,
            "provider_minutes_since_prev": max(0.0, p_minutes),
            "route_minutes_since_prev": max(0.0, r_minutes),
            "provider_route_minutes_since_prev": max(0.0, pr_minutes),
        }

    def _apply_outcome_update(
        self,
        provider_key: str,
        route: str,
        outcome_label: str,
        now: datetime,
        source: str,
        prediction_id: Optional[str] = None,
    ) -> None:
        """Apply true-outcome update to counters."""
        if outcome_label not in self.OUTCOME_LABELS:
            return
        with self._lock:
            self._provider_counts[provider_key] = self._provider_counts.get(provider_key, 0) + 1
            if provider_key not in self._provider_outcomes:
                self._provider_outcomes[provider_key] = {}
            self._provider_outcomes[provider_key][outcome_label] = (
                self._provider_outcomes[provider_key].get(outcome_label, 0) + 1
            )

            self._route_counts[route] = self._route_counts.get(route, 0) + 1
            if route not in self._route_outcomes:
                self._route_outcomes[route] = {}
            self._route_outcomes[route][outcome_label] = (
                self._route_outcomes[route].get(outcome_label, 0) + 1
            )

            pr_key = f"{provider_key}_{route}"
            if pr_key not in self._prov_route_outcomes:
                self._prov_route_outcomes[pr_key] = {}
            self._prov_route_outcomes[pr_key][outcome_label] = (
                self._prov_route_outcomes[pr_key].get(outcome_label, 0) + 1
            )

            self._provider_last_time[provider_key] = now
            self._route_last_time[route] = now
            self._prediction_log.append({
                "timestamp": now.isoformat(),
                "provider_key": provider_key,
                "route": route,
                "outcome_label": outcome_label,
                "source": source,
                "prediction_id": prediction_id,
            })

    def register_prediction(
        self,
        provider_key: str,
        route: str,
        predicted_label: str,
        now: datetime,
    ) -> str:
        """Register prediction event without updating counters."""
        prediction_id = str(uuid4())
        with self._lock:
            self._pending_feedback[prediction_id] = {
                "provider_key": provider_key,
                "route": route,
                "predicted_label": predicted_label,
                "timestamp": now.isoformat(),
            }
            self._prediction_log.append({
                "timestamp": now.isoformat(),
                "provider_key": provider_key,
                "route": route,
                "predicted_label": predicted_label,
                "source": "prediction",
                "prediction_id": prediction_id,
            })
        return prediction_id

    def apply_feedback(
        self,
        true_label: str,
        now: datetime,
        prediction_id: Optional[str] = None,
        provider_key: Optional[str] = None,
        route: Optional[str] = None,
    ) -> Dict[str, object]:
        """Apply true-outcome feedback and update reliability counters."""
        normalized_label = self.OUTCOME_ALIASES.get(str(true_label).strip().lower())
        if normalized_label not in self.OUTCOME_LABELS:
            return {"status": "error", "message": f"Unsupported true_label: {true_label}"}

        with self._lock:
            if prediction_id and prediction_id in self._pending_feedback:
                pending = self._pending_feedback.pop(prediction_id)
                provider_key = provider_key or pending["provider_key"]
                route = route or pending["route"]
            elif prediction_id and prediction_id not in self._pending_feedback:
                return {"status": "error", "message": "prediction_id not found or already consumed"}

        if not provider_key or not route:
            return {
                "status": "error",
                "message": "provider_key and route are required when prediction_id is missing",
            }
        provider_key = str(provider_key).strip() or "Unknown"
        route = str(route).strip().upper()

        self._apply_outcome_update(
            provider_key=provider_key,
            route=route,
            outcome_label=normalized_label,
            now=now,
            source="feedback",
            prediction_id=prediction_id,
        )
        return {"status": "ok", "provider_key": provider_key, "route": route, "true_label": normalized_label}

    def recent_events(self, n: int = 10) -> List[Dict]:
        with self._lock:
            log = list(self._prediction_log)
        return log[-n:]

    def stats(self) -> Dict[str, int]:
        with self._lock:
            log = list(self._prediction_log)
            prediction_events = sum(1 for row in log if row.get("source") == "prediction")
            feedback_events = sum(1 for row in log if row.get("source") == "feedback")
            return {
                "total_events": len(log),
                "total_predictions": prediction_events,
                "total_feedback_events": feedback_events,
                "unique_providers": len(self._provider_counts),
                "unique_routes": len(self._route_counts),
                "pending_feedback": len(self._pending_feedback),
            }


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Tracks rolling inference latency in milliseconds."""

    def __init__(self, maxlen: int = 1000):
        self._lock = threading.Lock()
        self._latencies: deque = deque(maxlen=maxlen)

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._latencies.append(latency_ms)

    def stats(self) -> Dict[str, object]:
        with self._lock:
            data = list(self._latencies)
        if not data:
            return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None}
        arr = np.array(data)
        return {
            "count": len(arr),
            "mean_ms": round(float(arr.mean()), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "min_ms": round(float(arr.min()), 2),
            "max_ms": round(float(arr.max()), 2),
        }


# ---------------------------------------------------------------------------
# Global objects (loaded once at startup)
# ---------------------------------------------------------------------------

_bundle: Optional[Dict] = None
_feature_store = InMemoryFeatureStore()
_latency_tracker = LatencyTracker()
_drift_monitor: Optional[OnlineDriftMonitor] = None


def _force_single_thread_inference(estimator: object) -> None:
    """Best-effort hardening for Windows environments with worker spawn restrictions."""
    targets = [estimator]
    if hasattr(estimator, "named_steps"):
        targets.extend(list(getattr(estimator, "named_steps", {}).values()))
    for obj in targets:
        if hasattr(obj, "n_jobs"):
            try:
                setattr(obj, "n_jobs", 1)
            except Exception:
                pass


def load_bundle() -> Optional[Dict]:
    global _bundle, _drift_monitor, _feature_store
    if not BUNDLE_PATH.exists():
        return None
    try:
        _bundle = joblib.load(BUNDLE_PATH)
        _force_single_thread_inference(_bundle.get("estimator"))
        inference_reference = _bundle.get("inference_reference", {})
        if isinstance(inference_reference, dict):
            _feature_store.bootstrap_defaults(inference_reference.get("history_defaults"))
        # Initialise online drift monitor with calibration reference probabilities
        ref_proba = _bundle.get("val_calib_proba")
        if ref_proba is not None:
            _drift_monitor = OnlineDriftMonitor(
                reference_proba=ref_proba,
                labels=_bundle["labels"],
                window_size=500,
            )
        return _bundle
    except Exception as exc:
        print(f"[WARNING] Could not load model bundle: {exc}")
        return None


def _load_json_report(filename: str) -> Optional[Dict]:
    """Safely load a JSON report artifact."""
    path = REPORTS_DIR / filename
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _required_artifact_status() -> Dict[str, object]:
    missing = [name for name in REQUIRED_REPORTS if not (REPORTS_DIR / name).exists()]
    present = len(REQUIRED_REPORTS) - len(missing)
    completion_pct = round((present / len(REQUIRED_REPORTS)) * 100.0, 1) if REQUIRED_REPORTS else 100.0
    return {
        "required_reports": list(REQUIRED_REPORTS),
        "missing_reports": missing,
        "present_reports_count": present,
        "required_reports_count": len(REQUIRED_REPORTS),
        "completion_pct": completion_pct,
        "all_present": len(missing) == 0,
    }


def _bundle_loaded() -> bool:
    return _bundle is not None and _bundle.get("estimator") is not None


# ---------------------------------------------------------------------------
# Real-time prediction core
# ---------------------------------------------------------------------------

def _normalized_request_keys(form_data: Dict[str, Any]) -> Dict[str, str]:
    origin = str(form_data.get("origin_airport", "LHR")).upper().strip() or "LHR"
    destination = str(form_data.get("destination_airport", "JFK")).upper().strip() or "JFK"
    provider_key = str(form_data.get("provider_key", "Unknown")).strip() or "Unknown"
    route = f"{origin}_{destination}"
    return {
        "origin": origin,
        "destination": destination,
        "provider_key": provider_key,
        "route": route,
    }

def _build_single_row_df(form_data: Dict[str, Any], bundle: Dict) -> pd.DataFrame:
    """Construct a single-row feature DataFrame from raw form input.

    Applies the same feature contract as training, including history
    features served from the in-memory feature store.
    """
    now = datetime.now(timezone.utc)
    search_dt = now
    dep_date_str = form_data.get("departure_date", "")
    try:
        departure_date = datetime.strptime(dep_date_str, "%Y-%m-%d")
    except Exception:
        departure_date = now + timedelta(days=30)

    days_to_departure = max(0, (departure_date.date() - now.date()).days)
    normalized = _normalized_request_keys(form_data)
    origin = normalized["origin"]
    destination = normalized["destination"]
    airline = str(form_data.get("airline_code", "BA")).upper().strip()
    route = normalized["route"]
    provider_key = normalized["provider_key"]

    history_feats = _feature_store.get_features(provider_key, route, now)

    row = {
        "trip_type": str(form_data.get("trip_type", "OneWay")),
        "cabin_class": str(form_data.get("cabin_class", "M")).upper(),
        "origin_airport": origin,
        "destination_airport": destination,
        "airline_code": airline,
        "route": route,
        "airline_route": f"{airline}_{route}",
        "provider_key": provider_key,
        "provider_route": f"{provider_key}_{route}",
        "provider_airline": f"{provider_key}_{airline}",
        "meta_engine": str(form_data.get("meta_engine", "Direct")),
        "referrer_domain": str(form_data.get("referrer_domain", "Unknown")),
        "market": str(form_data.get("market", "Unknown")),
        "locale": str(form_data.get("locale", "Unknown")),
        "device_os": str(form_data.get("device_os", "Unknown")),
        "days_to_departure": float(days_to_departure),
        "days_to_departure_bucket": _dtd_bucket(days_to_departure),
        "search_hour": float(search_dt.hour),
        "search_dayofweek": float(search_dt.weekday()),
        "search_month": float(search_dt.month),
        "departure_month": float(departure_date.month),
        "is_weekend_search": float(1 if search_dt.weekday() >= 5 else 0),
        "is_short_lead": float(1 if days_to_departure <= 7 else 0),
        "is_long_lead": float(1 if days_to_departure >= 60 else 0),
        "passenger_count": float(form_data.get("passenger_count", 1)),
        "adults": float(form_data.get("adults", 1)),
        "children": float(form_data.get("children", 0)),
        "infants": float(form_data.get("infants", 0)),
        "stop_count": float(form_data.get("stop_count", 0)),
        "is_multi_stop": float(1 if int(form_data.get("stop_count", 0)) >= 2 else 0),
        "unknown_provider": float(1 if provider_key == "Unknown" else 0),
        "price_total": float(form_data.get("price_total", 0) or 0),
        "route_day_min_price": float(form_data.get("price_total", 0) or 0),
        "route_day_median_price": float(form_data.get("price_total", 0) or 0),
        "price_gap_to_min": 0.0,
        "price_ratio_to_median": 1.0,
        "price_rank_in_route_day": 0.0,
        "offers_in_route_day": 0.0,
        "missing_price": float(1 if not form_data.get("price_total") else 0),
        "route_search_density": 1.0,
        "provider_offer_density": 1.0,
        **history_feats,
    }

    # Fill any missing columns from the feature contract with 0/Unknown
    feature_columns = bundle["feature_columns"]
    cat_features = bundle.get("categorical_features", [])
    for col in feature_columns:
        if col not in row:
            row[col] = "Unknown" if col in cat_features else 0.0

    df = pd.DataFrame([row])
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("Unknown")
    num_features = bundle.get("numeric_features", [])
    for col in num_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df[feature_columns]


def _dtd_bucket(days: int) -> str:
    if days <= 1:
        return "same_day"
    if days <= 3:
        return "1_3_days"
    if days <= 7:
        return "3_7_days"
    if days <= 14:
        return "7_14_days"
    if days <= 30:
        return "14_30_days"
    if days <= 90:
        return "30_90_days"
    if days <= 365:
        return "90_365_days"
    return "365_plus_days"


def _run_prediction(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """Core prediction function with latency measurement and drift recording."""
    if not _bundle_loaded():
        return {"error": "Model not loaded. Run training first: python -m src.train"}

    bundle = _bundle
    t0 = time.perf_counter()

    try:
        X = _build_single_row_df(form_data, bundle)
        raw_proba = predict_proba_aligned(
            bundle["estimator"], X, list(range(len(bundle["labels"])))
        )
        calibrated = align_calibrated_probabilities(
            bundle["calibrator"], raw_proba, list(range(len(bundle["labels"])))
        )
    except Exception as exc:
        return {"error": f"Prediction failed: {exc}"}

    latency_ms = (time.perf_counter() - t0) * 1000.0
    _latency_tracker.record(latency_ms)

    labels = bundle["labels"]
    proba_vector = calibrated[0]

    if _drift_monitor is not None:
        _drift_monitor.record(proba_vector)

    predicted_idx = int(np.argmax(proba_vector))
    predicted_label = labels[predicted_idx]

    # Register prediction for delayed feedback update (no pseudo-label counter update)
    now = datetime.now(timezone.utc)
    normalized = _normalized_request_keys(form_data)
    route = normalized["route"]
    provider_key = normalized["provider_key"]
    prediction_id = _feature_store.register_prediction(provider_key, route, predicted_label, now)

    probabilities = {label: round(float(proba_vector[i]), 4) for i, label in enumerate(labels)}

    color_map = {
        "bookable": "success",
        "price_changed": "warning",
        "unavailable": "danger",
        "technical_failure": "secondary",
    }

    return {
        "predicted_label": predicted_label,
        "predicted_label_display": predicted_label.replace("_", " ").title(),
        "probabilities": probabilities,
        "confidence": round(float(proba_vector[predicted_idx]) * 100, 1),
        "latency_ms": round(latency_ms, 2),
        "badge_class": color_map.get(predicted_label, "secondary"),
        "timestamp": now.isoformat(),
        "model_name": bundle.get("model_name", "unknown"),
        "calibrator_name": bundle.get("calibrator_name", "unknown"),
        "prediction_id": prediction_id,
        "feedback_required": True,
    }


# ---------------------------------------------------------------------------
# Routes - Pages
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    """Research dashboard - loads all training artifacts."""
    bundle_loaded = _bundle_loaded()
    artifact_status = _required_artifact_status()
    summary = _load_json_report("summary.json") or {}
    model_comparison = _load_json_report("model_comparison.json") or {}
    calibration = _load_json_report("calibration_comparison.json") or {}
    final_metrics = _load_json_report("final_test_metrics.json") or {}
    ablation = _load_json_report("ablation_results.json") or {}
    rolling_backtest = _load_json_report("rolling_backtest.json") or {}
    baselines = _load_json_report("naive_baselines.json") or {}
    shap = _load_json_report("shap_explainability.json") or {}
    perm_imp = _load_json_report("explainability_permutation_importance.json") or {}
    drift = _load_json_report("drift_analysis.json") or {}
    significance = _load_json_report("significance_tests.json") or {}
    eda = _load_json_report("eda_report.json") or {}
    split_meta = _load_json_report("split_metadata.json") or {}
    feature_sel = _load_json_report("feature_selection.json") or {}
    subgroup = _load_json_report("subgroup_evaluation.json") or {}
    learning_curves = _load_json_report("learning_curves.json") or {}
    error_analysis = _load_json_report("error_analysis.json") or {}
    policy = _load_json_report("policy_simulation_results.json") or {}
    class_presence = _load_json_report("class_presence.json") or {}
    provider_holdout = _load_json_report("provider_holdout_evaluation.json") or {}

    live_stats = _feature_store.stats()
    latency_stats = _latency_tracker.stats()
    drift_check = _drift_monitor.check_drift() if _drift_monitor else {"status": "no_monitor"}

    return render_template(
        "index.html",
        bundle_loaded=bundle_loaded,
        summary=summary,
        model_comparison=model_comparison.get("rows", []),
        calibration=calibration.get("rows", []),
        final_metrics=final_metrics,
        ablation=ablation.get("rows", []),
        rolling_backtest=rolling_backtest.get("rows", []),
        baselines=baselines.get("rows", []),
        shap=shap,
        perm_imp=perm_imp,
        drift=drift,
        significance=significance,
        eda=eda,
        split_meta=split_meta,
        feature_sel=feature_sel,
        subgroup=subgroup,
        learning_curves=learning_curves.get("rows", []),
        error_analysis=error_analysis,
        policy=policy,
        class_presence=class_presence,
        provider_holdout=provider_holdout,
        live_stats=live_stats,
        latency_stats=latency_stats,
        drift_check=drift_check,
        artifact_status=artifact_status,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    """Single-flight real-time prediction UI."""
    result = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()
        result = _run_prediction(form_data)

    bundle_loaded = _bundle_loaded()
    labels = _bundle["labels"] if bundle_loaded else ["bookable", "price_changed", "unavailable", "technical_failure"]
    recent = _feature_store.recent_events(n=5)

    return render_template(
        "predict.html",
        result=result,
        form_data=form_data,
        bundle_loaded=bundle_loaded,
        labels=labels,
        recent_predictions=recent,
        latency_stats=_latency_tracker.stats(),
    )


# ---------------------------------------------------------------------------
# Routes - REST API
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON REST API endpoint for real-time flight bookability prediction.

    Validates required fields before processing. Returns 400 for missing fields,
    503 if model is not loaded, 200 with calibrated probabilities on success.

    Required fields: origin_airport, destination_airport, airline_code, departure_date
    Optional: cabin_class, trip_type, meta_engine, provider_key, price_total,
              stop_count, passenger_count, market, locale, device_os
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json", "hint": "Set Content-Type: application/json header"}), 400

    form_data = request.get_json(force=True) or {}

    # Input validation schema
    REQUIRED_FIELDS = {"origin_airport", "destination_airport", "airline_code", "departure_date"}
    missing = REQUIRED_FIELDS - set(form_data.keys())
    validation_errors = []
    if missing:
        validation_errors.append(f"Missing required fields: {sorted(missing)}")
    if "departure_date" in form_data:
        try:
            from datetime import datetime as _dt
            _dt.strptime(str(form_data["departure_date"]), "%Y-%m-%d")
        except ValueError:
            validation_errors.append("departure_date must be in YYYY-MM-DD format")
    for field in ("origin_airport", "destination_airport"):
        if field in form_data:
            code = str(form_data[field]).strip().upper()
            if len(code) < 3 or len(code) > 4 or not code.isalnum():
                validation_errors.append(f"{field} must be a 3-4 character alphanumeric airport code")
    if validation_errors:
        return jsonify({"error": "Validation failed", "details": validation_errors}), 400

    result = _run_prediction(form_data)
    if "error" in result:
        return jsonify(result), 503
    return jsonify(result), 200


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return final test metrics as JSON."""
    final_metrics = _load_json_report("final_test_metrics.json")
    if not final_metrics:
        return jsonify({"error": "No metrics found. Run training first."}), 404
    return jsonify({
        "final_test_metrics": final_metrics,
        "summary": _load_json_report("summary.json"),
        "model_comparison": _load_json_report("model_comparison.json"),
        "naive_baselines": _load_json_report("naive_baselines.json"),
        "significance_tests": _load_json_report("significance_tests.json"),
    }), 200


@app.route("/api/latency", methods=["GET"])
def api_latency():
    """Return real-time inference latency statistics."""
    return jsonify({
        "latency_stats": _latency_tracker.stats(),
        "feature_store_stats": _feature_store.stats(),
        "recent_events": _feature_store.recent_events(n=10),
    }), 200


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """True-outcome feedback endpoint used to update online history features."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    payload = request.get_json(force=True) or {}

    true_label = str(payload.get("true_label", "")).strip()
    prediction_id = payload.get("prediction_id")
    provider_key = payload.get("provider_key")
    route = payload.get("route")
    result = _feature_store.apply_feedback(
        true_label=true_label,
        now=datetime.now(timezone.utc),
        prediction_id=str(prediction_id) if prediction_id else None,
        provider_key=str(provider_key) if provider_key else None,
        route=str(route) if route else None,
    )
    if result.get("status") != "ok":
        return jsonify(result), 400
    return jsonify(result), 200


@app.route("/api/drift", methods=["GET"])
def api_drift():
    """Return current concept drift status."""
    offline_drift = _load_json_report("drift_analysis.json")
    online_drift = _drift_monitor.check_drift() if _drift_monitor else {"status": "no_monitor"}
    return jsonify({
        "offline_drift_analysis": offline_drift,
        "online_prediction_drift": online_drift,
        "latency_stats": _latency_tracker.stats(),
    }), 200


@app.route("/api/explainability", methods=["GET"])
def api_explainability():
    """Return SHAP and permutation importance reports."""
    return jsonify({
        "shap": _load_json_report("shap_explainability.json"),
        "permutation_importance": _load_json_report("explainability_permutation_importance.json"),
        "feature_selection": _load_json_report("feature_selection.json"),
    }), 200


@app.route("/api/ablation", methods=["GET"])
def api_ablation():
    """Return ablation study results (isolated + cumulative)."""
    return jsonify({
        "ablation_results": _load_json_report("ablation_results.json"),
        "rolling_backtest": _load_json_report("rolling_backtest.json"),
        "provider_holdout": _load_json_report("provider_holdout_evaluation.json"),
    }), 200


@app.route("/api/eda", methods=["GET"])
def api_eda():
    """Return EDA report."""
    return jsonify(_load_json_report("eda_report.json") or {}), 200


@app.route("/api/significance", methods=["GET"])
def api_significance():
    """Return statistical significance test results (with Bonferroni correction)."""
    return jsonify(_load_json_report("significance_tests.json") or {}), 200


@app.route("/api/learning-curves", methods=["GET"])
def api_learning_curves():
    """Return learning curve data (train/val F1 at different training fractions)."""
    return jsonify(_load_json_report("learning_curves.json") or {}), 200


@app.route("/api/error-analysis", methods=["GET"])
def api_error_analysis():
    """Return error analysis report (top confident misclassifications)."""
    return jsonify(_load_json_report("error_analysis.json") or {}), 200


@app.route("/api/cost-sensitive", methods=["GET"])
def api_cost_sensitive():
    """Return cost-sensitive decision simulation results."""
    policy = _load_json_report("policy_simulation_results.json") or {}
    return jsonify(policy.get("cost_sensitive", {})), 200


@app.route("/api/class-presence", methods=["GET"])
def api_class_presence():
    """Return class presence audit report (zero-support class warnings)."""
    return jsonify(_load_json_report("class_presence.json") or {}), 200


@app.route("/api/partial-dependence", methods=["GET"])
def api_partial_dependence():
    """Return partial dependence plot metadata."""
    return jsonify(_load_json_report("partial_dependence.json") or {}), 200


@app.route("/api/artifacts", methods=["GET"])
def api_artifacts():
    """Return required artifact completeness status."""
    return jsonify(_required_artifact_status()), 200


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": _bundle_loaded(),
        "model_name": _bundle.get("model_name") if _bundle_loaded() else None,
        "calibrator_name": _bundle.get("calibrator_name") if _bundle_loaded() else None,
        "labels": _bundle.get("labels") if _bundle_loaded() else None,
        "feature_store_predictions": _feature_store.stats()["total_predictions"],
        "artifacts": _required_artifact_status(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), 200


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

with app.app_context():
    load_bundle()


if __name__ == "__main__":
    print("=" * 60)
    print("Real-Time Flight Bookability Prediction System")
    print("=" * 60)
    if _bundle_loaded():
        print(f"[OK] Model loaded: {_bundle['model_name']}")
        print(f"[OK] Calibrator: {_bundle['calibrator_name']}")
        print(f"[OK] Labels: {_bundle['labels']}")
    else:
        print("[WARNING] No model bundle found.")
        print("         Run: python -m src.train")
    artifact_status = _required_artifact_status()
    if not artifact_status["all_present"]:
        print(f"[WARNING] Missing required report artifacts: {artifact_status['missing_reports']}")
    print("-" * 60)
    print("Dashboard:        http://localhost:5000/")
    print("Predict UI:       http://localhost:5000/predict")
    print("REST API:         http://localhost:5000/api/predict  [POST]")
    print("Feedback API:     http://localhost:5000/api/feedback [POST]")
    print("Metrics API:      http://localhost:5000/api/metrics")
    print("Drift API:        http://localhost:5000/api/drift")
    print("Latency API:      http://localhost:5000/api/latency")
    print("Health check:     http://localhost:5000/api/health")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000)
