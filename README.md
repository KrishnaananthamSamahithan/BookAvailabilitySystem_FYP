# Offline Multi-Class Flight Bookability Prediction Using Calibrated Machine Learning

## Project Purpose

This project implements a **research-grade machine learning pipeline** for predicting the outcome
category of a flight offer in a flight metasearch evaluation setting.

The final classifier predicts one of four classes:

1. `bookable` — offer can be purchased at the quoted price
2. `price_changed` — offer exists but price differs from the quoted amount
3. `unavailable` — offer is no longer available
4. `technical_failure` — booking system error or timeout during verification

Rows that map to `ambiguous` are preserved during preprocessing for auditability
and included in a 5-class diagnostic experiment (Phase 21), but excluded from
the default supervised 4-class training path.

---

## Pipeline Overview (24 Phases)

| # | Phase | Output Artifact |
|---|-------|----------------|
| 1 | Preprocessing & EDA | `eda_report.json`, `processed.csv` |
| 2 | Label presence + zero-support warnings | `class_presence.json` |
| 3 | 4-way temporal split (no double-dipping) | `split_metadata.json` |
| 4 | Feature selection audit | `feature_selection.json`, `feature_inventory.json` |
| 5 | DummyClassifier baselines | `naive_baselines.json` |
| 6 | **5-model comparison** (LR, RF, CatBoost, XGBoost, LightGBM) | `model_comparison.json` |
| 7 | Calibration (5 methods incl. Dirichlet) | `calibration_comparison.json` |
| 8 | Final test evaluation (full metric suite) | `final_test_metrics.json`, `reliability_diagrams.png` |
| 9 | Statistical significance (McNemar + Bootstrap CI + **Bonferroni**) | `significance_tests.json` |
| 10 | Subgroup evaluation & calibration parity | `subgroup_evaluation.json`, `calibration_parity.json` |
| 11 | Ablation study (ISOLATED + CUMULATIVE, 3 seeds) | `ablation_results.json` |
| 12 | Rolling backtest (**5 windows**) | `rolling_backtest.json` |
| 13 | Policy simulation (suppression + reranking + **cost-sensitive**) | `policy_simulation_results.json` |
| 14 | Concept drift analysis (PSI/KS) | `drift_analysis.json`, `prediction_drift.json` |
| 15–16 | SHAP explainability + **dependency plots (PNG)** | `shap_explainability.json`, `shap_dependency_*.png` |
| 17 | Permutation importance | `explainability_permutation_importance.json` |
| 18 | **Partial dependence plots (PDP, PNG)** | `pdp_bookable.png`, `pdp_price_changed.png` |
| 19 | **Learning curves** (5 fractions) | `learning_curves.json` |
| 20 | **Error analysis** (top-N misclassified) | `error_analysis.json` |
| 21 | Ambiguous handling experiment (5-class vs 4-class) | `ambiguous_handling_experiment.json` |
| 22 | Provider holdout generalisation | `provider_holdout_evaluation.json` |
| 23 | Model bundle saved | `final_model_bundle.joblib` |
| 24 | Summary report | `summary.json` |

---

## What The Model Predicts

The model is a multiclass classifier that predicts the most likely outcome state of a flight offer
at the moment the offer is observed. `LandingTime` is treated as the configurable proxy for the
prediction-time event because it is the best available timestamp in the current raw log export.

---

## Model Architecture

The pipeline compares **five models** via a fair grid-search protocol:

| Model | Tuning Strategy |
|-------|-----------------|
| Logistic Regression | C ∈ {0.1, 0.5, 1.0, 2.0, 5.0} |
| **Random Forest** | n_estimators × max_depth × min_leaf (5 configs) |
| CatBoost | depth × lr × iterations (6 configs), eval_set **disabled during comparison** |
| XGBoost | max_depth × lr × n_estimators (5 configs) |
| LightGBM | num_leaves × lr × n_estimators (5 configs) |

> **Fix applied:** Random Forest previously used fixed defaults without grid search and was
> incorrectly selected as the best model. It now uses a proper 5-config grid search, ensuring
> fair comparison. CatBoost eval_set was disabled during the comparison phase to prevent
> information leakage.

---

## Calibration Architecture

Five calibrators are compared on a **clean, held-out val_calibration split** (no double-dipping):

| Calibrator | Method |
|-----------|--------|
| Identity | No calibration (baseline) |
| Temperature Scaling | Single scalar on logits (preserves rank) |
| OVR Isotonic | Class-wise non-parametric regression |
| Multinomial Logistic | Parametric Platt scaling (multi-class) |
| **Dirichlet** | Principled multiclass extension (Kull et al., 2019) |

Acceptance criteria strictly enforced: balanced_accuracy ≥ 0.40, macro_f1 ≥ 0.30,
degradation vs uncalibrated ≤ 0.03.

---

## Feature Groups

| Group | Features |
|-------|----------|
| **Topology** | route, airline, cabin, provider, referrer, market, locale, device, passenger count, stop count |
| **Temporal** | days-to-departure, DTD bucket, search hour, search dayofweek, search month, departure month, lead flags |
| **Price** | price_total, price_ratio_to_median, price_gap_to_min, price_rank (when available) |
| **Reliability/History** | provider/route/provider-route prior bookable/unavailable rates (shift-based, no leakage) |

All history features use **prior-only** logic (`cumsum - current_value`), preventing
target contamination from the same row's outcome.

---

## Important Leakage Fixes (Applied)

- All history features use `cumsum - current` or `shift(1)` — the current row's outcome never contributes to its own features.
- No synthetic cache age, price gaps, or provider health derived from `outcome_label`.
- CatBoost `eval_set` disabled during model comparison phase to avoid unfair information leakage.
- 4-way split eliminates calibration double-dipping.

---

## Bug Fixes Applied

### Critical: InMemoryFeatureStore (app.py)
**Bug:** `provider_prior_rate_price_changed = 1 - bookable_rate` — assumed a binary world.
**Fix:** Now tracks independent per-outcome counters for all 4 classes. Each rate is computed as `count(outcome) / total`.

### Critical: OnlineDriftMonitor (drift.py)
**Bug:** `list.pop(0)` is O(n) and not thread-safe under concurrent Flask requests.
**Fix:** Replaced with `collections.deque(maxlen=window_size)` (O(1)) and added `threading.Lock`.

### Critical: Random Forest not tuned (models.py)
**Bug:** RF was fitted with hardcoded defaults while all other models used grid search.
**Fix:** RF now has a 5-configuration grid search identical in rigor to other models.

### Critical: XGBoost deprecation warning (models.py)
**Bug:** `use_label_encoder=False` removed in XGBoost 2.x causes AttributeError.
**Fix:** Parameter removed.

### Critical: Missing models in comparison artifact
**Bug:** XGBoost and LightGBM were missing from saved `model_comparison.json`.
**Fix:** All 5 models always produce a result (skipped models use `_make_skipped_result`).

### Academic: McNemar tests without Bonferroni correction (significance.py)
**Bug:** 10 pairwise tests at α=0.05 leads to ~40% false-positive rate.
**Fix:** Bonferroni correction applied: corrected_alpha = 0.05 / n_pairs.

### Academic: Only 3 rolling backtest windows (config.py + train.py)
**Bug:** Config default was 3 windows; code claimed 5.
**Fix:** Config updated to 5 windows; all 5 are executed.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|------------|
| `/` | GET | Research dashboard |
| `/predict` | GET/POST | Real-time prediction UI |
| `/api/predict` | POST | REST API — validated input, calibrated probabilities |
| `/api/metrics` | GET | Final test metrics + model comparison |
| `/api/drift` | GET | Drift analysis (offline + online) |
| `/api/latency` | GET | Inference latency statistics |
| `/api/explainability` | GET | SHAP + permutation importance |
| `/api/ablation` | GET | Ablation results + rolling backtest |
| `/api/significance` | GET | McNemar + Bootstrap CI (Bonferroni corrected) |
| `/api/learning-curves` | GET | Learning curve data |
| `/api/error-analysis` | GET | Top confident misclassification analysis |
| `/api/cost-sensitive` | GET | Cost-sensitive simulation |
| `/api/class-presence` | GET | Zero-support class warnings |
| `/api/partial-dependence` | GET | PDP metadata |
| `/api/health` | GET | Health check |

### API Input Validation (POST /api/predict)

Required fields:
- `origin_airport` (string, IATA code)
- `destination_airport` (string, IATA code)
- `airline_code` (string, IATA code)
- `departure_date` (string, YYYY-MM-DD format)

Returns HTTP 400 with `{"error": "Validation failed", "details": [...]}` on invalid input.

---

## How To Run Training

```bash
python3 -m src.train
```

This runs all 24 pipeline phases and saves all artifacts to `artifacts/reports/`.

Expected runtime: ~20–60 minutes depending on dataset size and hardware.

## How To Run The Flask Application

```bash
python app.py
```

Or for production-mode serving:

```bash
flask --app app run --host=0.0.0.0 --port=5000
```

## How To Run Batch Inference

```bash
python3 -m src.inference \
    --input path/to/input.csv \
    --history path/to/history.csv \
    --output path/to/predictions.csv
```

---

## Raw Data Limitations (Documented)

- **Currency**: Raw logs contain mixed currencies (GBP, USD, EUR, etc.). Price features are
  computed on raw values without normalization. This is documented as a limitation and price
  features are used with caution. A currency normalization step using exchange rates would
  improve price feature quality in future work.
- **Provider identity**: A stable provider identifier is not guaranteed; the pipeline falls back
  to a composite key from domain and meta engine.
- **Real-time labelling**: The `InMemoryFeatureStore` bootstraps on predicted labels (not ground
  truth) because ground truth is not available in real-time. This is a known limitation of all
  real-time serving systems and is documented in the system design.
- **Temporal coverage**: The current dataset covers approximately 60 days. Wider temporal coverage
  would improve generalization.

---

## Artifact Inventory

| Artifact | Description |
|---------|------------|
| `eda_report.json` | Full EDA: label distribution, missing rates, cardinality, temporal density |
| `class_presence.json` | Zero-support class warnings, active/inactive labels |
| `split_metadata.json` | 4-way temporal split with date ranges and class distributions |
| `feature_inventory.json` | Feature groups, categorical/numeric classification |
| `feature_selection.json` | Mutual information scores and availability flags |
| `naive_baselines.json` | 4 DummyClassifier baseline strategies |
| `model_comparison.json` | All 5 model results with tuning metadata |
| `calibration_comparison.json` | All 5 calibrator results with acceptance flags |
| `final_test_metrics.json` | Complete test metrics: F1, ECE, PR-AUC, thresholds, reliability table |
| `significance_tests.json` | Bootstrap CIs + Bonferroni-corrected McNemar tests |
| `subgroup_evaluation.json` | Per-group metrics by provider, route, airline, DTD bucket |
| `calibration_parity.json` | Calibration quality across demographic/operational subgroups |
| `ablation_results.json` | ISOLATED + CUMULATIVE ablation (3 seeds each) |
| `rolling_backtest.json` | 5-window forward-chaining backtest |
| `policy_simulation_results.json` | Suppression + reranking + cost-sensitive simulation |
| `drift_analysis.json` | Feature drift (PSI + KS) train vs test |
| `prediction_drift.json` | Output probability drift analysis |
| `shap_explainability.json` | Per-class SHAP top-25 feature rankings |
| `shap_dependency_*.png` | SHAP dependence scatter plots for top 3 features |
| `explainability_permutation_importance.json` | Model-agnostic permutation importance |
| `partial_dependence.json` | PDP metadata |
| `pdp_bookable.png` | Partial dependence plots for bookable class |
| `pdp_price_changed.png` | Partial dependence plots for price_changed class |
| `learning_curves.json` | Train/val F1 at 5 training fractions |
| `error_analysis.json` | Top confident misclassifications with context |
| `ambiguous_handling_experiment.json` | 5-class vs 4-class comparison |
| `provider_holdout_evaluation.json` | Generalization to unseen providers |
| `reliability_diagrams.png` | Calibration reliability curves per class |
| `summary.json` | Pipeline summary: best model, best calibrator, all headline metrics |
| `final_model_bundle.joblib` | Serialized estimator + calibrator + feature contract |
