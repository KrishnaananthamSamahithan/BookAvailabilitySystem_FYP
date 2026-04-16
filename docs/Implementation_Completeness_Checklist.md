# Flight Bookability - Master Formative Tracking Checklist

This checklist tracks the implementation status of all the required improvements specified in `Flight_Bookability_Master_Formative_TrackSheet.xlsx` against the current Python codebase (`process_full_data.py`, `engineer_features.py`, `calibrate_and_simulate.py`, `experiment_runner.py`, and `train_production.py`).

## 1. Problem Formulation & Label Engineering
- [x] **Incorrect class definition**: Addressed. `TARGET_CLASSES` introduces the correct taxonomy (`bookable`, `price_changed`, `unavailable`, `technical_failure`, `ambiguous`).
- [x] **Missing technical failure class**: Addressed. Handled explicitly in the `canonicalize_label()` logic.
- [x] **No ambiguous handling**: Addressed. Ambiguous outcomes tracked and treated as lower confidence labels in the mapping logic.
- [x] **Direct mapping only**: Addressed. Robust normalization rules applied prior to mapping strings.
- [x] **No label confidence**: Addressed. `label_confidence` implemented in `build_labels()`.

## 2. Data Preprocessing & Cleaning
- [x] **Columns used before checking existence**: Addressed. Explicit `validate_schema()` introduced to prevent silent errors.
- [x] **Timestamp Handling**: Addressed. Reframed correctly around `prediction_time` vs `departure_date`.
- [x] **No preprocessing audit**: Addressed. Fully covered via `audit` JSON logging tracking missing/drops seamlessly.
- [x] **Data Quality (No duplicate removal)**: Addressed. Strict duplicate string rules processed using `dup_mask`.
- [x] **Missing Data (Blind dropna)**: Addressed. Controlled subset missingness handled gracefully using `required_postmap`.
- [x] **Date Logic (Negative days)**: Addressed. Included correctly in the audit tracker before elimination.

## 3. Feature Engineering
- [x] **Feature Depth**: Addressed. Shifted to a strong functional feature set spanning OS, temporal factors, categories.
- [ ] **Price Features**: **Pending**. `price_gap_to_min` and `price_ratio_to_median` relative market-context logic has not been added (possibly due to price column missing entirely from initial raw extracts).
- [x] **Provider Signals**: Addressed. Carrier historical reliability implemented via rolling/expanding operations.
- [ ] **Temporal Features**: **Pending**. No recency features tracking the raw time difference since the `minutes_since_prev` metric.
- [x] **Competition Features**: Addressed. Formulated partially through Route market share queries metrics.
- [x] **Missing Indicators**: Addressed. Added explicitly (e.g. `missing_airline`).

## 4. Temporal Design & Leakage
- [x] **Temporal Split**: Addressed. Strong chronological splits executed via `temporal_split()`.
- [x] **Leakage Risk**: Addressed. Proper `shift(1)` logic and expanding means used reliably.
- [ ] **Validation Robustness**: **Pending**. The codebase uses a static chronological split (e.g., 60/20/20 or 80/20) and not dynamic rolling backtests over multiple out-of-time sliding windows.

## 5. Model Development
- [x] **Model Variety**: Addressed. Extended natively across Logistic Regression, Random Forest, and CatBoost in `experiment_runner.py`.
- [ ] **Model Selection**: **Pending**. Evaluation occurs over multiple algorithms, but algorithmic selection is hardcoded to CatBoost instead of a formalized runtime validation auto-picker.
- [x] **Hyperparameters**: Addressed / Partial. Solid initial tuning defined statically, but no automated Bayesian tuning (e.g., Optuna) has been integrated.
- [x] **Imbalance Handling**: Addressed. Automated `class_weight='balanced'` and `loss_function='MultiClass'` adopted effectively.

## 6. Calibration
- [ ] **Calibration Depth**: **Pending**. Currently limited primarily to the `TemperatureScaler`. Classical isotonic loops have not explicitly been contrasted yet.
- [x] **Calibration Type**: Addressed. Custom `TemperatureScaler` optimized smoothly.
- [ ] **Multiclass Calibration**: **Pending**. The stronger non-parametric structural `MultinomialLogitCalibrator` remains unimplemented.
- [ ] **Selection**: **Pending**. Does not auto-select the best calibrator via validation thresholds yet.

## 7. Evaluation Metrics
- [x] **Metrics Coverage**: Addressed. Macro and Weighted F1 cleanly reported through `classification_report`.
- [x] **Classification Insight**: Addressed. Complete confusion matrices/classification diagnostics are dumped effectively.
- [x] **Calibration Metrics**: Addressed. Added explicit reporting using generic Brier formulations and calibrated Log Loss estimates.
- [x] **Class-Level Analysis**: Addressed. Detailed slice-based classification reporting integrated across subgroup outputs.

## 8. Business Logic & Policy Simulation
- [x] **Simulation Quality**: Addressed. Successfully shifted representation of actual policy through an accurate Metasearch cost-simulation structure (Commission vs Penalties).
- [x] **Suppression Logic**: Addressed. Risk thresholds mapped systematically across probability distributions (e.g. Coverage logic).
- [ ] **Ranking Logic**: **Pending**. Suppression logic successfully evaluates the filter matrix, but continuous probability reranking (`rank_score_ml`) hasn't natively been scored against standard heuristics.
- [x] **Business Metrics**: Addressed. Explicitly estimating coverage against profit / average UX impacts successfully.

## 9. Engineering & Pipeline
- [x] **Pipeline Structure**: Addressed. Extracted comprehensively away from Jupyter to modular production structures like `process_full_data.py`, `train_production.py`.
- [x] **Reproducibility**: Addressed. Persisted strongly through `audit.json` mappings tracking schema and volume shifts.
- [ ] **Feature Tracking**: **Pending**. Formal JSON maps holding categorical schemas vs variables are absent.
- [x] **Model Persistence**: Addressed. Output reliably bound to `catboost_production.cbm`.
