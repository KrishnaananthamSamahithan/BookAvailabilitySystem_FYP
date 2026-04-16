# Offline Multi-Class Flight Bookability Prediction Using Calibrated Machine Learning

## Project Purpose

This project implements a research-grade machine learning pipeline for predicting the outcome category of a flight offer in an offline or batch metasearch evaluation setting.

The final classifier predicts one of four classes:

1. `bookable`
2. `price_changed`
3. `unavailable`
4. `technical_failure`

Rows that map to `ambiguous` are preserved during preprocessing for auditability, but they are excluded from the default supervised training and evaluation path.

## What The Model Predicts

The model is a multiclass classifier that predicts the most likely outcome state of a flight offer at the moment the offer is observed. In the current raw export, `LandingTime` is treated as the configurable proxy for the prediction-time event because it is the best available timestamp for the real-time decision moment.

## Why CatBoost Is A Strong Candidate

CatBoost is suitable for this problem because the data is tabular, mixed-type, and includes many categorical fields such as route, airline, cabin, provider proxy, referrer domain, and locale. CatBoost is robust with missing values, handles categorical features well, and usually performs strongly on nonlinear tabular problems without requiring fragile manual encoding.

## Why Calibration Is Necessary

Raw model probabilities are not automatically trustworthy. In metasearch decision systems, predicted probabilities are used for suppression and reranking decisions, so overconfident or underconfident outputs can lead to poor business logic. The final pipeline therefore compares multiple calibration strategies before choosing a final calibrated output.

## Why Temporal Splitting Is Required

Bookability is time-dependent. Provider behavior, route behavior, search mix, and operational failure patterns can shift over time. Random train/test splits would leak future information and overestimate performance. The final pipeline uses chronological train, validation, and test windows, plus a rolling backtest utility.

## Why Point-In-Time Features Matter

Only information available before the outcome occurs is allowed in the final model. This means:

- no target-conditioned synthetic features in the final path
- no history features that look at the current row outcome
- all rolling reliability features use prior-only logic such as `shift(1)` or equivalent grouped cumulative calculations

## Final Architecture

The final coherent implementation lives in `src/`:

- `src/config.py`: central configuration and path management
- `src/schema.py`: schema resolution and alias-based validation
- `src/labels.py`: canonical label engineering
- `src/preprocessing.py`: cleaned preprocessing pipeline with audit logging
- `src/features.py`: valid feature engineering only
- `src/split.py`: temporal split and rolling backtest helpers
- `src/models.py`: model comparison for logistic regression, random forest, and CatBoost
- `src/calibration.py`: none, temperature scaling, one-vs-rest isotonic, and multinomial logistic calibration
- `src/metrics.py`: classification and calibration metrics
- `src/evaluation.py`: comparison table helpers
- `src/reporting.py`: subgroup evaluation, calibration parity, and schema diagnostics
- `src/simulation.py`: suppression and reranking proxy simulation
- `src/train.py`: main training pipeline
- `src/inference.py`: batch inference entry point aligned with the final feature contract

Legacy prototype code is kept outside the final path and should not be treated as the authoritative implementation.

## Feature Groups

The final pipeline uses only valid feature families:

### Topology / Basic
- trip type
- cabin class
- origin airport
- destination airport
- airline code
- route
- airline-route interaction
- provider proxy key
- referrer domain / meta engine
- market / locale / device if available
- passenger count
- stop count and multi-stop flags

### Temporal
- days to departure
- search hour
- search day of week
- search month
- departure month
- weekend search flag
- days-to-departure bucket
- short-lead / long-lead flags
- route search density
- provider offer density

### Price
- only used when a real price column exists in the raw logs
- if price is unavailable in the raw logs, the feature family is omitted safely and documented rather than faked

### Reliability / History
- provider prior count
- provider prior bookable rate
- provider prior unavailable rate
- route prior count
- route prior bookable rate
- provider-route prior bookable rate
- airline-route prior unavailable rate
- minutes since previous provider event
- minutes since previous route event
- minutes since previous provider-route event

## Important Leakage Fixes

The old research notebook contained synthetic features conditioned on `outcome_label`, such as synthetic cache age and target-derived price behavior. Those features are not defensible for a final model and are removed from the final core pipeline.

The final path does **not**:

- generate cache age from the target
- generate price gaps from the target
- invent provider health from labels
- use current-row outcomes inside rolling features

## How To Run Training

From the project root:

```bash
python3 -m src.train
```

This will:

- preprocess the raw dataset
- save the processed dataset
- engineer valid features
- perform temporal splitting
- compare models
- compare calibrators
- run final test evaluation
- run ablation studies
- run rolling backtests
- run policy simulations
- save the final model bundle and all report artifacts

## How To Run Inference

After training has produced `artifacts/models/final_model_bundle.joblib`:

```bash
python3 -m src.inference --input path/to/input.csv --output path/to/predictions.csv
```

The inference script accepts an input CSV, applies the same preprocessing logic, uses the saved feature contract, and emits calibrated probabilities plus a predicted label.

## Saved Artifacts

The final training script is designed to save:

- processed dataset
- preprocessing audit
- schema resolution report
- split metadata
- prediction-time metadata
- label diagnostics
- feature inventory
- feature availability audit
- model comparison table
- calibration comparison table
- subgroup evaluation and calibration parity
- final test metrics
- ablation results
- rolling backtest results
- policy simulation results
- summary report
- final model bundle

## Raw Data Limitations

The current raw log export does not expose some stronger real-world signals directly. In particular:

- a clean provider identifier is not guaranteed, so the pipeline falls back to a provider proxy
- a stable real price field is not clearly available in the current export
- stronger market-health and cache-health signals are therefore omitted rather than invented

That limitation is documented intentionally. The final pipeline avoids synthetic validity and prefers honest omissions over leakage.
