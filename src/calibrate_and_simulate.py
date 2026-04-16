import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, classification_report
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

TARGET_COL = 'outcome_label'
CAT_COLS = ['trip_type', 'origin_airport', 'destination_airport', 'airline_code', 'cabin_class', 'meta_engine', 'device_os']
NUM_COLS = ['days_to_departure', 'search_hour', 'search_day', 'dep_month', 'is_weekend', 'itinerary_segments',
             'airline_success_rate_7d', 'airline_tech_fail_rate_7d', 'route_airline_success_rate_7d', 'airline_route_share_7d',
             'cache_age_hours', 'price_usd_imputed', 'price_gap_to_min']

def load_and_temporal_split():
    df = pd.read_csv('data/processed/training_data.csv')
    df = df[df[TARGET_COL].isin(['bookable', 'price_changed', 'unavailable', 'technical_failure'])]
    
    df['prediction_time'] = pd.to_datetime(df['prediction_time'])
    df = df.sort_values('prediction_time')
    
    for c in CAT_COLS:
        df[c] = df[c].fillna('Unknown').astype('category')
        
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)
        else:
            df[c] = 0

    # Chronological Split
    train_size = int(len(df) * 0.6)
    calib_size = int(len(df) * 0.8)
    
    train_df = df.iloc[:train_size]
    calib_df = df.iloc[train_size:calib_size]
    test_df = df.iloc[calib_size:]
    
    classes = sorted(list(df[TARGET_COL].unique()))
    class_map = {c: i for i, c in enumerate(classes)}
    
    def format_split(split_df):
        X = split_df[CAT_COLS + NUM_COLS]
        y = split_df[TARGET_COL].map(class_map).values
        return X, y, split_df
        
    X_tr, y_tr, df_tr = format_split(train_df)
    X_ca, y_ca, df_ca = format_split(calib_df)
    X_te, y_te, df_te = format_split(test_df)
    
    return X_tr, y_tr, df_tr, X_ca, y_ca, df_ca, X_te, y_te, df_te, classes

class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0
        
    def fit(self, logits, y_true):
        def obj(temp):
            scaled_logits = logits / temp[0]
            probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)
            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            return log_loss(y_true, probs)
            
        res = minimize(obj, [1.0], bounds=[(1e-3, 10.0)])
        self.temperature = res.x[0]
        
    def predict_proba(self, logits):
        scaled_logits = logits / self.temperature
        probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)
        return probs

class MultiClassIsotonicCalibrator:
    def __init__(self):
        self.calibrators = []
        
    def fit(self, probs, y_true, classes):
        self.calibrators = []
        for i in range(len(classes)):
            iso = IsotonicRegression(out_of_bounds='clip')
            y_binary = (y_true == i).astype(int)
            iso.fit(probs[:, i], y_binary)
            self.calibrators.append(iso)
            
    def predict_proba(self, probs):
        calibrated = np.zeros_like(probs)
        for i, iso in enumerate(self.calibrators):
            calibrated[:, i] = iso.transform(probs[:, i])
        # Normalize to maintain probability simplex
        calibrated = np.clip(calibrated, 1e-15, 1 - 1e-15)
        calibrated = calibrated / np.sum(calibrated, axis=1, keepdims=True)
        return calibrated

def expected_calibration_error(y_true, y_probs, n_bins=10):
    y_pred = np.argmax(y_probs, axis=1)
    conf = np.max(y_probs, axis=1)
    
    acc = y_pred == y_true
    ece = 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_mask = (conf > bins[i]) & (conf <= bins[i+1])
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(acc[bin_mask])
            bin_conf = np.mean(conf[bin_mask])
            bin_weight = np.sum(bin_mask) / len(conf)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    return ece

def plot_reliability_diagram(y_true, y_probs, title="Reliability Diagram"):
    y_pred = np.argmax(y_probs, axis=1)
    conf = np.max(y_probs, axis=1)
    acc = (y_pred == y_true).astype(float)
    
    bins = np.linspace(0, 1, 10 + 1)
    bin_accs = []
    bin_confs = []
    
    for i in range(10):
        mask = (conf > bins[i]) & (conf <= bins[i+1])
        if np.sum(mask) > 0:
            bin_accs.append(np.mean(acc[mask]))
            bin_confs.append(np.mean(conf[mask]))
            
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly Calibrated')
    plt.plot(bin_confs, bin_accs, 'o-', label='Model Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def run_advanced_evaluation():
    print("Loading data for Out-of-Time backtest...")
    X_tr, y_tr, df_tr, X_ca, y_ca, df_ca, X_te, y_te, df_te, classes = load_and_temporal_split()
    
    cb_clf = CatBoostClassifier(
        iterations=150, learning_rate=0.1, loss_function='MultiClass', 
        cat_features=CAT_COLS, verbose=0, thread_count=-1
    )
    
    print("Training base model (CatBoost)...")
    cb_clf.fit(X_tr, y_tr)
    
    print("\n--- Calibration Phase (Phase 5) ---")
    calib_logits = cb_clf.predict(X_ca, prediction_type='RawFormulaVal')
    calib_probs = cb_clf.predict_proba(X_ca)
    
    temp_scaler = TemperatureScaler()
    temp_scaler.fit(calib_logits, y_ca)
    
    iso_cal = MultiClassIsotonicCalibrator()
    iso_cal.fit(calib_probs, y_ca, classes)
    
    print("\n--- Out-Of-Time Test Evaluation (Phase 6 Drift/Stress testing) ---")
    test_logits = cb_clf.predict(X_te, prediction_type='RawFormulaVal')
    uncal_probs = cb_clf.predict_proba(X_te)
    cal_probs_temp = temp_scaler.predict_proba(test_logits)
    cal_probs_iso = iso_cal.predict_proba(uncal_probs)
    
    ll_uncal = log_loss(y_te, uncal_probs)
    ll_temp = log_loss(y_te, cal_probs_temp)
    ll_iso = log_loss(y_te, cal_probs_iso)
    
    ece_uncal = expected_calibration_error(y_te, uncal_probs)
    ece_temp = expected_calibration_error(y_te, cal_probs_temp)
    ece_iso = expected_calibration_error(y_te, cal_probs_iso)
    
    print(f"Uncalibrated -> Log Loss: {ll_uncal:.4f} | ECE: {ece_uncal:.4f}")
    print(f"Temp-Scaled  -> Log Loss: {ll_temp:.4f}  | ECE: {ece_temp:.4f}")
    print(f"Isotonic     -> Log Loss: {ll_iso:.4f} | ECE: {ece_iso:.4f}")
    
    # We choose Isotonic for the diagrams and simulation if it performs best (which it usually does on ECE)
    best_probs = cal_probs_iso if ece_iso < ece_uncal else uncal_probs
    
    print("\nRendering Reliability Diagram for best calibrator...")
    plot_reliability_diagram(y_te, best_probs, "Calibrated Reliability Diagram")
    
    print("\n--- Policy / Operational Simulation (Phase 6) ---")
    # Simulate a realistic Metasearch environment cost matrix
    costs = np.array([20, -5, -15, -5]) # aligned with classes
    
    bookable_idx = classes.index('bookable')
    
    print("Simulating Reranking / Bid Shading Expected Value (EV)...")
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for t in thresholds:
        shown_mask = (best_probs[:, bookable_idx] >= t)
        
        actual_outcomes = y_te[shown_mask]
        if len(actual_outcomes) > 0:
            outcome_costs = np.array([costs[y] for y in actual_outcomes])
            total_profit = outcome_costs.sum()
            avg_profit = outcome_costs.mean()
        else:
            total_profit = 0
            avg_profit = 0
            
        coverage = shown_mask.mean() * 100
        print(f"Threshold P(Book) > {t:.2f} | Coverage: {coverage:5.1f}% | Total EV Profit: ${total_profit:,.0f} | Avg/Shown: ${avg_profit:.2f}")

if __name__ == '__main__':
    run_advanced_evaluation()
