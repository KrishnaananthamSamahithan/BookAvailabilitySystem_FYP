import pandas as pd
import numpy as np
import os
import json
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, f1_score

TARGET_COL = 'outcome_label'

def load_and_prep_ablation_data():
    df = pd.read_csv('data/processed/training_data.csv')
    df = df[df[TARGET_COL].isin(['bookable', 'price_changed', 'unavailable', 'technical_failure'])]
    
    df['prediction_time'] = pd.to_datetime(df['prediction_time'])
    df = df.sort_values('prediction_time')
    
    # Target Mapping
    classes = sorted(list(df[TARGET_COL].unique()))
    class_map = {c: i for i, c in enumerate(classes)}
    df['target'] = df[TARGET_COL].map(class_map)
    
    # Splitting logic
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    return train_df, val_df, classes

def train_and_evaluate(train_df, val_df, cat_cols, num_cols, classes, name="Model"):
    print(f"\n--- Training {name} ---")
    X_train = train_df[cat_cols + num_cols].copy()
    y_train = train_df['target'].values
    X_val = val_df[cat_cols + num_cols].copy()
    y_val = val_df['target'].values
    
    for c in cat_cols:
        X_train[c] = X_train[c].fillna('Unknown').astype(str)
        X_val[c] = X_val[c].fillna('Unknown').astype(str)
        
    for c in num_cols:
        X_train[c] = X_train[c].fillna(0)
        X_val[c] = X_val[c].fillna(0)
        
    clf = CatBoostClassifier(
        iterations=100, learning_rate=0.1, loss_function='MultiClass',
        cat_features=cat_cols, auto_class_weights='Balanced', verbose=0, thread_count=-1
    )
    
    clf.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    val_probs = clf.predict_proba(X_val)
    val_preds = clf.predict(X_val)
    
    ll = log_loss(y_val, val_probs)
    macro_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
    
    print(f"[{name}] Log-Loss: {ll:.4f} | Macro F1: {macro_f1:.4f}")
    return ll, macro_f1

def run_ablation_studies():
    print("Starting Structural Feature Ablation Studies (RO4 Phase)")
    train_df, val_df, classes = load_and_prep_ablation_data()
    
    # Define Cascading Feature Blocks
    base_cat = ['trip_type', 'origin_airport', 'destination_airport', 'airline_code', 'cabin_class', 'device_os']
    base_num = ['days_to_departure', 'search_hour']
    
    temporal_num = [
        'airline_success_rate_7d', 'airline_tech_fail_rate_7d', 
        'route_airline_success_rate_7d', 'airline_route_share_7d'
    ]
    
    market_num = ['cache_age_hours', 'price_usd_imputed', 'price_gap_to_min']
    
    results = {}
    
    # 1. Base Model
    ll_base, f1_base = train_and_evaluate(train_df, val_df, base_cat, base_num, classes, name="1. Base Topology")
    results['Base'] = {'LogLoss': ll_base, 'MacroF1': f1_base}
    
    # 2. Base + Temporal Model
    ll_temp, f1_temp = train_and_evaluate(train_df, val_df, base_cat, base_num + temporal_num, classes, name="2. Base + Temporal Dynamics")
    results['Base+Temporal'] = {'LogLoss': ll_temp, 'MacroF1': f1_temp}
    
    # 3. Full Engine
    ll_full, f1_full = train_and_evaluate(train_df, val_df, base_cat, base_num + temporal_num + market_num, classes, name="3. Full Engineered Pipeline")
    results['FullPipeline'] = {'LogLoss': ll_full, 'MacroF1': f1_full}
    
    # Print Summary Analysis Table
    print("\n==================================")
    print("      ABLATION STUDY RESULTS      ")
    print("==================================")
    print(pd.DataFrame(results).T.to_string())
    
    pd.DataFrame(results).T.to_json('metrics/ablation_results.json', indent=4)
    print("\nSaved `ablation_results.json` for persistence framework.")

if __name__ == '__main__':
    run_ablation_studies()
