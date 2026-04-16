import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

INPUT_FILE = 'data/processed/processed_flight_data_full.csv'
MODEL_FILE = 'models/catboost_production.cbm'
METRICS_FILE = 'metrics/metrics.json'

# Phase 1 Taxonomy Implementation
outcome_map = {
    'bookable': 0,
    'price_changed': 1,
    'unavailable': 2,
    'technical_failure': 3,
    # 'ambiguous': excluded from main training loop to prevent noise.
}

def engineer_features(df):
    print("Engineering features...")
    # Ensure sorted by prediction time to prevent leakage
    df['prediction_time'] = pd.to_datetime(df['prediction_time'], format='mixed', errors='coerce')
    df = df.sort_values('prediction_time').reset_index(drop=True)
    
    # Phase 3: Missing indicators
    df['missing_airline'] = (df['airline_code'] == 'Unknown').astype(int)
    
    # Phase 3: Advanced Categorical Interaction
    df['route'] = df['origin_airport'].astype(str) + "_" + df['destination_airport'].astype(str)
    
    # Phase 3 & 4: Temporal proxies (expanding reliable rate)
    # Target definition for rate logic
    df['is_bookable'] = (df['outcome_label'] == 'bookable').astype(float)
    
    # Shift by 1 to prevent leakage: row's outcome cannot predict itself
    # Calculate rolling historical reliability of a provider
    df['provider_bookable_rate'] = df.groupby('meta_engine')['is_bookable'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(df['is_bookable'].mean())
    
    # Calculate rolling historical reliability of a route
    df['route_bookable_rate'] = df.groupby('route')['is_bookable'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(df['is_bookable'].mean())
    
    df.drop(columns=['is_bookable'], inplace=True)
    
    return df

def temporal_split(df, train_frac=0.7, val_frac=0.15):
    # Phase 4: Strict Chronological Split
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    print(f"Temporal Split -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

def train_eval_models(X_train, y_train, X_val, y_val, cat_features):
    print("Training models...")
    
    # Phase 5: CatBoost with Balanced Class Weights for Rare classes (e.g. price_changed)
    cb_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.08,
        depth=6,
        loss_function='MultiClass',
        auto_class_weights='Balanced',
        verbose=100,
        random_seed=42
    )
    
    cb_model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        use_best_model=True
    )
    
    # Predict
    val_preds = cb_model.predict(X_val)
    val_probs = cb_model.predict_proba(X_val)
    
    macro_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
    ll = log_loss(y_val, val_probs)
    
    print(f"CatBoost Validation - Macro F1: {macro_f1:.4f}, Log Loss: {ll:.4f}")
    
    return cb_model

def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records.")
    
    # Filter valid labels
    df = df[df['outcome_label'].isin(outcome_map.keys())].copy()
    df['target'] = df['outcome_label'].map(outcome_map)
    print(f"Filtered to {len(df)} confident rows (dropped ambiguous).")
    
    df = engineer_features(df)
    
    categorical_features = ['trip_type', 'cabin_class', 'meta_engine', 'origin_airport', 'destination_airport', 'airline_code', 'route']
    numerical_features = ['days_to_departure', 'search_hour', 'search_day', 'dep_month', 'is_weekend', 
                          'missing_airline', 'provider_bookable_rate', 'route_bookable_rate']
    
    target = 'target'
    
    # Fill categoricals
    for col in categorical_features:
        df[col] = df[col].fillna('Unknown').astype(str)
        
    train, val, test = temporal_split(df)
    
    X_train = train[categorical_features + numerical_features]
    y_train = train[target]
    
    X_val = val[categorical_features + numerical_features]
    y_val = val[target]
    
    model = train_eval_models(X_train, y_train, X_val, y_val, categorical_features)
    
    # Phase 7: Test Evaluation
    X_test = test[categorical_features + numerical_features]
    y_test = test[target]
    test_preds = model.predict(X_test)
    
    print("\n--- Test Classification Report ---")
    inverse_map = {v: k for k, v in outcome_map.items()}
    # We use unique targets present in y_test
    unique_targets = sorted(y_test.unique())
    target_names = [inverse_map[t] for t in unique_targets]
    
    print(classification_report(y_test, test_preds, labels=unique_targets, target_names=target_names, zero_division=0))
    
    model.save_model(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()
