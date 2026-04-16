import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# import lightgbm as lgb
# import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import time

TARGET_COL = 'outcome_label'
CAT_COLS = ['trip_type', 'origin_airport', 'destination_airport', 'airline_code', 'cabin_class', 'meta_engine', 'device_os']
NUM_COLS = ['days_to_departure', 'search_hour', 'search_day', 'dep_month', 'is_weekend', 'itinerary_segments',
             'airline_success_rate_7d', 'airline_tech_fail_rate_7d', 'route_airline_success_rate_7d', 'airline_route_share_7d']

def load_and_split():
    df = pd.read_csv('data/processed/training_data.csv')
    df = df[df[TARGET_COL].isin(['bookable', 'price_changed', 'unavailable', 'technical_failure'])]
    
    # Needs to be sorted chronologically
    df['prediction_time'] = pd.to_datetime(df['prediction_time'])
    df = df.sort_values('prediction_time')
    
    for c in CAT_COLS:
        df[c] = df[c].fillna('Unknown').astype('category')
        
    for c in NUM_COLS:
        df[c] = df[c].fillna(0)

    # 80/20 chronological split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    X_train = train_df[CAT_COLS + NUM_COLS]
    y_train = train_df[TARGET_COL]
    
    X_val = val_df[CAT_COLS + NUM_COLS]
    y_val = val_df[TARGET_COL]
    
    # Target encoding for metrics
    classes = sorted(list(y_train.unique()))
    class_map = {c: i for i, c in enumerate(classes)}
    y_train_num = y_train.map(class_map)
    y_val_num = y_val.map(class_map)
    
    return X_train, X_val, y_train_num, y_val_num, classes

def eval_model(name, y_true, y_probs, classes):
    y_pred = np.argmax(y_probs, axis=1)
    ll = log_loss(y_true, y_probs, labels=range(len(classes)))
    
    # Brier Score approx for multiclass: sum of binary brier scores / n_classes
    brier = 0
    for i in range(len(classes)):
        y_true_bin = (y_true == i).astype(int)
        y_prob_bin = y_probs[:, i]
        brier += brier_score_loss(y_true_bin, y_prob_bin)
    brier /= len(classes)
    
    print(f"--- {name} ---")
    print(f"Log Loss: {ll:.4f} | Multiclass Brier: {brier:.4f}")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    print("\n")
    return ll, brier

def run_experiments():
    print("Loading data...")
    X_train, X_val, y_train, y_val, classes = load_and_split()
    
    print("Classes distribution (Train):")
    print(pd.Series(y_train).value_counts().sort_index().rename(index=dict(enumerate(classes))))
    
    # 1. Logistic Regression (Baseline)
    print("Training Logistic Regression...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS)
        ])
    lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=500, class_weight='balanced', n_jobs=-1))
    ])
    start = time.time()
    lr.fit(X_train, y_train)
    print(f"LR took {time.time() - start:.1f}s")
    lr_probs = lr.predict_proba(X_val)
    eval_model("Logistic Regression Baseline", y_val, lr_probs, classes)
    
    # 2. LightGBM (Disabled due to libomp.dylib missing on macOS)
    # print("Training LightGBM...")
    # lgb_train = lgb.Dataset(X_train, label=y_train)
    # lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    # params = {
    #     'objective': 'multiclass',
    #     'num_class': len(classes),
    #     'metric': 'multi_logloss',
    #     'boosting_type': 'gbdt',
    #     'learning_rate': 0.1,
    #     'verbose': -1
    # }
    # start = time.time()
    # gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_val])
    # print(f"LightGBM took {time.time() - start:.1f}s")
    # lgb_probs = gbm.predict(X_val)
    # eval_model("LightGBM", y_val, lgb_probs, classes)

    # 3. XGBoost (Disabled due to libomp on Mac) / Replaced with Random Forest
    print("Training Random Forest...")
    rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=15, n_jobs=-1))
    ])
    start = time.time()
    rf.fit(X_train, y_train)
    print(f"Random Forest took {time.time() - start:.1f}s")
    rf_probs = rf.predict_proba(X_val)
    eval_model("Random Forest", y_val, rf_probs, classes)
    
    # 4. CatBoost
    print("Training CatBoost...")
    cb_clf = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        loss_function='MultiClass',
        cat_features=CAT_COLS,
        verbose=0,
        thread_count=-1
    )
    start = time.time()
    cb_clf.fit(X_train, y_train, eval_set=(X_val, y_val))
    print(f"CatBoost took {time.time() - start:.1f}s")
    cb_probs = cb_clf.predict_proba(X_val)
    eval_model("CatBoost", y_val, cb_probs, classes)

if __name__ == '__main__':
    run_experiments()
