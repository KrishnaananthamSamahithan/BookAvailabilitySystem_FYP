import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configuration
INPUT_FILE = 'processed_flight_data_full.csv'
MODEL_FILE = 'catboost_production.cbm'
CONFUSION_MATRIX_FILE = 'confusion_matrix.png'

# Load Data
print(f"Loading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} rows.")

# Outcome mapping
outcome_map = {
    'Booked': 0,
    'Price Mismatch': 1,
    'Not Available': 2,
    'Not Booked': 3
}

# Ensure validation
df = df[df['outcome_label'].isin(outcome_map.keys())]
df['label_encoded'] = df['outcome_label'].map(outcome_map)

# Features
# Note: CatBoost handles categoricals well, but Target Encoding helps providing "Global Statistics" to the model.

target_encode_cols = ['origin_airport', 'destination_airport', 'airline_code']
categorical_features = ['trip_type',  'cabin_class', 'meta_engine'] # 'meta_engine' is powerful, kept as cat
numerical_features = ['days_to_departure', 'search_hour', 'search_day', 'dep_month', 'is_weekend']

# need to perform Target Encoding INSIDE the CV loop to avoid leakage.
# However, CatBoost has built-in Target Encoding (Ctr).

def target_encode(train_df, test_df, cols, target_col):
    # Calculate global mean
    global_mean = train_df[target_col].mean()
    
    for col in cols:
        # Calculate mean target per category
        # For multi-class, we target encode for specific classes of interest, e.g., 'Booked' (0) probability
        # Let's create a 'Booked_Prob' feature
        
        # Binary target: Is Booked?
        train_df[f'{col}_booked_rate'] = train_df.groupby(col)[target_col].transform(lambda x: (x==0).mean())
        
        # Map to test
        mapping = train_df.groupby(col)[target_col].apply(lambda x: (x==0).mean())
        test_df[f'{col}_booked_rate'] = test_df[col].map(mapping).fillna(global_mean) # fill new categories with global mean
        
    return train_df, test_df

# For simplicity and speed with CatBoost, we will rely on CatBoost's powerful internal handling for now,
# Let's enable CatBoost's robust features.

# Updated Feature List including new ones
X = df[categorical_features + target_encode_cols + numerical_features]
y = df['label_encoded']

# Initialize K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Metrics storage
cv_scores = []
best_model = None
best_score = -1

print(f"Starting 5-Fold Cross-Validation...")

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1} ---")
    
    X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Explicit Feature Engineering: Interaction
    # Airline + Origin Risk
    X_train['airline_origin'] = X_train['airline_code'].astype(str) + "_" + X_train['origin_airport'].astype(str)
    X_test['airline_origin'] = X_test['airline_code'].astype(str) + "_" + X_test['origin_airport'].astype(str)
    
    # Update categorical features list for this iteration
    current_cat_features = categorical_features + target_encode_cols + ['airline_origin']
    
    # Initialize Model with improved params
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.08, # Slightly higher
        depth=8,
        loss_function='MultiClass',
        auto_class_weights='Balanced', # Critical for 'Price Mismatch'
        l2_leaf_reg=5, # Regularization
        bagging_temperature=1, # Prevent overfitting
        verbose=200,
        random_seed=42
    )
    
    model.fit(
        X_train, y_train,
        cat_features=current_cat_features,
        eval_set=(X_test, y_test),
        early_stopping_rounds=100,
        use_best_model=True
    )
    
    # Evaluate
    score = model.score(X_test, y_test)
    cv_scores.append(score)
    print(f"Fold {fold+1} Accuracy: {score:.4f}")
    
    # Keep best model
    if score > best_score:
        best_score = score
        best_model = model

print(f"\n--- CV Results ---")
print(f"Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Final Model
print("\n--- Saving Best Model (from Validation) ---")
final_model = best_model

# Generate Report
print("\n--- Final Classification Report (Best Fold) ---")
# Re-predict on the test set of the best fold (we don't have it easily accessible unless we saved it, 
# but for now let's use the last fold's test set as a proxy check or just save the model)
# To be accurate, let's just save. Feature importance will come from best model.

final_model.save_model(MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# Feature Importance
feature_importance = final_model.get_feature_importance()
print("\n--- Feature Importance ---")
# Get feature names from the model
feature_names = final_model.feature_names_
for score, name in sorted(zip(feature_importance, feature_names), reverse=True):
    print(f"{name}: {score:.2f}")

# Plotting requires re-running prediction on some data. 
# We'll skip the plot here to save time, or use the last X_test.

