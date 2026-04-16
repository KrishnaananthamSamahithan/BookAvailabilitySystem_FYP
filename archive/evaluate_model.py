import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
print("Loading data...")
df = pd.read_csv('processed_flight_data_full.csv')

# Outcome mapping
outcome_map = {
    'Booked': 0,
    'Price Mismatch': 1,
    'Not Available': 2,
    'Not Booked': 3
}

df = df[df['outcome_label'].isin(outcome_map.keys())]
df['label_encoded'] = df['outcome_label'].map(outcome_map)

# Features
# Features to match Training Order EXACTLY
# 1. Base Features
target_encode_cols = ['origin_airport', 'destination_airport', 'airline_code']
base_categorical = ['trip_type',  'cabin_class', 'meta_engine']
numerical_features = ['days_to_departure', 'search_hour', 'search_day', 'dep_month', 'is_weekend']

# 2. Create Interaction Feature
df['airline_origin'] = df['airline_code'].astype(str) + "_" + df['origin_airport'].astype(str)

# 3. Construct X in correct order
# Training was: X = df[base_categorical + target_encode_cols + numerical_features]
# Then X['airline_origin'] was added.
# So the order is: base_categorical, target_encode_cols, numerical_features, 'airline_origin'

feature_order = base_categorical + target_encode_cols + numerical_features + ['airline_origin']
X = df[feature_order]
y = df['label_encoded']

# Load Model
print("Loading model...")
model = CatBoostClassifier()
model.load_model('catboost_production.cbm')

# Predict
print("Predicting...")
# use the whole dataset for evaluation summary (or split?)
# To be fast, let's use the last 20% separate from training "conceptually" 
# (though the model was trained on folds, so it has seen most data).
# Let's just evaluate on everything to see "Training" performance at least.

y_pred = model.predict(X)

print("\n--- Classification Report ---")
print(classification_report(y, y_pred, target_names=outcome_map.keys()))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y, y_pred))

# Binary Classification Check
# What if we only predicted Booked vs Not Booked?
y_binary = (y == 0).astype(int)
y_pred_binary = (y_pred == 0).astype(int) 
# Note: y_pred returns [0], need to flatten? CatBoost predict returns column vector?
y_pred_flat = y_pred.ravel()
y_pred_binary = (y_pred_flat == 0).astype(int)

print("\n--- Binary Classification (Booked vs Rest) ---")
print(classification_report(y_binary, y_pred_binary, target_names=['Not Booked', 'Booked']))
