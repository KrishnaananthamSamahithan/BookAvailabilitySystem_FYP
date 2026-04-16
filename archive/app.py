import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
import datetime
import os

# --- Configuration ---
MODEL_FILE = 'catboost_production.cbm'
PAGE_TITLE = "Real-Time Flight Bookability Predictor"
PAGE_ICON = "✈️"

# --- Load Model ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    model = CatBoostClassifier()
    model.load_model(MODEL_FILE)
    return model

model = load_model()

# --- Page Setup ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.markdown("""
This prototype demonstrates the **ML-Centric** capability of predicting flight offer outcomes in real-time.
By analyzing flight details and search context, we predict the likelihood of a booking, price mismatch, or unavailability.
""")

if model is None:
    st.error(f"Model file `{MODEL_FILE}` not found. Please ensure the model is trained and saved in the directory.")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("Flight & Search Details")

# Define options based on previous analysis
trip_type_options = ['OneWay', 'Return', 'Multicity']
cabin_class_options = ['M', 'W', 'C', 'F', 'ANY'] # Economy, Premium, Business, First
meta_engine_options = ['Skyscanner', 'Google', 'Kayak', 'Momondo', 'Other'] # Common ones + what we saw
# Top 50 airports (simplified list for prototype + others)
common_airports = [
    'LHR', 'JFK', 'DXB', 'HKG', 'SIN', 'LAX', 'CDG', 'AMS', 'FRA', 'BKK', 
    'IST', 'KUL', 'ICN', 'DEL', 'BOM', 'SYD', 'MEL', 'YYZ', 'YVR', 'SFO',
    'ORD', 'ATL', 'DFW', 'DEN', 'MIA', 'MCO', 'LAS', 'SEA', 'CLT', 'PHX',
    'IAH', 'EWR', 'MSP', 'DTW', 'BOS', 'SLC', 'PHL', 'BWI', 'SAN', 'TPA'
]
common_airlines = [
    'BA', 'AA', 'DL', 'UA', 'EK', 'QR', 'SQ', 'CX', 'LH', 'AF', 'KL', 'QF', 'NZ', 'VS', 'EY',
    'TK', 'AI', '9W', 'MH', 'TG', 'JL', 'NH', 'KE', 'OZ', 'CI', 'BR', 'CA', 'MU', 'CZ'
]

with st.sidebar.form("prediction_form"):
    st.subheader("1. Flight Information")
    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox("Origin Airport", common_airports + ["Other"])
        if origin == "Other":
            origin = st.text_input("Enter Origin Code (e.g. NRT)", value="NRT").upper()
    with col2:
        destination = st.selectbox("Destination Airport", common_airports + ["Other"])
        if destination == "Other":
            destination = st.text_input("Enter Dest Code (e.g. HND)", value="HND").upper()

    col3, col4 = st.columns(2)
    with col3:
        airline = st.selectbox("Airline", common_airlines + ["Other"])
        if airline == "Other":
            airline = st.text_input("Enter Airline Code (e.g. JL)", value="JL").upper()
    with col4:
        flight_date = st.date_input("Flight Date", datetime.date.today() + datetime.timedelta(days=30))

    st.subheader("2. Search Context")
    col5, col6 = st.columns(2)
    with col5:
        trip_type = st.selectbox("Trip Type", trip_type_options)
    with col6:
        cabin_class = st.selectbox("Cabin Class", cabin_class_options)

    meta_engine = st.selectbox("Meta Search Engine", meta_engine_options)
    
    # Hidden / Calculated features context
    st.subheader("3. Simulation Context")
    search_date = st.date_input("Search Date (Simulation)", datetime.date.today())
    search_time = st.time_input("Search Time", datetime.time(12, 0))

    submit_button = st.form_submit_button("Predict Outcome")

# --- Inference Logic ---
if submit_button:
    # Feature Engineering on the Fly
    
    # 1. Calculate Days to Departure
    days_to_departure = (flight_date - search_date).days
    if days_to_departure < 0:
        st.error("Flight Date cannot be in the past relative to Search Date.")
        st.stop()
        
    # 2. Extract Time Features
    search_hour = search_time.hour
    search_day = search_date.day 
    dep_month = flight_date.month
    is_weekend = 1 if search_date.weekday() >= 5 else 0
    
    # 3. Create DataFrame for Model
    # Order mapping: ['trip_type', 'cabin_class', 'meta_engine', 'origin_airport', 'destination_airport', 'airline_code', 'days_to_departure', 'search_hour', 'search_day', 'dep_month', 'is_weekend', 'airline_origin']
    
    # Handle custom text inputs if used
    
    input_data = {
        'trip_type': trip_type,
        'cabin_class': cabin_class,
        'meta_engine': meta_engine,
        'origin_airport': origin,
        'destination_airport': destination,
        'airline_code': airline,
        'days_to_departure': float(days_to_departure),
        'search_hour': float(search_hour),
        'search_day': float(search_day),
        'dep_month': float(dep_month),
        'is_weekend': int(is_weekend),
        'airline_origin': f"{airline}_{origin}" # Interaction feature
    }
    
    # Feature Order must match training EXACTLY
    # Based on evaluate_model.py: 
    # feature_order = base_categorical + target_encode_cols + numerical_features + ['airline_origin']
    # base_categorical = ['trip_type',  'cabin_class', 'meta_engine']
    # target_encode_cols = ['origin_airport', 'destination_airport', 'airline_code']
    # numerical_features = ['days_to_departure', 'search_hour', 'search_day', 'dep_month', 'is_weekend']
    
    feature_order = [
         'trip_type',  'cabin_class', 'meta_engine',
         'origin_airport', 'destination_airport', 'airline_code',
         'days_to_departure', 'search_hour', 'search_day', 'dep_month', 'is_weekend',
         'airline_origin'
    ]
    
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_order]
    
    # Display Input Data
    st.subheader("Input Features")
    st.dataframe(input_df)

    # --- Prediction ---
    try:
        # Get Probabilities
        probs = model.predict_proba(input_df)[0]
        # Get Class
        pred_class_idx = np.argmax(probs)
        
        # Mapping from train_production.py
        outcome_map_reverse = {
            0: 'Booked',
            1: 'Price Mismatch',
            2: 'Not Available',
            3: 'Not Booked'
        }
        
        predicted_label = outcome_map_reverse[pred_class_idx]
        
        # --- Display Results ---
        st.divider()
        st.subheader("Prediction Results")
        
        # Main Result
        color_map = {
            'Booked': 'green',
            'Price Mismatch': 'orange',
            'Not Available': 'red',
            'Not Booked': 'grey'
        }
        st.markdown(f"### Predicted Outcome: <span style='color:{color_map[predicted_label]}'>{predicted_label}</span>", unsafe_allow_html=True)
        
        # Probability Bars
        st.write("#### Probability Distribution")
        
        # Create a nice bar chart
        prob_df = pd.DataFrame({
            'Outcome': list(outcome_map_reverse.values()),
            'Probability': probs
        })
        
        st.bar_chart(prob_df.set_index('Outcome'))
        
        # Detailed Probs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Booked", f"{probs[0]:.2%}")
        c2.metric("Price Mismatch", f"{probs[1]:.2%}")
        c3.metric("Not Available", f"{probs[2]:.2%}")
        c4.metric("Not Booked", f"{probs[3]:.2%}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Ensure that the input features match what the model expects (e.g. unknown categories depending on how CatBoost was trained).")

