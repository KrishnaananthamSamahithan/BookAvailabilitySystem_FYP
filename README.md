# Real-Time Flight Bookability Prediction

This project implements a machine learning pipeline to predict the outcome of flight offers (Booked, Price Mismatch, Not Available, Not Booked) in real-time.

## Project Structure

-   `app.py`: Streamlit application for live demonstration.
-   `train_production.py`: Script to train the CatBoost model.
-   `evaluate_model.py`: Script to evaluate the trained model.
-   `process_full_data.py`: Script to process raw data (if applicable).
-   `catboost_production.cbm`: The trained model file.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Live Demonstration (Streamlit App)
Run the interactive web application to test the model in real-time.

```bash
streamlit run app.py
```

### 2. Training the Model
To retrain the model on the dataset (`processed_flight_data_full.csv`):

```bash
python3 train_production.py
```
This will save the model to `catboost_production.cbm` and generate a confusion matrix.

### 3. Evaluating the Model
To run specific evaluation metrics on the saved model:

```bash
python3 evaluate_model.py
```

## Model Details
The system uses a **CatBoostClassifier** trained on historical flight search data. It predicts the probability of 4 classes:
-   **Booked**: User converts.
-   **Price Mismatch**: Price changes after click.
-   **Not Available**: Flight is no longer available.
-   **Not Booked**: User clicks but does not book (no error).
