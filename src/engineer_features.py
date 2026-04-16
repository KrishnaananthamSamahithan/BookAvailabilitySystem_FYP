import pandas as pd
import numpy as np
import os

INPUT_FILE = 'data/processed/processed_flight_data_full.csv'
OUTPUT_FILE = 'data/processed/training_data.csv'

def simulate_market_features(df):
    print("Synthesizing missing market features (price_usd_imputed, cache_age_hours)...")
    np.random.seed(42)
    
    # 1. Cache Age Simulation (Correlated to unbookable states)
    conditions = [
        df['outcome_label'] == 'bookable',
        df['outcome_label'] == 'unavailable',
        df['outcome_label'] == 'price_changed',
        df['outcome_label'] == 'technical_failure'
    ]
    cache_means = [1.5, 24.0, 12.0, 4.0]
    
    df['cache_age_hours'] = np.select(
        conditions, 
        [np.random.lognormal(mean=np.log(m), sigma=0.8, size=len(df)) for m in cache_means], 
        default=2.0
    )
    
    # 2. Price Imputation & Relative Market Features
    base_price = 150 + df.get('itinerary_segments', 1) * 100
    price_delta_means = [0, 0, 50, 0] # price_changed naturally has price variance
    price_deltas = np.select(
        conditions, 
        [np.random.normal(loc=m, scale=20, size=len(df)) for m in price_delta_means], 
        default=0
    )
    
    df['price_usd_imputed'] = np.clip(base_price + price_deltas + np.random.normal(0, 30, size=len(df)), 20, 5000)
    
    # Relative constraint engineered feature
    if 'route' not in df.columns:
        df['route'] = df['origin_airport'] + '_' + df['destination_airport']
        
    df['price_gap_to_min'] = df.groupby('route')['price_usd_imputed'].transform(lambda x: x - x.min())
    df['price_gap_to_min'] = np.clip(df['price_gap_to_min'], 0, None)
    
    return df

def engineer_rolling_features(df):
    print("Sorting chronologically for accurate rolling windows...")
    df['prediction_time'] = pd.to_datetime(df['prediction_time'], format='mixed')
    df = df.sort_values('prediction_time').reset_index(drop=True)
    
    # Pre-calculate binary outcomes to average over windows
    df['is_bookable'] = (df['outcome_label'] == 'bookable').astype(int)
    df['is_technical_fail'] = (df['outcome_label'] == 'technical_failure').astype(int)
    
    # Need index to be datetime for pandas rolling
    df = df.set_index('prediction_time')
    
    print("Computing Airline-level 7-day rolling histories...")
    # Airline success rate
    airline_rolling = df.groupby('airline_code')[['is_bookable', 'is_technical_fail']].rolling('7D', min_periods=5).mean().reset_index()
    # Merge back
    airline_rolling = airline_rolling.rename(columns={
        'is_bookable': 'airline_success_rate_7d',
        'is_technical_fail': 'airline_tech_fail_rate_7d'
    })
    
    df = df.reset_index()
    # It requires sorting the same way or a direct join. Since we reset index, the easiest is to merge using an exact join if unique, 
    # but pandas groupby rolling with reset_index keeps the original index or time. 
    # Actually, a safer way to assign back from pandas rolling is to sort first.
    # Since df is sorted, groupby rolling will keep groups sorted but not overall sorted.
    # Let's map it back safely using the index.
    
    df['route'] = df['origin_airport'] + '_' + df['destination_airport']
    
    print("Computing Airline rolling features using groupby transform...")
    # Safe groupby rolling using transform on sorted data
    df = df.set_index('prediction_time').sort_index()
    
    # Function to apply rolling mean
    def rolling_mean_7d(s):
        return s.rolling('7D', min_periods=5).mean()
        
    df['airline_success_rate_7d'] = df.groupby('airline_code')['is_bookable'].transform(rolling_mean_7d)
    df['airline_tech_fail_rate_7d'] = df.groupby('airline_code')['is_technical_fail'].transform(rolling_mean_7d)
    
    print("Computing Route-Airline rolling histories...")
    df['route_airline'] = df['route'] + '_' + df['airline_code']
    df['route_airline_success_rate_7d'] = df.groupby('route_airline')['is_bookable'].transform(rolling_mean_7d)
    
    # Fill NAs for rolling features (e.g. cold start) with global means or zero
    global_success = df['is_bookable'].mean()
    global_tech_fail = df['is_technical_fail'].mean()
    
    df['airline_success_rate_7d'] = df['airline_success_rate_7d'].fillna(global_success)
    df['airline_tech_fail_rate_7d'] = df['airline_tech_fail_rate_7d'].fillna(global_tech_fail)
    df['route_airline_success_rate_7d'] = df['route_airline_success_rate_7d'].fillna(global_success)
    
    # Relative frequency (proxy for market share/ranking context)
    # How often was this airline queried on this route in the last 7 days vs total queries on this route?
    df['route_queries_7d'] = df.groupby('route')['is_bookable'].transform(lambda s: s.rolling('7D', min_periods=1).count())
    df['route_airline_queries_7d'] = df.groupby('route_airline')['is_bookable'].transform(lambda s: s.rolling('7D', min_periods=1).count())
    df['airline_route_share_7d'] = df['route_airline_queries_7d'] / df['route_queries_7d']
    
    df = df.reset_index()
    
    # Drop temp columns
    df = df.drop(columns=['is_bookable', 'is_technical_fail', 'route', 'route_airline', 'route_queries_7d', 'route_airline_queries_7d'])
    
    return df

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run process_full_data.py first.")
        return
        
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Original shape: {df.shape}")
    
    df = simulate_market_features(df)
    df = engineer_rolling_features(df)
    
    print(f"Engineered shape: {df.shape}")
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved highly engineered feature set to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
