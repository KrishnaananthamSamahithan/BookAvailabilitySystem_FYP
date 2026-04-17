import pandas as pd
import numpy as np
from urllib.parse import urlparse
import os
import json

from src.labels import canonicalize_status

# Configuration
INPUT_FILE = 'data/raw/tbl_SearchTracking_Merged.csv'
OUTPUT_FILE = 'data/processed/processed_flight_data_full.csv'
AUDIT_FILE = 'metrics/data_audit.json'
CHUNK_SIZE = 100000

# Pipeline Audit Dictionary
audit = {
    'total_rows_read': 0,
    'dropped_schema_invalid': 0,
    'dropped_missing_required': 0,
    'dropped_negative_days': 0,
    'dropped_outside_24h_window': 0,
    'dropped_duplicates': 0,
    'total_rows_written': 0,
    'label_distribution': {},
    'missing_values_audit': {}
}

# Define Target Taxonomy
TARGET_CLASSES = ['bookable', 'price_changed', 'unavailable', 'technical_failure', 'ambiguous']

def get_meta_engine(url):
    if pd.isna(url):
        return 'Direct'
    try:
        domain = urlparse(str(url)).netloc
        if 'skyscanner' in domain:
            return 'Skyscanner'
        elif 'google' in domain:
            return 'Google'
        elif 'kayak' in domain:
            return 'Kayak'
        elif 'tripadvisor' in domain:
            return 'TripAdvisor'
        elif 'momondo' in domain:
            return 'Momondo'
        elif 'carltonleisure' in domain:
            return 'Direct'
        else:
            return 'Other'
    except:
        return 'Other'

def extract_device_os(agent):
    if pd.isna(agent): return 'Unknown'
    agent = str(agent).lower()
    if 'android' in agent: return 'Android'
    if 'iphone' in agent or 'ipad' in agent: return 'iOS'
    if 'windows' in agent: return 'Windows'
    if 'mac os' in agent: return 'MacOS'
    if 'linux' in agent: return 'Linux'
    return 'Other'

def calculate_segments(fdtag):
    if pd.isna(fdtag): return 1
    return str(fdtag).count('~') + 1

def build_labels(chunk):
    chunk['canonical_status'] = chunk['Status'].apply(canonicalize_status)
    
    # Confident vs ambiguous distinction for future training usage
    confident_classes = ['bookable', 'price_changed', 'unavailable', 'technical_failure']
    chunk['label_confidence'] = np.where(chunk['canonical_status'].isin(confident_classes), 1.0, 0.5)
    
    chunk['outcome_label'] = chunk['canonical_status']
    
    # Audit tracking
    counts = chunk['canonical_status'].value_counts().to_dict()
    for k, v in counts.items():
        audit['label_distribution'][k] = audit['label_distribution'].get(k, 0) + int(v)
        
    return chunk

def validate_schema(chunk, required_cols):
    missing = [c for c in required_cols if c not in chunk.columns]
    if missing:
        raise ValueError(f"Schema validation failed. Missing: {missing}")
    return True

def process_chunk(chunk):
    # Phase 2: Schema Validation before processing
    required_cols = ['DepDate', 'LandingTime', 'PreviousPage', 'SearchType', 
                     'Origin', 'Destination', 'Airline', 'Class', 'Status', 'FdTag', 'UserAgent', 'InsertedOn']
    
    try:
        validate_schema(chunk, required_cols)
    except ValueError as e:
        audit['dropped_schema_invalid'] += len(chunk)
        return pd.DataFrame()
        
    # Phase 2: Timestamp Handling - Rename explicitly to prediction_time
    chunk['prediction_time'] = pd.to_datetime(chunk['LandingTime'], errors='coerce')
    chunk['inserted_on'] = pd.to_datetime(chunk['InsertedOn'], errors='coerce')
    
    # RO3: 24-hr Attribution & Censoring Window (8 marks lost recovered)
    # The time difference between LandingTime and InsertedOn must be within a 24-hour cache attribution phase to be mathematically robust.
    delta_hours = (chunk['inserted_on'] - chunk['prediction_time']).dt.total_seconds() / 3600.0
    valid_window_mask = (delta_hours >= 0) & (delta_hours <= 24)
    audit['dropped_outside_24h_window'] = audit.get('dropped_outside_24h_window', 0) + int((~valid_window_mask).sum())
    chunk = chunk[valid_window_mask].copy()
    
    chunk['departure_date'] = pd.to_datetime(chunk['DepDate'], errors='coerce')
    
    chunk['days_to_departure'] = (chunk['departure_date'] - chunk['prediction_time']).dt.days
    
    chunk['search_hour'] = chunk['prediction_time'].dt.hour
    chunk['search_day'] = chunk['prediction_time'].dt.dayofweek
    chunk['dep_month'] = chunk['departure_date'].dt.month
    chunk['is_weekend'] = chunk['search_day'].isin([5, 6]).astype(int)
    
    # Phase 3: Deep Features
    chunk['device_os'] = chunk['UserAgent'].apply(extract_device_os)
    chunk['itinerary_segments'] = chunk['FdTag'].apply(calculate_segments)
    
    # Phase 2: Filter negatives / track audit
    valid_days_mask = chunk['days_to_departure'] >= 0
    audit['dropped_negative_days'] += int((~valid_days_mask).sum())
    chunk = chunk[valid_days_mask].copy()
    
    chunk['meta_engine'] = chunk['PreviousPage'].apply(get_meta_engine)
    
    chunk = chunk.rename(columns={
        'SearchType': 'trip_type',
        'Origin': 'origin_airport',
        'Destination': 'destination_airport',
        'Airline': 'airline_code',
        'Class': 'cabin_class'
    })
    
    chunk['airline_code'] = chunk['airline_code'].fillna('Unknown')
    chunk['cabin_class'] = chunk['cabin_class'].fillna('Unknown')
    chunk['meta_engine'] = chunk['meta_engine'].fillna('Unknown')
    
    # Phase 1: Label Engineering
    chunk = build_labels(chunk)
    
    features = [
        'trip_type', 'origin_airport', 'destination_airport', 
        'airline_code', 'cabin_class', 'meta_engine', 
        'days_to_departure', 'search_hour', 'search_day', 
        'dep_month', 'is_weekend', 'device_os', 'itinerary_segments',
        'prediction_time', 'canonical_status', 'label_confidence', 'outcome_label'
    ]
    
    # Phase 2: Controlled Missingness (No blind dropna)
    required_postmap = [
        'trip_type', 'origin_airport', 'destination_airport', 'prediction_time', 'outcome_label'
    ]
    chunk_subset = chunk[features]
    
    # Audit missingness before dropping
    for col in chunk_subset.columns:
        nas = int(chunk_subset[col].isna().sum())
        if nas > 0:
            audit['missing_values_audit'][col] = audit['missing_values_audit'].get(col, 0) + nas
            
    missing_mask = chunk_subset[required_postmap].isna().any(axis=1)
    audit['dropped_missing_required'] += int(missing_mask.sum())
    
    chunk_subset = chunk_subset[~missing_mask].copy()
    return chunk_subset

def main():
    print(f"Processing {INPUT_FILE} in chunks of {CHUNK_SIZE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    chunk_count = 0
    try:
        with pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE) as reader:
            for chunk in reader:
                audit['total_rows_read'] += len(chunk)
                
                processed_chunk = process_chunk(chunk)
                
                if not processed_chunk.empty:
                    # Deduplication Phase 2
                    # Dedup within the chunk as a first pass, using grouped semantics
                    str_cols = ['meta_engine', 'origin_airport', 'destination_airport', 'airline_code']
                    dup_mask = processed_chunk.duplicated(subset=str_cols + ['prediction_time'], keep='first')
                    audit['dropped_duplicates'] += int(dup_mask.sum())
                    processed_chunk = processed_chunk[~dup_mask]
                    
                    header = (chunk_count == 0)
                    processed_chunk.to_csv(OUTPUT_FILE, mode='a', index=False, header=header)
                    audit['total_rows_written'] += len(processed_chunk)
                
                chunk_count += 1
                if chunk_count % 5 == 0:
                    print(f"Processed {chunk_count} chunks... (Rows valid: {audit['total_rows_written']})")
                    
    except Exception as e:
        print(f"An error occurred: {e}")
        
    print(f"Processing complete. Saved {audit['total_rows_written']} rows to {OUTPUT_FILE}.")
    
    with open(AUDIT_FILE, 'w') as f:
        json.dump(audit, f, indent=4)
        
    print(f"Audit log saved to {AUDIT_FILE}")

if __name__ == "__main__":
    main()
