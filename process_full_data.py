import pandas as pd
import numpy as np
from urllib.parse import urlparse
import os

# Configuration
INPUT_FILE = 'tbl_SearchTracking_Merged.csv'
OUTPUT_FILE = 'processed_flight_data_full.csv'
CHUNK_SIZE = 100000  # Process 100k rows at a time

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

def process_chunk(chunk):
    # 1. Date Parsing & Feature Engineering
    # Ensure date columns are strings before parsing
    chunk['DepDate'] = chunk['DepDate'].astype(str)
    chunk['LandingTime'] = chunk['LandingTime'].astype(str)

    # Coerce errors to NaT
    chunk['departure_date'] = pd.to_datetime(chunk['DepDate'], errors='coerce')
    chunk['booking_time'] = pd.to_datetime(chunk['LandingTime'], errors='coerce')
    
    # Calculate days to departure
    chunk['days_to_departure'] = (chunk['departure_date'] - chunk['booking_time']).dt.days
    
    # 2. Extract Temporal Features
    chunk['search_hour'] = chunk['booking_time'].dt.hour
    chunk['search_day'] = chunk['booking_time'].dt.dayofweek
    chunk['dep_month'] = chunk['departure_date'].dt.month
    chunk['is_weekend'] = chunk['search_day'].isin([5, 6]).astype(int)
    
    # Filter out negative days (data errors) or invalid days
    chunk = chunk[chunk['days_to_departure'] >= 0]
    
    # 3. Meta Engine Extraction
    chunk['meta_engine'] = chunk['PreviousPage'].apply(get_meta_engine)
    
    # 4. Rename and Select Columns
    chunk = chunk.rename(columns={
        'SearchType': 'trip_type',
        'Origin': 'origin_airport',
        'Destination': 'destination_airport',
        'Airline': 'airline_code',
        'Class': 'cabin_class',
        'Status': 'outcome_label'
    })
    
    # 5. Handle Categorical Missing Values
    chunk['airline_code'] = chunk['airline_code'].fillna('Unknown')
    chunk['cabin_class'] = chunk['cabin_class'].fillna('Unknown')
    
    # Select final features
    features = [
        'trip_type', 'origin_airport', 'destination_airport', 
        'airline_code', 'cabin_class', 'meta_engine', 
        'days_to_departure', 'search_hour', 'search_day', 
        'dep_month', 'is_weekend', 'outcome_label'
    ]
    
    # Ensure all columns exist
    if not all(col in chunk.columns for col in features):
        return pd.DataFrame() # Skip bad chunks if any
        
    return chunk[features].dropna()

def main():
    print(f"Propcessing {INPUT_FILE} in chunks of {CHUNK_SIZE}...")
    
    # Check if input exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # Delete output if exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    chunk_count = 0
    total_rows = 0
    
    # Read CSV in chunks
    # Note: We need to handle the header correctly based on previous analysis
    # The file has a header, but let's rely on pandas reading the first row as header if we don't specify header=None
    # However, create_optimization logic used 'read_csv' default which infers header.
    
    try:
        with pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE) as reader:
            for chunk in reader:
                processed_chunk = process_chunk(chunk)
                
                if not processed_chunk.empty:
                    # Append to output file
                    # Write header only for the first chunk
                    header = (chunk_count == 0)
                    processed_chunk.to_csv(OUTPUT_FILE, mode='a', index=False, header=header)
                    
                    total_rows += len(processed_chunk)
                
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(f"Processed {chunk_count} chunks...")
                    
    except Exception as e:
        print(f"An error occurred: {e}")
        
    print(f"Processing complete. Saved {total_rows} rows to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
