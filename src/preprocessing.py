import os
import pandas as pd
import json
from tqdm import tqdm

def load_json(filepath, columns=None):
    """Load a JSON file as a DataFrame."""
    # Get total line count for progress bar
    with open(filepath, 'r') as file:
        total_lines = sum(1 for _ in file)
    
    data = []
    with open(filepath, 'r') as file:
        for line in tqdm(file, total=total_lines, desc=f"Loading {os.path.basename(filepath)}"):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                tqdm.write(f"Error in line: {e}")
    
    df = pd.DataFrame(data)
    if columns:
        return df[columns]
    return df

def preprocess_business(filepath):
    """Preprocess business dataset."""
    tqdm.write("Processing business data...")
    df = load_json(filepath, columns=['business_id', 'name', 'stars', 'categories', 'review_count'])
    df['categories'] = df['categories'].fillna('Unknown')
    tqdm.write(f"Processed {len(df)} business records.")
    return df

def preprocess_reviews(filepath):
    """Preprocess review dataset."""
    tqdm.write("Processing review data...")
    df = load_json(filepath, columns=['review_id', 'user_id', 'business_id', 'stars', 'text', 'date'])
    
    # Handle missing values or incorrect formats
    tqdm.write("Filtering and cleaning review data...")
    df = df.dropna(subset=['review_id', 'user_id', 'business_id', 'stars', 'text', 'date'])
    tqdm.write(f"Processed {len(df)} review records.")
    return df

def preprocess_checkins(filepath):
    """Preprocess check-in dataset."""
    tqdm.write("Processing check-in data...")
    df = load_json(filepath, columns=['business_id', 'date'])
    tqdm.write("Calculating check-in counts...")
    df['checkin_count'] = df['date'].progress_apply(lambda x: len(x.split(',')))
    return df[['business_id', 'checkin_count']]

def preprocess_tips(filepath):
    """Preprocess tip dataset."""
    tqdm.write("Processing tips data...")
    df = load_json(filepath, columns=['business_id', 'text', 'date'])
    tqdm.write("Converting dates in tips...")
    df['date'] = pd.to_datetime(df['date'])
    return df

def preprocess_users(filepath):
    """Preprocess user dataset."""
    tqdm.write("Processing user data...")
    df = load_json(filepath, columns=['user_id', 'review_count', 'average_stars', 'fans'])
    tqdm.write(f"Processed {len(df)} user records.")
    return df

# Enable tqdm for pandas operations
tqdm.pandas()
