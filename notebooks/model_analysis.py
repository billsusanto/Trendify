import os
import pandas as pd
from joblib import load
import logging
import matplotlib.pyplot as plt
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_processed_datasets():
    try:
        logger.info("Loading processed datasets...")
        business_df = pd.read_csv("../data/processed/business.csv")
        review_df = pd.read_csv("../data/processed/reviews.csv", on_bad_lines="skip", engine="python")
        user_df = pd.read_csv("../data/processed/users.csv")
        return business_df, review_df, user_df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

def prepare_meta_model_data(filtered_businesses, user_df, global_mean_rating):
    logger.info("Preparing meta-model data...")
    # Use global statistics
    user_avg_review_count = user_df["review_count"].mean()
    user_avg_average_rating = user_df["average_stars"].mean()

    return pd.DataFrame({
        "business_review_count": filtered_businesses["review_count"],
        "average_business_rating": filtered_businesses["stars"],
        "normalized_review_count": filtered_businesses["review_count"] / filtered_businesses["review_count"].max(),
        "rating_deviation": filtered_businesses["stars"] - global_mean_rating,
        "rating_review_interaction": (filtered_businesses["stars"] - global_mean_rating) 
                                     * (filtered_businesses["review_count"] / filtered_businesses["review_count"].max()),
        "std_dev_of_ratings": filtered_businesses["std_dev_of_ratings"],
        "review_text_avg_length": filtered_businesses["review_text_avg_length"],
        "user_review_count": [user_avg_review_count] * len(filtered_businesses),
        "user_average_rating": [user_avg_average_rating] * len(filtered_businesses),
    }).fillna(0)

def main():
    # Load processed datasets
    business_df, review_df, user_df = load_processed_datasets()

    # Validate business_df
    required_business_cols = {"categories", "review_count", "stars"}
    if not required_business_cols.issubset(business_df.columns):
        missing_cols = required_business_cols - set(business_df.columns)
        logger.error(f"Missing columns in business_df: {missing_cols}")
        raise KeyError("Business DataFrame does not contain required columns.")

    # User input: Category to filter businesses
    category = input("Enter a category to analyze (e.g., 'restaurants', 'cinemas', 'gyms'): ").strip().lower()

    # Filter businesses by category
    logger.info(f"Filtering businesses in category: {category}")
    filtered_businesses = business_df[business_df['categories'].str.contains(category, case=False, na=False)].copy()

    # Add variability using features from reviews
    logger.info("Enhancing filtered businesses with review-based features...")
    review_df["text_length"] = review_df["text"].str.len().fillna(0)
    review_stats = review_df.groupby("business_id").agg(
        std_dev_of_ratings=("stars", "std"),
        review_text_avg_length=("text_length", "mean")
    ).reset_index()

    # Merge review stats into filtered businesses
    filtered_businesses = filtered_businesses.merge(review_stats, on="business_id", how="left")
    filtered_businesses["std_dev_of_ratings"] = filtered_businesses["std_dev_of_ratings"].fillna(0)
    filtered_businesses["review_text_avg_length"] = filtered_businesses["review_text_avg_length"].fillna(0)

    # Compute global mean rating for rating deviation calculation
    global_mean_rating = business_df["stars"].mean()

    # Prepare meta-model data
    meta_X = prepare_meta_model_data(filtered_businesses, user_df, global_mean_rating)

    # Load pre-trained meta-model
    logger.info("Loading meta-model...")
    meta_model = load("../models/meta_model.joblib")

    # Predict trendiness scores
    logger.info("Predicting trendiness scores...")
    trendiness_scores = meta_model.predict(meta_X)

    # Normalize trendiness scores to enhance spread
    logger.info("Normalizing trendiness scores for better spread...")
    min_score, max_score = 0.5, 5.0
    trendiness_scores = min_score + (trendiness_scores - trendiness_scores.min()) * (max_score - min_score) / (trendiness_scores.max() - trendiness_scores.min())

    # Add visualization of trendiness scores
    plt.figure(figsize=(10, 6))
    plt.hist(trendiness_scores, bins=30, edgecolor='black')
    plt.title(f'Distribution of Trendiness Scores for {category.title()} Businesses')
    plt.xlabel('Trendiness Score')
    plt.ylabel('Number of Businesses')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Add trendiness scores to businesses
    filtered_businesses["trendiness_score"] = trendiness_scores

    # Load business names from raw JSON
    logger.info("Loading business names from raw JSON...")
    raw_business_df = pd.read_json("../data/raw/yelp_dataset/yelp_academic_dataset_business.json", lines=True)
    business_id_to_name = raw_business_df.set_index('business_id')['name'].to_dict()
    filtered_businesses['name'] = filtered_businesses['business_id'].map(business_id_to_name)

    # Rank businesses by trendiness
    logger.info("Ranking businesses by trendiness...")
    top_trendy_businesses = filtered_businesses.sort_values(
        by=["trendiness_score", "stars", "review_count"], ascending=[False, False, False]
    ).head(10)

    # Output results
    logger.info(f"Top 10 trendy businesses for category: {category}")
    print(top_trendy_businesses[["name", "stars", "review_count", "trendiness_score"]])

if __name__ == "__main__":
    main()