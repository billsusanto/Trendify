import os
import pandas as pd
from joblib import dump
from src.preprocessing import preprocess_business, preprocess_reviews, preprocess_users
from src.meta_model import train_meta_model
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # File paths
    raw_data_path = "data/raw/yelp_dataset/"
    processed_data_path = "data/processed/"
    models_path = "models/"
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    # User input to skip preprocessing and model training
    user_choice = input("Do you want to skip preprocessing and model training? (yes/no): ").strip().lower()

    if user_choice != "yes":
        # Preprocess datasets
        logger.info("Preprocessing data...")
        business_df = preprocess_business(os.path.join(raw_data_path, "yelp_academic_dataset_business.json"))
        review_df = preprocess_reviews(os.path.join(raw_data_path, "yelp_academic_dataset_review.json"))
        user_df = preprocess_users(os.path.join(raw_data_path, "yelp_academic_dataset_user.json"))

        # Save processed data
        logger.info("Saving processed data...")
        business_df.to_csv(os.path.join(processed_data_path, "business.csv"), index=False)
        review_df.to_csv(os.path.join(processed_data_path, "reviews.csv"), index=False)
        user_df.to_csv(os.path.join(processed_data_path, "users.csv"), index=False)

    else:
        logger.info("Skipping preprocessing and model training.")
        # Load processed data
        business_df = pd.read_csv(os.path.join(processed_data_path, "business.csv"))
        review_df = pd.read_csv(
            os.path.join(processed_data_path, "reviews.csv"),
            on_bad_lines="skip",
            engine="python"
        )
        user_df = pd.read_csv(os.path.join(processed_data_path, "users.csv"))

    # Calculate weighted average ratings per business using reviews
    logger.info("Calculating weighted ratings from reviews...")
    review_df["weight"] = review_df["text"].str.len()
    review_df["weight"] = review_df["weight"].fillna(1).replace(0, 1)

    # Add additional review-based features
    logger.info("Adding additional review-based features...")
    review_stats = review_df.groupby("business_id").agg(
        std_dev_of_ratings=("stars", "std"),
        review_text_avg_length=("text", lambda x: x.str.len().mean())
    ).reset_index()

    weighted_ratings = review_df.groupby("business_id").apply(
        lambda x: (x["stars"] * x["weight"]).sum() / x["weight"].sum()
    ).rename("weighted_rating").reset_index()

    # Merge weighted ratings and review stats with business data
    business_df = business_df.merge(weighted_ratings, on="business_id", how="left")
    business_df = business_df.merge(review_stats, on="business_id", how="left")

    # Deduct rating for businesses with no reviews
    logger.info("Handling businesses with no reviews...")
    penalty_value = business_df["stars"].min() - 1
    business_df["weighted_rating"] = business_df["weighted_rating"].fillna(penalty_value)
    business_df["std_dev_of_ratings"] = business_df["std_dev_of_ratings"].fillna(0)
    business_df["review_text_avg_length"] = business_df["review_text_avg_length"].fillna(0)

    # Debugging: Check merge results
    logger.info("Debugging: Business data after merging review stats...")
    print(business_df[["business_id", "stars", "review_count", "weighted_rating", "std_dev_of_ratings", "review_text_avg_length"]].head())

    # Prepare meta-model features
    logger.info("Preparing meta-model features...")
    global_mean_rating = business_df["weighted_rating"].mean()
    max_review_count = business_df["review_count"].max()

    meta_features = pd.DataFrame({
        "business_review_count": business_df["review_count"],
        "average_business_rating": business_df["weighted_rating"],
        "normalized_review_count": business_df["review_count"] / max_review_count,
        "rating_deviation": business_df["weighted_rating"] - global_mean_rating,
        "rating_review_interaction": (business_df["weighted_rating"] - global_mean_rating) 
                                     * (business_df["review_count"] / max_review_count),
        "std_dev_of_ratings": business_df["std_dev_of_ratings"],
        "review_text_avg_length": business_df["review_text_avg_length"],
        "user_review_count": user_df["review_count"].mean(),
        "user_average_rating": user_df["average_stars"].mean(),
    }).fillna(0)

    # Train the meta-model
    logger.info("Training the meta-model...")
    meta_labels = business_df["weighted_rating"]
    meta_model = train_meta_model(meta_features, meta_labels)

    # Save the meta-model
    dump(meta_model, os.path.join(models_path, "meta_model.joblib"))
    logger.info("Meta-model training completed.")

if __name__ == "__main__":
    main()
