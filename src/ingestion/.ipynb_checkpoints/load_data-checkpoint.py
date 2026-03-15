import pandas as pd
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_raw_dataset(file_path):

    logger.info("Loading dataset")

    df = pd.read_csv(file_path)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def save_raw_dataset(df):

    raw_dir = "C:/Users/USER/Projects/Productivity Score Prediction System/Datasets/raw"
    os.makedirs(raw_dir, exist_ok=True)

    output_path = os.path.join(raw_dir, "productivity_raw.csv")

    df.to_csv(output_path, index=False)

    logger.info(f"Raw dataset saved to {output_path}")


def run_ingestion():

    input_file = "C:/Users/USER/Projects/Productivity Score Prediction System/Datasets/raw/social_media_vs_productivity.csv"

    df = load_raw_dataset(input_file)

    save_raw_dataset(df)

    logger.info("Data ingestion completed")


if __name__ == "__main__":
    run_ingestion()