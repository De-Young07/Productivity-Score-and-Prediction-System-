import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger(__name__)

def split_dataset(df):

    logger.info("Splitting dataset into train and test")

    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    logger.info(f"Train shape: {train.shape}")
    logger.info(f"Test shape: {test.shape}")

    return train, test