import pandas as pd
import os

from src.preprocessing.split_data import split_dataset
from src.preprocessing.imputation import impute_missing_values
from src.preprocessing.encoding import encode_features
from src.preprocessing.feature_engineering import create_features

from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_preprocessing():

    logger.info("PHASE 3: DATA PREPROCESSING STARTED")

    df = pd.read_csv("/Datasets/raw/productivity_raw.csv")

    train, test = split_dataset(df)

    train = impute_missing_values(train)
    test = impute_missing_values(test)

    train = create_features(train)
    test = create_features(test)

    train = encode_features(train)
    test = encode_features(test)

    os.makedirs("/Datasets/processed", exist_ok=True)

    train.to_csv("/Datasets/processed/train.csv", index=False)
    test.to_csv("/Datasets/processed/test.csv", index=False)

    logger.info("Processed datasets saved")

    logger.info("PHASE 3 COMPLETED")