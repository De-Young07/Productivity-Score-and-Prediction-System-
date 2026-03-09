import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def impute_missing_values(df):

    logger.info("Starting missing value imputation")

    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    logger.info("Imputation completed")

    return df