import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def encode_features(df):

    logger.info("Encoding categorical variables")

    categorical_cols = df.select_dtypes(include=["object"]).columns

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    logger.info("Encoding completed")

    return df