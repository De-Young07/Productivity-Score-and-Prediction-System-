import pandas as pd

from src.validation.schema_validator import validate_schema
from src.validation.missing_value_analysis import analyze_missing_values
from src.validation.outlier_detection import detect_outliers

from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_validation():

    logger.info("PHASE 2 STARTED")

    df = pd.read_csv("Datasets/raw/productivity_raw.csv")

    logger.info(f"Dataset loaded for validation: {df.shape}")

    validate_schema(df, "configs/data_schema.json")

    logger.info("Schema validation completed")

    analyze_missing_values(df)

    logger.info("Missing value analysis completed")

    detect_outliers(df)

    logger.info("Outlier detection completed")

    logger.info("PHASE 2 FINISHED")