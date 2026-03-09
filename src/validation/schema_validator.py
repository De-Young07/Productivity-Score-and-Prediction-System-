import json
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def validate_schema(df, schema_path):

    with open(schema_path) as f:
        schema = json.load(f)

    num_cols = schema["numerical_columns"]
    cat_cols = schema["categorical_columns"]
    bool_cols = schema["boolean_columns"]

    missing_columns = []

    for col in num_cols + cat_cols + bool_cols:
        if col not in df.columns:
            missing_columns.append(col)

    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
        raise Exception("Dataset schema mismatch")

    logger.info("Schema validation passed")

    return True