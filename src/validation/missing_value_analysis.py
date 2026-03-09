import pandas as pd
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

def analyze_missing_values(df):

    logger.info("Starting missing value analysis")

    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100

    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing,
        "missing_percent": percent
    })

    os.makedirs("reports", exist_ok=True)

    report_path = "reports/missing_values_report.csv"

    report.to_csv(report_path, index=False)

    logger.info(f"Missing value report saved to {report_path}")

    return report