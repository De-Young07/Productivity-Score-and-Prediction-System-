import pandas as pd
import numpy as np
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

def detect_outliers(df):

    logger.info("Starting outlier detection")

    numeric_df = df.select_dtypes(include=[np.number])

    outlier_rows = []

    for col in numeric_df.columns:

        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        outlier_rows.append({
            "column": col,
            "outlier_count": len(outliers)
        })

    report = pd.DataFrame(outlier_rows)

    os.makedirs("reports", exist_ok=True)

    report_path = "reports/outlier_report.csv"

    report.to_csv(report_path, index=False)

    logger.info(f"Outlier report saved to {report_path}")

    return report