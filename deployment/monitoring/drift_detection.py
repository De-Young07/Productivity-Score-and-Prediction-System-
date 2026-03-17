import pandas as pd
from scipy.stats import ks_2samp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def detect_drift(new_data_path):

    train_data = pd.read_csv(
        ROOT / "Datasets" / "processed" / "train.csv"
    )

    new_data = pd.read_csv(new_data_path)

    drift_results = {}

    for column in train_data.columns:

        stat, p_value = ks_2samp(
            train_data[column],
            new_data[column]
        )

        drift_results[column] = p_value

    return drift_results