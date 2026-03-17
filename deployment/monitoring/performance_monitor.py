import pandas as pd
from sklearn.metrics import mean_squared_error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def monitor_model(predictions, true_values):

    rmse = mean_squared_error(
        true_values,
        predictions,
        squared=False
    )

    threshold = 0.25

    if rmse > threshold:

        print("Model performance degraded")
        return True

    return False