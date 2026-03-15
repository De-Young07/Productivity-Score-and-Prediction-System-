import mlflow
from pathlib import Path

from src.utils.paths import PROJECT_ROOT


def setup_experiment():

    # store experiments locally inside project
    tracking_path = PROJECT_ROOT / "mlruns"

    mlflow.set_tracking_uri(f"file:{tracking_path}")

    mlflow.set_experiment("productivity_prediction_experiment")