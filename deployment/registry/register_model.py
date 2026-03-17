import pandas as pd
import mlflow
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def register_best_model():

    performance = pd.read_csv(
        ROOT / "reports" / "tables" / "model_performance.csv"
    )

    best_model = performance.sort_values("RMSE").iloc[0]["model"]

    if best_model == "RandomForest":
        model_path = ROOT / "models" / "random_forest_model.pkl"
    else:
        model_path = ROOT / "models" / "xgboost_model.pkl"

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    with mlflow.start_run():

        mlflow.log_artifact(str(model_path))

        mlflow.register_model(
            str(model_path),
            "productivity_prediction_model"
        )

    print("Model registered:", best_model)