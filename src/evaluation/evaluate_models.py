import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]


def evaluate_models():

    test_path = ROOT / "Datasets" / "processed" / "test.csv"
    df = pd.read_csv(test_path)

    X_test = df.drop(columns=["actual_productivity_score"])
    y_test = df["actual_productivity_score"]

    rf_model = joblib.load(ROOT / "models" / "random_forest_model.pkl")
    xgb_model = joblib.load(ROOT / "models" / "xgboost_model.pkl")

    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    results = []

    def compute_metrics(y_true, y_pred, model_name):

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        results.append({
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

    compute_metrics(y_test, rf_pred, "RandomForest")
    compute_metrics(y_test, xgb_pred, "XGBoost")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        ROOT / "reports" / "tables" / "model_performance.csv",
        index=False
    )

    return results_df