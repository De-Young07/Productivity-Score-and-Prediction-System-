import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def evaluate_models():

    test_path = ROOT / "data" / "processed" / "test_data.csv"

    df = pd.read_csv(test_path)

    X_test = df.drop(columns=["productivity_score"])
    y_test = df["productivity_score"]

    rf_model = joblib.load(ROOT / "models" / "random_forest_model.pkl")
    xgb_model = joblib.load(ROOT / "models" / "xgboost_model.pkl")

    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    results = []

    def compute_metrics(y_true, y_pred, model):

        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)

        results.append({
            "model": model,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

    compute_metrics(y_test, rf_pred, "RandomForest")
    compute_metrics(y_test, xgb_pred, "XGBoost")

    results_df = pd.DataFrame(results)

    results_df.to_csv(ROOT / "reports" / "tables" / "model_performance.csv", index=False)

    print(results_df)

    return results_df