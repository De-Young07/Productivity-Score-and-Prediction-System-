import pandas as pd
import os

def save_metrics(rf_metrics, xgb_metrics):

    os.makedirs("reports", exist_ok=True)

    results = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost"],
        "RMSE": [rf_metrics[0], xgb_metrics[0]],
        "R2 Score": [rf_metrics[1], xgb_metrics[1]]
    })

    results.to_csv("reports/model_metrics.csv", index=False)

    return results