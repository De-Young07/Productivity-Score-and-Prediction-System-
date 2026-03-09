import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

def feature_importance():

    data = pd.read_csv(ROOT / "data" / "processed" / "train_data.csv")

    X = data.drop(columns=["productivity_score"])

    model = joblib.load(ROOT / "models" / "xgboost_model.pkl")

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    importance_df.to_csv(
        ROOT / "reports" / "tables" / "feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10,6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.gca().invert_yaxis()

    plt.title("Feature Importance for Productivity Prediction")

    plt.savefig(ROOT / "reports" / "figures" / "feature_importance.png")

    return importance_df