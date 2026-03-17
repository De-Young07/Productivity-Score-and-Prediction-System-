import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def compute_feature_importance(model_file, model_name):

    train_data = pd.read_csv(ROOT / "Datasets" / "processed" / "train.csv")

    X = train_data.drop(columns=["actual_productivity_score"])

    model = joblib.load(ROOT / "models" / model_file)

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    importance_df.to_csv(
        ROOT / "reports" / "tables" / f"{model_name}_feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10,6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.gca().invert_yaxis()

    plt.title(f"{model_name} Feature Importance")

    plt.savefig(
        ROOT / "reports" / "figures" / f"{model_name}_feature_importance.png"
    )

    return importance_df


def run_feature_importance():

    compute_feature_importance(
        "random_forest_model.pkl",
        "rf"
    )

    compute_feature_importance(
        "xgboost_model.pkl",
        "xgb"
    )