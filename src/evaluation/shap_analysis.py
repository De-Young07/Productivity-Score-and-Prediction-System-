import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_shap_analysis():

    data = pd.read_csv(ROOT / "Datasets" / "processed" / "train.csv")

    X = data.drop(columns=["actual_productivity_score"])

    model = joblib.load(ROOT / "models" / "xgboost_model.pkl")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    plt.savefig(
        ROOT / "reports" / "figures" / "shap_summary.png"
    )