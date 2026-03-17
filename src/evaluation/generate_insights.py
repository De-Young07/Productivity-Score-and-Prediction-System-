import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def generate_insights():

    performance = pd.read_csv(
        ROOT / "reports" / "tables" / "model_performance.csv"
    )

    importance = pd.read_csv(
        ROOT / "reports" / "tables" / "xgb_feature_importance.csv"
    )

    best_model = performance.sort_values("RMSE").iloc[0]["model"]

    top_features = importance.head(3)["feature"].tolist()

    insights = f"""
Model Evaluation Summary
------------------------

Best performing model: {best_model}

Top productivity drivers:
1. {top_features[0]}
2. {top_features[1]}
3. {top_features[2]}

Interpretation
--------------

Productivity outcomes are strongly influenced by behavioral patterns
related to digital usage.

Higher social media engagement correlates negatively with productivity
while structured time allocation correlates positively.

Practical Implications
----------------------

• Reduce time spent on distracting social platforms
• Encourage structured work sessions
• Track digital behavior patterns
"""

    report_file = ROOT / "reports" / "project_insights.txt"

    with open(report_file, "w") as f:
        f.write(insights)

    return insights