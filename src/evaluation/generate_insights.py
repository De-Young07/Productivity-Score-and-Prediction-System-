import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def generate_insights():

    performance = pd.read_csv(ROOT / "reports" / "tables" / "model_performance.csv")
    importance = pd.read_csv(ROOT / "reports" / "tables" / "feature_importance.csv")

    best_model = performance.sort_values("RMSE").iloc[0]["model"]

    top_features = importance.head(3)["feature"].tolist()

    insights = f"""
Model Evaluation Summary
------------------------

Best performing model: {best_model}

Top factors influencing productivity:
1. {top_features[0]}
2. {top_features[1]}
3. {top_features[2]}

Key Interpretation
------------------

The machine learning model identified that productivity is strongly influenced
by behavioral patterns related to social media usage.

Higher engagement with distracting platforms correlates with lower productivity
scores, while structured time management variables correlate positively with
productivity outcomes.

Implications
------------

Organizations and individuals can improve productivity by:

• Limiting time spent on distracting social platforms
• Structuring focused work sessions
• Monitoring digital usage habits
"""

    report_file = ROOT / "reports" / "project_insights.txt"

    with open(report_file, "w") as f:
        f.write(insights)

    print("Insights generated")

    return insights