import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_drift_monitor():

    reference = pd.read_csv(
        ROOT / "Datasets" / "processed" / "train.csv"
    )

    current = pd.read_csv(
        ROOT / "Datasets" / "incoming" / "new_data.csv"
    )

    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference,
        current_data=current
    )

    report.save_html(
        ROOT / "reports" / "figures" / "drift_report.html"
    )