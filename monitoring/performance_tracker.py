import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]

def log_performance(model, rmse, r2):

    log_file = ROOT / "reports" / "tables" / "performance_history.csv"

    new_row = pd.DataFrame({
        "date":[datetime.now()],
        "model":[model],
        "rmse":[rmse],
        "r2":[r2]
    })

    if log_file.exists():
        df = pd.read_csv(log_file)
        df = pd.concat([df,new_row])
    else:
        df = new_row

    df.to_csv(log_file,index=False)