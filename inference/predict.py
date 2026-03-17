import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "xgboost_model.pkl"

model = joblib.load(MODEL_PATH)

def predict(data):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return prediction[0]