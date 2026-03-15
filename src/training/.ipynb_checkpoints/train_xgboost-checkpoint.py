import mlflow
import mlflow.xgboost
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.metrics import mean_squared_error, r2_score

from src.utils.paths import RAW_DATA, MODELS_DIR


def train_xgboost(params):

    data = pd.read_csv(RAW_DATA)

    target = "actual_productivity_score"

    X = data.drop(columns=[target])
    y = data[target]

    model = xgb.XGBRegressor(**params, random_state=42)

    model.fit(X, y)

    preds = model.predict(X)

    rmse = mean_squared_error(y, preds, squared=False)
    r2 = r2_score(y, preds)

    with mlflow.start_run():

        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        mlflow.xgboost.log_model(model, "xgboost_model")

    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODELS_DIR / "xgboost.pkl")

    return rmse, r2