import mlflow
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from src.preprocessing.build_features import build_preprocessing_pipeline
from src.utils.paths import TRAIN_DATA, TEST_DATA, MODELS_DIR


def train_xgboost(params):

    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    
    target = "actual_productivity_score"
    
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    preprocessor = build_preprocessing_pipeline(train_df, target)

    model = xgb.XGBRegressor(**params, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    with mlflow.start_run():

        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.xgboost.log_model(model, "xgboost_model")

    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(pipeline, MODELS_DIR / "xgboost.pkl")

    return rmse, r2