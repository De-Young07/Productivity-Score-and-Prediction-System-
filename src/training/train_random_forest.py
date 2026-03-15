import mlflow
import pandas as pd
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.preprocessing.build_features import build_preprocessing_pipeline
from src.utils.paths import MODELS_DIR, TRAIN_DATA, TEST_DATA


def train_random_forest(params):

    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    
    target = "actual_productivity_score"
    
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    model = RandomForestRegressor(**params, random_state=42)

    preprocessor = build_preprocessing_pipeline(train_df, target)

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

        mlflow.sklearn.log_model(pipeline, "random_forest_model")

    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(pipeline, MODELS_DIR / "random_forest.pkl")

    return rmse, r2