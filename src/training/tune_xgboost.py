import optuna
import pandas as pd
import xgboost as xgb
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.preprocessing.build_features import build_preprocessing_pipeline
from src.utils.paths import TRAIN_DATA, TEST_DATA


def objective(trial):

    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    
    target = "actual_productivity_score"
    
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0)
    }

    model = xgb.XGBRegressor(**params, random_state=42)

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

    return rmse


def tune_xgboost():

    study = optuna.create_study(direction="minimize")

    study.optimize(objective, n_trials=30)

    return study.best_params