import pandas as pd
import joblib
import xgboost as xgb
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_xgboost():

    logger.info("Training XGBoost model")

    train = pd.read_csv("Datasets/processed/train.csv")
    test = pd.read_csv("Datasets/processed/test.csv")

    target = "actual_productivity_score"

    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    joblib.dump(model, "models/xgboost_model.pkl")

    logger.info("XGBoost training completed")

    return rmse, r2