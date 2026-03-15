import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from src.utils.paths import MODELS_DIR, TEST_DATA

def run_evaluation():

    test_df = pd.read_csv(TEST_DATA)
    
    target = "actual_productivity_score"

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    results = []

    for model_name in ["random_forest.pkl","xgboost.pkl"]:

        model = joblib.load(MODELS_DIR / model_name)

        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append({
            "model":model_name,
            "rmse":rmse,
            "r2":r2
        })

    df = pd.DataFrame(results)

    return df