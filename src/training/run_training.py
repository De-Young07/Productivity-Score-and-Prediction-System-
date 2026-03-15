from src.training.experiment_tracking import setup_experiment
from src.training.tune_random_forest import tune_random_forest
from src.training.tune_xgboost import tune_xgboost
from src.training.train_random_forest import train_random_forest
from src.training.train_xgboost import train_xgboost
from src.training.compare_models import compare_models


def run_training():

    setup_experiment()

    rf_params = tune_random_forest()
    xgb_params = tune_xgboost()

    rf_rmse, rf_r2 = train_random_forest(rf_params)
    xgb_rmse, xgb_r2 = train_xgboost(xgb_params)

    results = [
        {"model": "RandomForest", "rmse": rf_rmse, "r2": rf_r2},
        {"model": "XGBoost", "rmse": xgb_rmse, "r2": xgb_r2}
    ]

    comparison_table, best_model = compare_models(results)

    print(comparison_table)
    print("Best model:", best_model)

    return best_model