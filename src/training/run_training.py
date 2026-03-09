from src.training.train_random_forest import train_random_forest
from src.training.train_xgboost import train_xgboost
from src.training.evaluate_models import save_metrics

from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_training():

    logger.info("PHASE 4: MODEL TRAINING STARTED")

    rf_metrics = train_random_forest()

    xgb_metrics = train_xgboost()

    results = save_metrics(rf_metrics, xgb_metrics)

    logger.info("Model evaluation saved")

    logger.info("PHASE 4 COMPLETED")

    return results