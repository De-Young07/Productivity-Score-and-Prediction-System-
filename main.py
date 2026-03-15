import sys
import os
from pathlib import Path
from src.utils.logger import get_logger
import traceback

# Resolve project root
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

logger = get_logger(__name__)


from src.ingestion.load_data import run_ingestion
from src.validation.run_validation import run_validation
from src.preprocessing.run_preprocessing import run_preprocessing
from src.training.run_training import run_training
from src.evaluation.run_evaluation import run_evaluation


def main():

    try:

        logger.info("===== PIPELINE STARTED =====")

        logger.info("Starting Phase 1: Data Ingestion")
        run_ingestion()

        logger.info("Starting Phase 2: Data Validation")
        run_validation()

        logger.info("Starting Phase 3: Data Preprocessing")
        run_preprocessing()

        logger.info("Starting Phase 4: Model Training")
        run_training()

        logger.info("Starting Phase 5: Evaluation & Insights")
        results = run_evaluation()

        print(results)
    
        print("Phase 5b: Explainability")
        run_explainability()

        logger.info("===== PIPELINE COMPLETED SUCCESSFULLY =====")

    except Exception as e:

        logger.error("Pipeline failed")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()