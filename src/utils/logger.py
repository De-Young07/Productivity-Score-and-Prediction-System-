import logging
from pathlib import Path
from src.utils.paths import LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name):

    log_file = LOG_DIR / "project.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:

        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger