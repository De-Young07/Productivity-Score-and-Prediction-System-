import logging
import os

def get_logger(name):

    log_dir = "logs"
    # os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{log_dir}/project.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger(name)