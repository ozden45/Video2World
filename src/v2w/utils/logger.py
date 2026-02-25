"""
logger.py


"""


import logging
import os
from datetime import datetime


def setup_logging(experiment_name: str, log_dir: str = "logs"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(log_dir, f"{timestamp}_{experiment_name}")
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "run.log")

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return run_dir

