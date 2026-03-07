import logging
import sys
import os


def setup_logging(log_file=None):
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        root.handlers.clear()

    # console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # optional file logging
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        
    # silence noisy libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    