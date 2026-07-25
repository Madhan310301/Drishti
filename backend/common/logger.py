"""
Drishti Logger Configuration

Provides a centralized logger for the entire project.
Logs are written to both the console and a log file.
"""

from pathlib import Path
import logging


# ---------------------------------------------------
# Project Paths
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "drishti.log"


# ---------------------------------------------------
# Logger Factory
# ---------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.

    Parameters
    ----------
    name : str
        Usually __name__

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger