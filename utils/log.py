"""Logging configuration."""

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger instance.

    The log level is determined by the `LOG_LEVEL` environment variable.
    If not set, it defaults to `INFO`.
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Configure formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure stream handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)

    # Get logger and add handler
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = False  # Prevents duplicating logs to the root logger

    return logger


log = get_logger("agentic_diagnosis")
