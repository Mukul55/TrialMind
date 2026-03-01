"""
Structured logging configuration for TrialMind.

Uses loguru for structured, configurable logging with:
- Console output (development)
- File rotation (production)
- JSON formatting (log aggregation)
"""

import sys
import os
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = None,
    json_format: bool = False
) -> None:
    """
    Configure the loguru logger for TrialMind.

    Args:
        log_level: Minimum log level to capture (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to write logs to file
        json_format: If True, output logs as JSON (useful for log aggregation)
    """
    # Remove default handler
    logger.remove()

    if json_format:
        fmt = "{time} {level} {module} {message}"
    else:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Console handler
    logger.add(
        sys.stderr,
        format=fmt,
        level=log_level,
        colorize=not json_format
    )

    # File handler (if specified)
    if log_file:
        logger.add(
            log_file,
            format=fmt,
            level=log_level,
            rotation="100 MB",      # Rotate at 100MB
            retention="7 days",     # Keep 7 days of logs
            compression="zip",      # Compress rotated logs
            serialize=json_format   # JSON if requested
        )
        logger.info(f"Logging to file: {log_file}")

    logger.info(f"Logger initialized at level {log_level}")


def get_ingestion_logger(source: str):
    """
    Get a contextualized logger for a specific data source.
    Adds source context to all log messages.
    """
    return logger.bind(source=source)
