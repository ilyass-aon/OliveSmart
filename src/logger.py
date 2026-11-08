"""
OliveSmart — Centralized Logging Module
=========================================
Provides a consistent, colored, timestamped logger
used throughout the entire framework.

Design decisions:
- Uses Python's standard `logging` library (no external dependency for core logging)
- Uses `colorlog` for colored terminal output (optional, falls back gracefully)
- Each run creates a new timestamped log file
- Log format includes dataset/model context when set via LogContext
"""

import io
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Try to import colorlog for colored output; fall back to plain logging if unavailable
try:
    import colorlog
    _COLORLOG_AVAILABLE = True
except ImportError:
    _COLORLOG_AVAILABLE = False


def setup_logger(
    name: str = "olivesmart",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    console: bool = True,
    file_output: bool = True,
    prefix: str = "olivesmart",
) -> logging.Logger:
    """
    Configure and return the main application logger.

    Parameters
    ----------
    name : str
        Logger name (used in log records).
    level : str
        Log level string: "DEBUG", "INFO", "WARNING", "ERROR".
    log_dir : Path, optional
        Directory where log files will be saved.
    console : bool
        Whether to output logs to the terminal.
    file_output : bool
        Whether to write logs to a file.
    prefix : str
        Prefix for the log filename.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # ----------------------------------------------------------------
    # Console handler with optional color support
    # ----------------------------------------------------------------
    if console:
        if _COLORLOG_AVAILABLE:
            console_formatter = colorlog.ColoredFormatter(
                fmt=(
                    "%(log_color)s%(asctime)s [%(levelname)8s]%(reset)s "
                    "%(message)s"
                ),
                datefmt="%H:%M:%S",
                log_colors={
                    "DEBUG":    "cyan",
                    "INFO":     "green",
                    "WARNING":  "yellow",
                    "ERROR":    "red",
                    "CRITICAL": "bold_red",
                },
            )
        else:
            console_formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)8s] %(message)s",
                datefmt="%H:%M:%S",
            )

        # Force UTF-8 on the console stream to avoid UnicodeEncodeError
        # on Windows terminals that default to cp1252.
        try:
            utf8_stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
            console_stream = utf8_stdout
        except AttributeError:
            # sys.stdout.buffer may not exist in some environments (e.g., IDLE)
            console_stream = sys.stdout

        console_handler = logging.StreamHandler(console_stream)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # ----------------------------------------------------------------
    # File handler (plain text, no colors)
    # ----------------------------------------------------------------
    if file_output and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{prefix}_{timestamp}.log"

        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)8s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Log the file location so the user knows where to find it
        logger.debug(f"Logs enregistrés dans : {log_file}")

    return logger


def get_logger(name: str = "olivesmart") -> logging.Logger:
    """
    Retrieve an existing logger by name.
    Call setup_logger() first; this is a lightweight getter for modules.

    Parameters
    ----------
    name : str
        Logger name (must match the one used in setup_logger).

    Returns
    -------
    logging.Logger
        The existing logger, or a new unconfigured one if not found.
    """
    return logging.getLogger(name)
