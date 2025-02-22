import sys
import logging
import colorlog


def setup_logger(name: str, level: int = logging.INFO, logfile: str = None) -> logging.Logger:
    """
    Set up a logger with colored console output and an optional file handler.

    Args:
        name (str): Name of the logger.
        level (int, optional): Logging level (default: logging.INFO).
        logfile (str, optional): Path to a log file. If provided, log messages will also be written to this file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False # Stops double printing
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s%(levelname)-8s %(blue)s%(message)s",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "red,bg_white",
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Optionally add file handler
    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger