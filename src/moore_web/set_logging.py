import sys
from pathlib import Path
from loguru import logger


def setup_logging(xp_folder: Path):
    """loguru setup with different levels for console vs file."""

    logger.remove()

    logger.add(
        sys.stdout,
        format="<green>{time:MM-DD HH:mm:ss}</green> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{level}</level> | "
        "<level>{message}</level>",
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        xp_folder / "parser.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {process} | {thread} | "
        "{name}:{function}:{line} | {level} | {message}",
        level="DEBUG",
        rotation="50 MB",
        retention="1 month",
        compression="zip",
        backtrace=True,
        diagnose=True,
    )

    return logger
