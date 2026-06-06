"""
Structured logging utilities.
"""
from __future__ import annotations
import logging
import sys
from typing import Any, Dict

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with rich handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_event(logger: logging.Logger, node: str, event: str, data: Dict[str, Any] = None) -> None:
    """Log a structured node event."""
    msg = f"[{node}] {event}"
    if data:
        msg += f" | {data}"
    logger.info(msg)
