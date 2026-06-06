"""
BaseController: shared utilities for all controllers.
"""
from __future__ import annotations
import time
from typing import Any, Dict

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseController:
    """Provides common execution utilities and timing for all controllers."""

    def __init__(self, name: str):
        self.name = name
        self._logger = get_logger(f"controller.{name}")

    def timed_run(self, fn, *args, **kwargs) -> Any:
        """Run a function and log its execution time."""
        t0 = time.time()
        result = fn(*args, **kwargs)
        elapsed = round(time.time() - t0, 3)
        self._logger.info(f"{self.name} completed in {elapsed}s")
        return result

    def log_info(self, msg: str, data: Dict[str, Any] = None) -> None:
        entry = f"[{self.name}] {msg}"
        if data:
            entry += f" | {data}"
        self._logger.info(entry)

    def log_error(self, msg: str, exc: Exception = None) -> None:
        entry = f"[{self.name}] ERROR: {msg}"
        if exc:
            entry += f" | {exc}"
        self._logger.error(entry)
