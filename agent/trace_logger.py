"""
Replayable event trace logger.
Writes structured events to a JSONL trace file for full auditability.
"""
from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)

_TRACE_PATH = str(Path(__file__).parent.parent / "outputs" / "trace.jsonl")


class TraceLogger:
    """Logs node-level events to an append-only JSONL trace file."""

    def __init__(self, trace_path: str = _TRACE_PATH, question_id: str = ""):
        self.trace_path = trace_path
        self.question_id = question_id
        self._events: List[Dict[str, Any]] = []
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, node: str, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a single event."""
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "question_id": self.question_id,
            "node": node,
            "event": event,
            "data": data or {},
        }
        self._events.append(record)
        logger.info(f"[TRACE] [{node}] {event}")
        # Append to file immediately for fault tolerance
        with open(self.trace_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def get_events(self) -> List[Dict[str, Any]]:
        """Return all events logged in this session."""
        return list(self._events)
