"""
JSONL read/write utilities.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    """Write records to a JSONL file (overwrites)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a single record to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
