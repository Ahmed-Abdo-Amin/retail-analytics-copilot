"""
Format hint validation utilities.
"""
from __future__ import annotations
import re
from typing import Any, List, Tuple


def validate_format_hint(answer: Any, format_hint: str) -> Tuple[bool, List[str]]:
    """
    Validate that `answer` matches `format_hint`.
    Returns (is_valid, list_of_errors).
    """
    errors: List[str] = []
    hint = format_hint.strip()

    if hint == "int":
        if not isinstance(answer, int):
            # Allow float that is whole number
            if isinstance(answer, float) and answer.is_integer():
                return True, []
            errors.append(f"Expected int, got {type(answer).__name__}: {answer}")

    elif hint == "float":
        if not isinstance(answer, (int, float)):
            errors.append(f"Expected float, got {type(answer).__name__}: {answer}")

    elif hint.startswith("{") and hint.endswith("}"):
        # Object hint like {category:str, quantity:int}
        if not isinstance(answer, dict):
            errors.append(f"Expected dict, got {type(answer).__name__}")
        else:
            keys = _parse_object_hint(hint)
            for key, typ in keys.items():
                if key not in answer:
                    errors.append(f"Missing key '{key}' in answer dict")
                else:
                    _check_type(answer[key], typ, key, errors)

    elif hint.startswith("list["):
        if not isinstance(answer, list):
            errors.append(f"Expected list, got {type(answer).__name__}")
        else:
            inner = hint[5:-1].strip()  # e.g. {product:str, revenue:float}
            if inner.startswith("{"):
                keys = _parse_object_hint(inner)
                for i, item in enumerate(answer):
                    if not isinstance(item, dict):
                        errors.append(f"Item {i} is not a dict")
                    else:
                        for key, typ in keys.items():
                            if key not in item:
                                errors.append(f"Item {i} missing key '{key}'")
                            else:
                                _check_type(item[key], typ, f"item[{i}].{key}", errors)

    return (len(errors) == 0), errors


def _parse_object_hint(hint: str) -> dict:
    """Parse {key:type, key:type} into a dict."""
    inner = hint.strip("{} ")
    result = {}
    for part in inner.split(","):
        part = part.strip()
        if ":" in part:
            k, t = part.split(":", 1)
            result[k.strip()] = t.strip()
    return result


def _check_type(value: Any, type_hint: str, label: str, errors: List[str]) -> None:
    """Check value matches type_hint string."""
    if type_hint == "str" and not isinstance(value, str):
        errors.append(f"'{label}' expected str, got {type(value).__name__}")
    elif type_hint == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            if isinstance(value, float) and value.is_integer():
                pass  # acceptable
            else:
                errors.append(f"'{label}' expected int, got {type(value).__name__}")
    elif type_hint == "float":
        if not isinstance(value, (int, float)):
            errors.append(f"'{label}' expected float, got {type(value).__name__}")
