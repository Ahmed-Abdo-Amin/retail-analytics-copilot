"""
Output contract models enforced on final answers.
"""
from __future__ import annotations
from typing import Any, List
from pydantic import BaseModel, Field


class FinalOutput(BaseModel):
    """Strict output contract per question."""
    id: str
    final_answer: Any
    sql: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    citations: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
