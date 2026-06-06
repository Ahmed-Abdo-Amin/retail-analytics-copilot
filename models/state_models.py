"""
State models for LangGraph agent.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single retrieved document chunk."""
    chunk_id: str
    source: str
    content: str
    score: float


class SQLResult(BaseModel):
    """Result from SQL execution."""
    columns: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    error: Optional[str] = None
    sql_executed: str = ""


class AgentState(BaseModel):
    """Full LangGraph state passed between nodes."""
    # Input
    question_id: str = ""
    question: str = ""
    format_hint: str = ""

    # Routing
    route: str = ""  # rag | sql | hybrid

    # Retrieval
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    retrieval_score: float = 0.0

    # Planning
    date_range: Optional[Dict[str, str]] = None
    kpi_formula: str = ""
    categories: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)

    # SQL generation & execution
    generated_sql: str = ""
    sql_result: Optional[SQLResult] = None

    # Synthesis
    final_answer: Any = None
    explanation: str = ""
    citations: List[str] = Field(default_factory=list)
    confidence: float = 0.0

    # Validation & repair
    is_valid: bool = False
    validation_errors: List[str] = Field(default_factory=list)
    repair_count: int = 0
    max_repairs: int = 2

    # Trace
    trace: List[Dict[str, Any]] = Field(default_factory=list)
