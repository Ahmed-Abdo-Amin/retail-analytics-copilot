"""
Single query execution route.
Runs a single question through the agent and returns the output dict.
"""
from __future__ import annotations
from typing import Any, Dict

from agent.state import AgentState


def run_single_query(
    compiled_graph,
    question_id: str,
    question: str,
    format_hint: str,
) -> Dict[str, Any]:
    """
    Run one question through the compiled LangGraph agent.
    Returns a dict matching the output contract.
    """
    initial_state = AgentState(
        question_id=question_id,
        question=question,
        format_hint=format_hint,
    )

    final_state = compiled_graph.invoke(initial_state)

    # Handle both dict and AgentState returns
    if isinstance(final_state, dict):
        state = AgentState(**final_state)
    else:
        state = final_state

    sql_executed = ""
    if state.sql_result and state.sql_result.sql_executed:
        sql_executed = state.sql_result.sql_executed

    return {
        "id": state.question_id,
        "final_answer": state.final_answer,
        "sql": sql_executed,
        "confidence": state.confidence,
        "explanation": state.explanation,
        "citations": state.citations,
    }
