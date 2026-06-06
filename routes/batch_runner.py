"""
Batch runner: processes a JSONL evaluation file and writes outputs.
"""
from __future__ import annotations
from typing import Any, Dict, List

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from routes.single_query import run_single_query
from utils.jsonl_utils import read_jsonl, write_jsonl
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_batch(
    compiled_graph,
    input_path: str,
    output_path: str,
) -> List[Dict[str, Any]]:
    """
    Process all questions in a JSONL file through the agent.
    Writes outputs to output_path and returns the results list.
    """
    questions = read_jsonl(input_path)
    logger.info(f"Loaded {len(questions)} questions from {input_path}")

    results: List[Dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing questions...", total=len(questions))

        for q in questions:
            qid = q.get("id", "unknown")
            question = q.get("question", "")
            format_hint = q.get("format_hint", "str")

            progress.update(task, description=f"Processing: {qid}")
            logger.info(f"Processing question: {qid}")

            try:
                output = run_single_query(
                    compiled_graph=compiled_graph,
                    question_id=qid,
                    question=question,
                    format_hint=format_hint,
                )
            except Exception as exc:
                logger.error(f"Failed on {qid}: {exc}")
                output = {
                    "id": qid,
                    "final_answer": None,
                    "sql": "",
                    "confidence": 0.0,
                    "explanation": f"Agent error: {exc}",
                    "citations": [],
                }

            results.append(output)
            progress.advance(task)

    write_jsonl(output_path, results)
    logger.info(f"Wrote {len(results)} outputs to {output_path}")
    return results
