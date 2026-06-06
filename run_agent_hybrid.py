"""
run_agent_hybrid.py — CLI entry point for Retail Analytics Copilot.

Usage:
    python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
    python run_agent_hybrid.py --question "What is the return window for beverages?" --format_hint int
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

console = Console()


def _build_agent():
    """Build and return the compiled LangGraph agent."""
    from main import initialize
    initialize()

    # Import after DSPy is configured
    from rag.document_loader import load_documents
    from rag.chunker import chunk_documents
    from rag.retriever import HybridRetriever
    from agent.dspy_modules import RouterModule, NL2SQLModule, SynthesisModule, PlannerModule
    from agent.optimizer import run_optimization
    from agent.graph_hybrid import build_graph, init_graph_components

    # Load and index documents
    console.print("[bold cyan]Loading and indexing documents...[/bold cyan]")
    docs = load_documents()
    chunks = chunk_documents(docs)
    retriever = HybridRetriever(chunks)
    console.print(f"  → {len(chunks)} chunks indexed from {len(docs)} documents")

    # Build DSPy modules
    router = RouterModule()
    nl2sql = NL2SQLModule()
    synthesis = SynthesisModule()
    planner = PlannerModule()

    # Run DSPy optimization on NL2SQL
    console.print("[bold cyan]Running DSPy NL→SQL optimization...[/bold cyan]")
    try:
        optimized_nl2sql, metrics = run_optimization(nl2sql)
    except Exception as exc:
        console.print(f"[yellow]Optimization skipped: {exc}[/yellow]")
        optimized_nl2sql = nl2sql
        metrics = {"before_sql_success_rate": 0.0, "after_sql_success_rate": 0.0, "delta": 0.0}

    # Display metrics table
    _display_metrics(metrics)

    # Inject modules into graph
    init_graph_components(
        retriever=retriever,
        router=router,
        nl2sql=optimized_nl2sql,
        synthesis=synthesis,
        planner=planner,
    )

    # Build and compile graph
    console.print("[bold cyan]Compiling LangGraph agent...[/bold cyan]")
    graph = build_graph()
    console.print("[bold green]✓ Agent ready[/bold green]")
    return graph


def _display_metrics(metrics: dict) -> None:
    """Display DSPy optimization metrics table."""
    table = Table(title="DSPy NL→SQL Optimization Results", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Delta", justify="right")

    before = f"{metrics.get('before_sql_success_rate', 0):.0%}"
    after = f"{metrics.get('after_sql_success_rate', 0):.0%}"
    delta = f"{metrics.get('delta', 0):+.0%}"

    table.add_row("SQL Execution Success Rate", before, after, delta)
    console.print(table)


def _display_results(results: list) -> None:
    """Display batch results summary."""
    table = Table(title="Batch Results Summary", show_header=True)
    table.add_column("ID", style="cyan", max_width=40)
    table.add_column("Answer", max_width=40)
    table.add_column("Confidence", justify="right")
    table.add_column("Citations", justify="right")

    for r in results:
        answer_str = str(r.get("final_answer", ""))[:40]
        conf = f"{r.get('confidence', 0):.2f}"
        cit = str(len(r.get("citations", [])))
        table.add_row(r.get("id", ""), answer_str, conf, cit)

    console.print(table)


@click.command()
@click.option("--batch", default=None, help="Path to JSONL evaluation file")
@click.option("--out", default="outputs/outputs_hybrid.jsonl", help="Output JSONL path")
@click.option("--question", default=None, help="Single question (alternative to --batch)")
@click.option("--format_hint", default="str", help="Format hint for single question")
@click.option("--no-optimize", is_flag=True, default=False, help="Skip DSPy optimization")
def main(batch, out, question, format_hint, no_optimize):
    """Retail Analytics Copilot — DSPy + LangGraph hybrid agent."""
    console.rule("[bold red]Retail Analytics Copilot[/bold red]")

    graph = _build_agent()

    if batch:
        # Batch mode
        console.print(f"\n[bold]Running batch: {batch} → {out}[/bold]")
        from routes.batch_runner import run_batch
        results = run_batch(compiled_graph=graph, input_path=batch, output_path=out)
        _display_results(results)
        console.print(f"\n[bold green]✓ Outputs written to {out}[/bold green]")

    elif question:
        # Single question mode
        from routes.single_query import run_single_query
        console.print(f"\n[bold]Question:[/bold] {question}")
        result = run_single_query(
            compiled_graph=graph,
            question_id="single",
            question=question,
            format_hint=format_hint,
        )
        console.print_json(json.dumps(result, indent=2))

    else:
        console.print("[red]Error: provide --batch or --question[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
