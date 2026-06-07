"""
api_server.py — FastAPI REST server for the Retail Analytics Copilot.

Endpoints:
  GET  /api/health       — health check + init status
  POST /api/query        — single question → typed answer
  POST /api/batch        — run full eval batch
  GET  /api/questions    — pre-loaded sample questions
  GET  /api/schema       — live DB schema (PRAGMA)
  GET  /api/docs         — all RAG document chunks
  GET  /api/outputs      — last batch run results
  GET  /api/trace        — latest trace events (last 200)
  POST /api/sql          — execute raw SELECT query
  GET  /api/optimize     — run DSPy optimization → before/after metrics
  GET  /                 — serve frontend/index.html
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

# ─── App ─────────────────────────────────────────────────────────────────────

# app created below with lifespan (see end of file)

_PROJECT = Path(__file__).parent
_GRAPH = None
_INIT_LOCK = threading.Lock()
_INITIALIZED = False
_WARMING_UP = False       # True while warmup query is running
_WARMUP_DONE = False      # True after warmup completes (success or skip)
_INIT_ERROR: Optional[str] = None


# ─── Pydantic request models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    format_hint: str = "str"
    id: str = "custom"


class SQLRequest(BaseModel):
    sql: str


# ─── Lazy initialization ─────────────────────────────────────────────────────

def _ensure_initialized():
    global _GRAPH, _INITIALIZED, _INIT_ERROR
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        try:
            from main import initialize
            initialize()

            from rag.document_loader import load_documents
            from rag.chunker import chunk_documents
            from rag.retriever import HybridRetriever
            from agent.dspy_modules import (
                RouterModule, NL2SQLModule, SynthesisModule, PlannerModule
            )
            from agent.graph_hybrid import build_graph, init_graph_components

            docs = load_documents()
            chunks = chunk_documents(docs)
            retriever = HybridRetriever(chunks)
            init_graph_components(
                retriever=retriever,
                router=RouterModule(),
                nl2sql=NL2SQLModule(),
                synthesis=SynthesisModule(),
                planner=PlannerModule(),
            )
            _GRAPH = build_graph()
            _INITIALIZED = True
            _INIT_ERROR = None
        except Exception as exc:
            _INIT_ERROR = str(exc)
            raise RuntimeError(_INIT_ERROR)


def _require_agent():
    """Ensure agent is initialized AND warmup is done before serving requests."""
    try:
        _ensure_initialized()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Agent not ready: {exc}")

    # Block until warmup finishes — prevents race where user query races warmup
    if _WARMING_UP:
        import time
        deadline = time.time() + 60          # max 60s wait
        while _WARMING_UP and time.time() < deadline:
            time.sleep(0.2)
        if _WARMING_UP:                       # still running after timeout
            raise HTTPException(status_code=503, detail="Agent warmup timed out")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler — runs init + warmup synchronously in a thread,
    then yields (which triggers uvicorn's 'Application startup complete').
    This guarantees the log order:
        ✓ Agent initialized
        🔥 Warming up...
        ✅ Warmup done — answer: "..."
        INFO: Application startup complete.   ← always last
    """
    import threading

    _done_event = threading.Event()

    def _init_and_warmup():
        global _WARMING_UP, _WARMUP_DONE

        # ── Step 1: initialise DSPy / LangGraph / RAG ────────────────────
        try:
            _ensure_initialized()
            print("\u2713 Agent initialized and ready")
        except Exception as exc:
            print(f"\u26a0 Agent init failed: {exc}")
            _WARMUP_DONE = True
            _done_event.set()
            return

        # ── Step 2: warmup query ──────────────────────────────────────────
        _WARMING_UP = True
        try:
            from agent.state import AgentState
            from utils.logging_utils import get_logger as _get_logger
            _wlog = _get_logger("warmup")
            warmup_q = "What is the return policy for beverages?"
            _wlog.info(f"\U0001f525 Warming up \u2014 question: \"{warmup_q}\"")
            warmup_state = AgentState(
                question_id="__warmup__",
                question=warmup_q,
                format_hint="str",
            )
            result = _GRAPH.invoke(warmup_state)
            answer = result.get("final_answer", "(no answer)")
            _wlog.info(f"\u2705 Warmup done \u2014 answer: \"{answer}\"")
        except Exception as exc:
            print(f"\u26a0 Warmup skipped: {exc}")
        finally:
            _WARMING_UP = False
            _WARMUP_DONE = True
            _done_event.set()   # unblock the lifespan yield

    t = threading.Thread(target=_init_and_warmup, daemon=True)
    t.start()
    _done_event.wait()   # block here until warmup finishes
    yield                # NOW uvicorn prints "Application startup complete."
    # (no teardown needed)




# Re-create app with lifespan attached
app = FastAPI(
    title="Retail Analytics Copilot",
    description="DSPy + LangGraph hybrid AI agent over Northwind SQLite",
    version="1.0.0",
    docs_url="/api/openapi",
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Re-mount static files after app recreation
_FRONTEND = _PROJECT / "frontend"
_FRONTEND.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    """Health check — returns initialization and warmup status."""
    if _WARMING_UP:
        status = "warming_up"
    elif _INITIALIZED:
        status = "ok"
    elif _INIT_ERROR:
        status = "error"
    else:
        status = "initializing"

    return {
        "status": status,
        "initialized": _INITIALIZED,
        "warming_up": _WARMING_UP,
        "warmup_done": _WARMUP_DONE,
        "mode": "local" if os.getenv("LOCAL_MODELS", "True").strip().lower() not in ("false", "0", "no") else "remote_ngrok",
        "error": _INIT_ERROR,
    }


@app.post("/api/query")
def query(body: QueryRequest):
    """Run a single question through the hybrid agent."""
    _require_agent()

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="'question' is required")

    try:
        from agent.state import AgentState
        state = AgentState(
            question_id=body.id,
            question=body.question.strip(),
            format_hint=body.format_hint.strip(),
        )
        result = _GRAPH.invoke(state)
        return {
            "id": result.get("question_id", body.id),
            "final_answer": result.get("final_answer"),
            "sql": result.get("generated_sql", ""),
            "confidence": result.get("confidence", 0.0),
            "explanation": result.get("explanation", ""),
            "citations": result.get("citations", []),
            "route": result.get("route", ""),
            "repair_count": result.get("repair_count", 0),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/query/stream")
async def query_stream(body: QueryRequest, request: Request):
    """
    SSE Streaming endpoint.
    Events emitted:
      {type:"progress", node, icon, message, ...extra}   — pipeline node progress
      {type:"token",    text}                             — live token from Ollama (synthesis/question phases)
      {type:"result",   ...full answer payload}           — final structured result
      {type:"done"}
    """
    _require_agent()
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="'question' is required")

    import asyncio, queue, threading, os, requests as _req

    progress_q: queue.Queue = queue.Queue()

    OLLAMA_BASE = os.getenv("_NGROK_SYNTHESIS_URL", "").rstrip("/") or "http://127.0.0.1:11434"
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3.5:3.8b-mini-instruct-q4_K_M")
    LOCAL = os.getenv("LOCAL_MODELS", "True").strip().lower() not in ("false", "0", "no")

    NODE_MESSAGES = {
        "router":       ("\U0001f500", "Classifying question…"),
        "retriever":    ("\U0001f4c4", "Retrieving relevant documents…"),
        "planner":      ("\U0001f9e0", "Extracting constraints & date ranges…"),
        "nl2sql":       ("\U0001f5c4\ufe0f",  "Generating SQL query…"),
        "sql_executor": ("\u26a1", "Executing SQL on database…"),
        "synthesizer":  ("\U0001f4ac", "Synthesizing final answer…"),
        "validator":    ("\u2705", "Validating answer format…"),
        "repair":       ("\U0001f527", "Repairing answer…"),
        "checkpoint":   ("\U0001f4ca", "Computing confidence score…"),
    }

    def _stream_tokens_from_ollama(prompt: str, phase: str):
        """
        Call Ollama /api/chat with stream=true and push each token chunk
        to the progress queue as {type:"token", text, phase}.
        Falls back silently if Ollama is unreachable.
        """
        try:
            if LOCAL:
                url = "http://127.0.0.1:11434/api/chat"
            else:
                url = f"{OLLAMA_BASE}/api/chat"

            resp = _req.post(
                url,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "options": {"temperature": 0, "num_predict": 300},
                },
                stream=True,
                timeout=60,
            )
            if not resp.ok:
                return
            import json as _json
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = _json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        progress_q.put({"type": "token", "text": token, "phase": phase})
                    if chunk.get("done"):
                        break
                except Exception:
                    continue
        except Exception:
            pass  # non-fatal — pipeline continues normally

    def run_agent():
        try:
            from agent.state import AgentState

            state = AgentState(
                question_id=body.id,
                question=body.question.strip(),
                format_hint=body.format_hint.strip(),
            )

            # ── Phase 1: Stream "thinking about question" tokens ──────────
            think_prompt = (
                f"You are a retail analytics assistant. A user asked:\n\"{body.question.strip()}\"\n"
                f"In 1-2 sentences, briefly describe what data you will look for to answer this."
            )
            _stream_tokens_from_ollama(think_prompt, "thinking")
            progress_q.put({"type": "token", "text": "\n", "phase": "thinking"})

            # ── Phase 2: Run full LangGraph pipeline (node by node) ───────
            for step in _GRAPH.stream(state):
                for node_name in step:
                    icon, msg = NODE_MESSAGES.get(node_name, ("\u2699\ufe0f", f"Running {node_name}…"))
                    node_state = step[node_name] or {}
                    extra = {}
                    if node_name == "router":
                        extra["route"] = node_state.get("route", "")
                    elif node_name == "nl2sql":
                        sql = node_state.get("generated_sql", "")
                        if sql:
                            extra["sql_preview"] = sql[:120] + ("…" if len(sql) > 120 else "")
                    elif node_name == "sql_executor":
                        res = node_state.get("sql_result")
                        if res:
                            extra["row_count"] = getattr(res, "row_count", 0) if hasattr(res, "row_count") else (res.get("row_count", 0) if isinstance(res, dict) else 0)
                    progress_q.put({"type": "progress", "node": node_name, "icon": icon, "message": msg, **extra})

            # ── Phase 3: Stream explanation tokens ────────────────────────
            final = list(step.values())[-1] if step else {}
            explanation = final.get("explanation", "")
            answer = final.get("final_answer")
            import json as _json

            if answer is not None:
                ans_str = _json.dumps(answer) if not isinstance(answer, str) else answer
                explain_prompt = (
                    f"Retail analytics question: \"{body.question.strip()}\"\n"
                    f"Answer found: {ans_str[:200]}\n"
                    f"In 2-3 concise sentences, explain this result to a business user. Be direct."
                )
                progress_q.put({"type": "token", "text": "", "phase": "answer_intro"})
                _stream_tokens_from_ollama(explain_prompt, "answer_intro")

            # ── Final result ──────────────────────────────────────────────
            result = {
                "type": "result",
                "id": final.get("question_id", body.id),
                "final_answer": answer,
                "sql": final.get("generated_sql", ""),
                "confidence": final.get("confidence", 0.0),
                "explanation": explanation,
                "citations": final.get("citations", []),
                "route": final.get("route", ""),
                "repair_count": final.get("repair_count", 0),
            }
            progress_q.put(result)
            progress_q.put({"type": "done"})

        except Exception as exc:
            progress_q.put({"type": "error", "message": str(exc)})
            progress_q.put({"type": "done"})

    async def event_generator():
        loop = asyncio.get_event_loop()
        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()
        while True:
            if await request.is_disconnected():
                break
            try:
                msg = await loop.run_in_executor(None, lambda: progress_q.get(timeout=0.1))
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") == "done":
                    break
            except queue.Empty:
                yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/batch")
def batch():
    """Run the full evaluation batch (6 sample questions)."""
    _require_agent()
    questions_path = _PROJECT / "sample_questions_hybrid_eval.jsonl"
    output_path = _PROJECT / "outputs" / "outputs_hybrid.jsonl"
    try:
        from routes.batch_runner import run_batch
        results = run_batch(
            compiled_graph=_GRAPH,
            input_path=str(questions_path),
            output_path=str(output_path),
        )
        return {"results": results, "count": len(results)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/questions")
def questions():
    """Return the 6 pre-loaded sample evaluation questions."""
    path = _PROJECT / "sample_questions_hybrid_eval.jsonl"
    return {"questions": _read_jsonl(path)}


@app.get("/api/schema")
def schema():
    """Return live Northwind database schema via PRAGMA introspection."""
    _require_agent()
    try:
        from tools.sqlite_tool import get_table_names, get_table_columns, get_foreign_keys
        tables = get_table_names()
        result = {}
        for t in tables:
            result[t] = {
                "columns": get_table_columns(t),
                "foreign_keys": get_foreign_keys(t),
            }
        return {"tables": result, "table_count": len(tables)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/docs")
def docs_chunks():
    """Return all loaded RAG document chunks with chunk IDs."""
    _require_agent()
    try:
        from rag.document_loader import load_documents
        from rag.chunker import chunk_documents
        docs = load_documents()
        chunks = chunk_documents(docs)
        return {
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "content": c.content,
                    "index": c.index,
                }
                for c in chunks
            ],
            "total": len(chunks),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/outputs")
def outputs():
    """Return last batch run output results."""
    path = _PROJECT / "outputs" / "outputs_hybrid.jsonl"
    return {"outputs": _read_jsonl(path)}


@app.get("/api/trace")
def trace():
    """Return latest execution trace events (last 200)."""
    path = _PROJECT / "outputs" / "trace.jsonl"
    events = _read_jsonl(path)
    return {"events": events[-200:], "total": len(events)}


@app.post("/api/sql")
def run_sql(body: SQLRequest):
    """Execute a raw SELECT query against Northwind SQLite."""
    _require_agent()
    sql = body.sql.strip()
    if not sql:
        raise HTTPException(status_code=400, detail="'sql' is required")
    if not sql.upper().lstrip().startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Only SELECT statements are allowed")
    try:
        from tools.sqlite_tool import execute_sql
        columns, rows, error = execute_sql(sql)
        if error:
            return JSONResponse(
                status_code=400,
                content={"error": error, "columns": [], "rows": []},
            )
        return {"columns": columns, "rows": rows[:500], "row_count": len(rows)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/optimize")
def optimize():
    """Run DSPy NL→SQL BootstrapFewShot optimization and return before/after metrics."""
    _require_agent()
    try:
        from agent.dspy_modules import NL2SQLModule
        from agent.optimizer import run_optimization
        _, metrics = run_optimization(NL2SQLModule())
        return {"metrics": metrics}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# static files mounted below with app creation


@app.get("/")
def index():
    return FileResponse(str(_FRONTEND / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
