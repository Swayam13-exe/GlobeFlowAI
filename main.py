"""
main.py
=======
FastAPI application for openenv-workforce.

Endpoints:
  GET  /              → health check (HuggingFace Space ping)
  GET  /health        → health check (explicit)
  GET  /tasks         → list available tasks
  POST /reset         → start a new episode
  POST /step          → apply one action
  GET  /state         → read current state
  POST /grade         → grade current state

Session management:
  Each POST /reset creates a new session (UUID).
  Sessions held in-memory — restarting clears all sessions.

Request format for /step is FLAT (aligns with inference.py):
  {"action_type": "...", "target": "...", "session_id": "..."}

Author: Team AI Kalesh
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import WorkforceEnv
from env.models import Action

# ---------------------------------------------------------------------------
# Request models — flat format matching inference.py
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name:  str           = "easy"
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    action_type: str           = "request_document"
    target:      str           = ""
    session_id:  Optional[str] = None   # optional — falls back to last active session


class GradeRequest(BaseModel):
    session_id: Optional[str] = None
    task_name:  Optional[str] = None


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

_sessions:      dict[str, WorkforceEnv] = {}
_last_session:  str | None              = None   # fallback when session_id not sent
MAX_SESSIONS    = 50


def _get_session(session_id: str | None) -> WorkforceEnv:
    """Get session by ID or fall back to last active session."""
    sid = session_id or _last_session
    if not sid or sid not in _sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call POST /reset first.",
        )
    return _sessions[sid]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _last_session
    try:
        env = WorkforceEnv()
        env.reset("easy")
        sid = str(uuid.uuid4())
        _sessions[sid] = env
        _last_session = sid
    except Exception:
        pass
    yield

app = FastAPI(
    title="openenv-workforce",
    description="Global Mobility & Compliance Orchestrator — OpenEnv RL Environment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup — pre-load so HF Space ping returns 200 immediately
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
@app.get("/health")
async def health() -> dict[str, str]:
    """Health check — required by HuggingFace Space automated ping."""
    return {
        "status":      "ok",
        "environment": "openenv-workforce",
        "version":     "1.0.0",
    }


@app.get("/tasks")
async def list_tasks() -> dict[str, Any]:
    return {
        "tasks": ["easy", "medium", "hard"],
        "descriptions": {
            "easy":   "India → Germany | Engineer | HR + 4 docs + tax_id + payroll",
            "medium": "India → Singapore | Manager | HR + Legal + 3 docs + PDPA + shadow payroll",
            "hard":   "India → Germany + UAE | Director | All depts + conflict resolution",
        },
    }


@app.post("/reset")
async def reset(request: ResetRequest) -> dict[str, Any]:
    """
    Start a new episode.

    Body: {"task_name": "easy" | "medium" | "hard", "session_id": "<optional>"}
    Returns: {"session_id": str, "observation": dict}
    """
    global _last_session

    if request.task_name not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{request.task_name}'. Valid: easy, medium, hard",
        )

    # Evict oldest session if at capacity
    if len(_sessions) >= MAX_SESSIONS:
        oldest = next(iter(_sessions))
        _sessions.pop(oldest)

    env = WorkforceEnv()
    try:
        observation = env.reset(task_name=request.task_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    sid = request.session_id or env._session_id
    _sessions[sid] = env
    _last_session = sid

    return {
        "session_id":  sid,
        "observation": observation.model_dump(),
    }


@app.post("/step")
async def step(request: StepRequest) -> dict[str, Any]:
    """
    Apply one action.

    Body: {"action_type": "...", "target": "...", "session_id": "<optional>"}
    Returns: {"observation": dict, "reward": float, "done": bool, "info": dict}
    """
    env = _get_session(request.session_id)

    try:
        action = Action(
            action_type=request.action_type,
            target=request.target,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid action: {exc}")

    try:
        result = env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Clean up finished sessions
    if result.done:
        sid = request.session_id or _last_session
        _sessions.pop(sid, None)

    return {
        "observation": result.observation.model_dump(),
        "reward":      result.reward,
        "done":        result.done,
        "info":        result.info,
    }


@app.get("/state")
async def state(session_id: Optional[str] = None) -> dict[str, Any]:
    """
    Get current episode state without mutating it.

    Query param: ?session_id=<id>  (optional — uses last active session)
    """
    env = _get_session(session_id)
    try:
        return env.state().model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/grade")
async def grade_endpoint(request: GradeRequest) -> dict[str, Any]:
    """
    Grade the current state and return score.

    Body: {"session_id": "<optional>", "task_name": "<optional>"}
    Returns: {"task": str, "score": float, "status": str}
    """
    env = _get_session(request.session_id)

    try:
        from graders.graders import grade

        current_state = env.state()
        task_name     = request.task_name or current_state.task_name or "easy"

        # Build grader-compatible flat dict
        state_dict = _flatten_for_grader(current_state)
        score       = grade(task_name, state_dict)

        return {
            "task":   task_name,
            "score":  round(score, 4),
            "status": current_state.status,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/sessions", include_in_schema=False)
async def list_sessions() -> dict[str, Any]:
    return {"count": len(_sessions), "ids": list(_sessions.keys())}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _flatten_for_grader(ws: Any) -> dict[str, Any]:
    """Convert WorkforceState model to flat dict for graders."""
    d = ws.model_dump()

    # Ensure documents are plain dicts
    docs = {}
    for name, doc in d.get("documents", {}).items():
        docs[name] = doc if isinstance(doc, dict) else {
            "status": doc.status, "is_valid": doc.is_valid
        }
    d["documents"] = docs

    # Ensure departments is a plain dict
    depts = d.get("departments", {})
    if not isinstance(depts, dict):
        depts = {"HR": depts.HR, "Legal": depts.Legal, "Finance": depts.Finance}
    d["departments"] = depts

    # Ensure compliance is a plain dict
    comp = d.get("compliance", {})
    if not isinstance(comp, dict):
        comp = {
            "tax_id":         comp.tax_id,
            "payroll":        comp.payroll,
            "pdpa":           comp.pdpa,
            "shadow_payroll": comp.shadow_payroll,
        }
    d["compliance"] = comp

    # Flatten conflicts
    conflicts = d.get("conflicts", [])
    d["conflicts"] = [
        c if isinstance(c, dict) else {
            "countries": c.countries,
            "rule":      c.rule,
            "resolved":  c.resolved,
        }
        for c in conflicts
    ]

    return d


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()