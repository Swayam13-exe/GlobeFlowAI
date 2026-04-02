"""
main.py
=======
FastAPI application for openenv-workforce.

Endpoints:
  GET  /              → health check (HuggingFace Space ping)
  POST /reset         → start a new episode
  POST /step          → apply one action
  GET  /state         → read current state without mutating it

Session management:
  Each POST /reset creates a new session (UUID).
  All subsequent calls pass session_id in the request body.
  Sessions are held in-memory — restarting the server clears all sessions.

Author: Team AI Kalesh
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from env.environment import WorkforceEnv
from env.models import (
    HealthResponse,
    ResetRequest,
    StepRequest,
)

# ---------------------------------------------------------------------------
# Session store — in-memory dict: session_id → WorkforceEnv instance
# ---------------------------------------------------------------------------

_sessions: dict[str, WorkforceEnv] = {}
MAX_SESSIONS = 100


def _get_session(session_id: str) -> WorkforceEnv:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session '{session_id}' not found. "
                "Call POST /reset to start a new episode."
            ),
        )
    return env


# ---------------------------------------------------------------------------
# App factory (IMPORTANT: no global app)
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="openenv-workforce",
        description=(
            "OpenEnv-compliant workforce mobility simulation. "
            "Relocate employees across countries while navigating visa, "
            "tax, and compliance rules."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.get("/", response_model=HealthResponse, tags=["health"])
    def health_check() -> HealthResponse:
        return HealthResponse(status="ok", name="openenv-workforce", version="1.0.0")

    @app.post("/reset", tags=["environment"])
    def reset(request: ResetRequest) -> dict[str, Any]:
        if len(_sessions) >= MAX_SESSIONS:
            oldest_key = next(iter(_sessions))
            del _sessions[oldest_key]

        env = WorkforceEnv()

        try:
            observation = env.reset(task_name=request.task_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        session_id = request.session_id or env._session_id
        _sessions[session_id] = env

        return {
            "session_id": session_id,
            "task_name": request.task_name,
            "observation": observation.model_dump(),
        }

    @app.post("/step", tags=["environment"])
    def step(request: StepRequest) -> dict[str, Any]:
        env = _get_session(request.session_id)
        result = env.step(request.action)

        if result.done:
            _sessions.pop(request.session_id, None)

        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
        }

    @app.get("/state", tags=["environment"])
    def get_state(session_id: str) -> dict[str, Any]:
        env = _get_session(session_id)
        return env.state().model_dump()

    @app.get("/sessions", include_in_schema=False)
    def list_sessions():
        return {
            "active_sessions": len(_sessions),
            "session_ids": list(_sessions.keys()),
        }

    return app