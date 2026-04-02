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

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env.environment import WorkforceEnv
from env.models import (
    Action,
    HealthResponse,
    Observation,
    ResetRequest,
    StepRequest,
    WorkforceState,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="openenv-workforce",
    description=(
        "OpenEnv-compliant workforce mobility simulation. "
        "Relocate employees across countries while navigating visa, "
        "tax, and compliance rules. Includes UAE no-tax trap."
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

# ---------------------------------------------------------------------------
# Session store — in-memory dict: session_id → WorkforceEnv instance
# ---------------------------------------------------------------------------

_sessions: dict[str, WorkforceEnv] = {}

MAX_SESSIONS = 100  # prevent unbounded growth


def _get_session(session_id: str) -> WorkforceEnv:
    """Retrieve a session or raise 404."""
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
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    """
    Health check endpoint — used by HuggingFace Spaces automated ping.
    Returns 200 OK with environment name and version.
    """
    return HealthResponse(status="ok", name="openenv-workforce", version="1.0.0")


@app.post("/reset", tags=["environment"])
def reset(request: ResetRequest) -> dict[str, Any]:
    """
    Start a new episode.

    Request body:
        task_name:   "easy" | "medium" | "hard"  (default: "easy")
        session_id:  Optional custom session ID. Auto-generated if not provided.

    Returns:
        {
            "session_id":    str,
            "task_name":     str,
            "observation":   Observation,
        }
    """
    # Evict oldest session if at capacity
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
        "session_id":  session_id,
        "task_name":   request.task_name,
        "observation": observation.model_dump(),
    }


@app.post("/step", tags=["environment"])
def step(request: StepRequest) -> dict[str, Any]:
    """
    Apply one agent action to the environment.

    Request body:
        session_id: str       — from POST /reset response
        action:     Action    — { action_type: str, target: str }

    Returns:
        {
            "observation": Observation,
            "reward":      float,
            "done":        bool,
            "info":        dict,
        }
    """
    env = _get_session(request.session_id)
    result = env.step(request.action)

    # Clean up session after episode ends to free memory
    if result.done:
        _sessions.pop(request.session_id, None)

    return {
        "observation": result.observation.model_dump(),
        "reward":      result.reward,
        "done":        result.done,
        "info":        result.info,
    }


@app.get("/state", tags=["environment"])
def get_state(session_id: str) -> dict[str, Any]:
    """
    Return the current full state without applying any action.

    Query param:
        session_id: str — from POST /reset response

    Returns:
        WorkforceState as a dict.
    """
    env = _get_session(session_id)
    return env.state().model_dump()


@app.get("/sessions", tags=["admin"], include_in_schema=False)
def list_sessions() -> dict[str, Any]:
    """Admin endpoint — list active sessions (not exposed in public docs)."""
    return {
        "active_sessions": len(_sessions),
        "session_ids":     list(_sessions.keys()),
        "max_sessions":    MAX_SESSIONS,
    }
