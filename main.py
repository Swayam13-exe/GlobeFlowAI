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
from env.models import HealthResponse, ResetRequest, StepRequest

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

_sessions: dict[str, WorkforceEnv] = {}
MAX_SESSIONS = 100


def _get_session(session_id: str) -> WorkforceEnv:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call POST /reset first."
        )
    return env


# ---------------------------------------------------------------------------
# App factory (NO global app)
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    api = FastAPI(
        title="openenv-workforce",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------- ROUTES ----------------------

    @api.get("/", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok", name="openenv-workforce")

    @api.post("/reset")
    def reset(request: ResetRequest) -> dict[str, Any]:
        if len(_sessions) >= MAX_SESSIONS:
            _sessions.pop(next(iter(_sessions)))

        env = WorkforceEnv()
        observation = env.reset(task_name=request.task_name)

        session_id = request.session_id or env._session_id
        _sessions[session_id] = env

        return {
            "session_id": session_id,
            "observation": observation.model_dump()
        }

    @api.post("/step")
    def step(request: StepRequest):
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

    @api.get("/state")
    def state(session_id: str):
        env = _get_session(session_id)
        return env.state().model_dump()

    @api.get("/sessions", include_in_schema=False)
    def sessions():
        return {
            "count": len(_sessions),
            "ids": list(_sessions.keys())
        }

    return api

app = create_app()

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
