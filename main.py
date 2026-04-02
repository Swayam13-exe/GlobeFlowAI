from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
from env.environment import GlobeFlowEnv

app = FastAPI(title="GlobeFlowAI Environment API")

# Single global instance for simplicity in standard Docker setup
env = GlobeFlowEnv()

class ResetRequest(BaseModel):
    task_level: str = "easy"

class StepRequest(BaseModel):
    action_type: str
    target: str

@app.post("/reset")
def reset_env(req: Optional[ResetRequest] = None):
    level = req.task_level if req else "easy"
    return env.reset(task_level=level)

@app.get("/state")
def get_state():
    return env.state()

@app.post("/step")
def step_env(action: StepRequest):
    next_state, reward, done, info = env.step(action.model_dump())
    return {
        "observation": next_state,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/")
def read_root():
    return {"status": "ok", "app": "GlobeFlowAI"}
