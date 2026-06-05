# server/app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import InventoryAction, InventoryObservation
from server.inventory_env import InventoryEnv
from server.grader import compute_baselines, score_episode
from server.constants import TASKS


# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HorizonEnv",
    description="A 90-step retail inventory RL environment for long-horizon LLM planning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env instance per server (single-session for now)
env: Optional[InventoryEnv] = None


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "easy"
    seed: int = 42

class GraderRequest(BaseModel):
    task_name: str
    agent_profit: float


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check if server is running."""
    return {"status": "ok", "project": "HorizonEnv"}


@app.post("/reset", response_model=InventoryObservation)
def reset(request: ResetRequest):
    """
    Start a new episode.
    Returns the initial observation.
    """
    global env

    if request.task_name not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{request.task_name}'. Choose from: {list(TASKS.keys())}"
        )

    env = InventoryEnv(task_name=request.task_name, seed=request.seed)
    obs = env.reset()
    return obs


@app.post("/step")
def step(action: InventoryAction):
    """
    Submit one day's action.
    Returns next observation, reward, and done flag.
    """
    global env

    if env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    if env.state.current_day > env.state.total_days:
        raise HTTPException(
            status_code=400,
            detail="Episode is already done. Call /reset to start a new one."
        )

    obs, reward, done = env.step(action)

    return {
        "observation": obs,
        "reward": round(reward, 4),
        "done": done,
        "day": obs.current_day,
        "total_profit": obs.total_profit,
    }


@app.get("/state")
def state():
    """Get a quick summary of current episode state."""
    global env

    if env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    return env.get_state_summary()


@app.get("/tasks")
def tasks():
    """List all available tasks with their configs."""
    return {
        name: {
            "name": name,
            "description": cfg["description"],
            "starting_cash": cfg["starting_cash"],
            "num_directives": cfg["num_directives"],
            "num_events": cfg["num_events"],
            "has_conflicts": cfg["has_conflicting_directives"],
            "has_deceptive": cfg["has_deceptive_directive"],
        }
        for name, cfg in TASKS.items()
    }


@app.post("/grader")
def grader(request: GraderRequest):
    """
    Score an agent's total profit for a task.
    Returns a 0.0 - 1.0 score.
    """
    if request.task_name not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{request.task_name}'. Choose from: {list(TASKS.keys())}"
        )

    floor, ceiling = compute_baselines(request.task_name)
    score = score_episode(request.task_name, request.agent_profit)

    return {
        "task": request.task_name,
        "agent_profit": request.agent_profit,
        "floor": round(floor, 2),
        "ceiling": round(ceiling, 2),
        "score": score,
        "interpretation": _interpret_score(score),
    }


@app.get("/baselines")
def baselines():
    """Compute floor and ceiling baselines for all tasks."""
    result = {}
    for task_name in TASKS:
        floor, ceiling = compute_baselines(task_name)
        result[task_name] = {
            "floor": round(floor, 2),
            "ceiling": round(ceiling, 2),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _interpret_score(score: float) -> str:
    if score >= 0.8:
        return "Excellent — well above heuristic baseline"
    elif score >= 0.6:
        return "Good — above average performance"
    elif score >= 0.4:
        return "Average — room for improvement"
    elif score >= 0.2:
        return "Below average — agent struggling"
    else:
        return "Poor — close to passive baseline"