from pydantic import BaseModel
from typing import Any


class CellState(BaseModel):
    row: int
    col: int
    surface_type: str
    temperature: float
    population_density: float


class ActiveIntervention(BaseModel):
    row: int
    col: int
    intervention_type: str
    age: int


class Proposal(BaseModel):
    row: int
    col: int
    intervention_type: str
    status: str  # "pending", "approved", "rejected"


class CityState(BaseModel):
    grid: list[list[CellState]]
    budget: float
    step_count: int
    avg_temperature: float
    episode_done: bool
    season: str
    next_heatwave_in: int
    active_interventions: list[ActiveIntervention]
    proposals: list[Proposal]
    zoning_queries: list[str]


class ActionRequest(BaseModel):
    task_id: str
    action_type: str  # "query_zoning", "propose_budget", "deploy_intervention"
    row: int
    col: int
    intervention_type: str = ""  # Optional, needed for budget and deploy


class TaskResult(BaseModel):
    task_id: str
    score: float  # 0.0 to 1.0
    details: dict[str, Any]


class ObsResult(BaseModel):
    state: CityState
    reward: float
    done: bool
    info: dict[str, Any]


class TaskDefinition(BaseModel):
    id: str
    difficulty: str
    description: str
