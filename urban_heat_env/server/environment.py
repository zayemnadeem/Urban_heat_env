"""
Core Urban Heat Island environment logic.

The 8x8 city grid holds cells with surface type, temperature, and population
density.  An RL agent targets the Long-Horizon, World Modeling, and Multi-agent 
themes by calling simulated APIs.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
from typing import Any

from models import CellState, CityState, TaskResult, ActiveIntervention, Proposal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE = 8
INITIAL_BUDGET = 20.0
MAX_STEPS = 120
BUDGET_INFUSION_INTERVAL = 12
BUDGET_INFUSION_AMOUNT = 5.0
HEATWAVE_INTERVAL = 12
HEATWAVE_MAGNITUDE = 4.0

SURFACE_TYPES = ["road", "building", "park", "water"]

# Intervention specs: cost (budget units), radius (cells), temp_reduction (°C)
INTERVENTIONS: dict[str, dict[str, float]] = {
    "green_roof":         {"cost": 2.0, "radius": 1, "temp_reduction": 3.0, "maintenance": 0.1},
    "reflective_surface": {"cost": 1.0, "radius": 1, "temp_reduction": 1.5},
    "tree_canopy":        {"cost": 3.0, "radius": 2, "temp_reduction": 4.0},
}

TASK_IDS = ["reduce_avg_temp", "protect_dense_zones", "full_mitigation"]

# ---------------------------------------------------------------------------
# CityGrid
# ---------------------------------------------------------------------------

class CityGrid:
    def __init__(self) -> None:
        self.surface_types: list[list[str]] = []
        self.temperatures: np.ndarray = np.zeros((GRID_SIZE, GRID_SIZE))
        self.base_temperatures: np.ndarray = np.zeros((GRID_SIZE, GRID_SIZE))
        self.population_density: np.ndarray = np.zeros((GRID_SIZE, GRID_SIZE))
        self.budget: float = INITIAL_BUDGET
        self.step_count: int = 0
        self.done: bool = False
        
        self.active_interventions: list[dict] = []
        self.proposals: list[dict] = []
        self.zoning_queries: list[str] = []

    def reset(self, seed: int = 42) -> CityState:
        """Initialise/reinitialise the grid deterministically."""
        np.random.seed(seed)

        self.step_count = 0
        self.budget = INITIAL_BUDGET
        self.done = False
        self.active_interventions = []
        self.proposals = []
        self.zoning_queries = []

        self.surface_types = [
            [
                np.random.choice(SURFACE_TYPES, p=[0.35, 0.40, 0.15, 0.10])
                for _ in range(GRID_SIZE)
            ]
            for _ in range(GRID_SIZE)
        ]

        surface_base: dict[str, tuple[float, float]] = {
            "road":     (35.0, 42.0),
            "building": (30.0, 40.0),
            "park":     (25.0, 32.0),
            "water":    (25.0, 28.0),
        }
        temps = np.zeros((GRID_SIZE, GRID_SIZE))
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                lo, hi = surface_base[self.surface_types[r][c]]
                temps[r, c] = np.round(np.random.uniform(lo, hi), 2)

        self.temperatures = temps.copy()
        self.base_temperatures = temps.copy()

        self.population_density = np.round(
            np.random.uniform(0.0, 1.0, (GRID_SIZE, GRID_SIZE)), 3
        )

        return self._build_city_state()

    def get_season(self) -> str:
        # 12 steps per year (months). Summer is Jun, Jul, Aug (months 5,6,7 index 0-based)
        month = self.step_count % 12
        if month in [2, 3, 4]: return "Spring"
        if month in [5, 6, 7]: return "Summer"
        if month in [8, 9, 10]: return "Autumn"
        return "Winter"

    def step(
        self,
        row: int,
        col: int,
        action_type: str,
        intervention_type: str,
        task_id: str,
    ) -> tuple[CityState, float, bool, dict[str, Any]]:
        
        info: dict[str, Any] = {"error": None}
        
        if self.done:
            return self._build_city_state(), 0.0, True, {"error": "episode already done"}
            
        row = max(0, min(GRID_SIZE-1, row))
        col = max(0, min(GRID_SIZE-1, col))

        # ---------------------------------------------------------
        # Theme 3: World Modeling / Enterprise workflow
        # ---------------------------------------------------------
        if action_type == "query_zoning":
            if self.surface_types[row][col] == "water":
                self.zoning_queries.append(f"Zoning: Denied at {row},{col}. Cannot build on water.")
            else:
                self.zoning_queries.append(f"Zoning: Allowed at {row},{col}.")
                
        elif action_type == "propose_budget":
            # Theme 1: Multi-Agent Interaction. The Mayor Agent only approves if density > 0.4 
            # and it's practically viable to do something.
            if intervention_type not in INTERVENTIONS:
                info["error"] = f"Unknown intervention: {intervention_type}"
            else:
                density = self.population_density[row][col]
                status = "approved" if density >= 0.4 and self.surface_types[row][col] != "water" else "rejected"
                
                # Check if proposal exists
                exists = False
                for p in self.proposals:
                    if p["row"] == row and p["col"] == col and p["intervention_type"] == intervention_type:
                        p["status"] = status
                        exists = True
                        break
                if not exists:
                    self.proposals.append({
                        "row": row, "col": col, 
                        "intervention_type": intervention_type, 
                        "status": status
                    })
                    
        elif action_type == "deploy_intervention":
            if intervention_type not in INTERVENTIONS:
                info["error"] = f"Unknown intervention: {intervention_type}"
            else:
                approved = False
                for p in self.proposals:
                    if p["row"] == row and p["col"] == col and p["intervention_type"] == intervention_type and p["status"] == "approved":
                        approved = True
                        break
                
                if not approved:
                    info["error"] = "Deployment failed: No approved budget proposal from Mayor."
                else:
                    cost = INTERVENTIONS[intervention_type]["cost"]
                    if cost > self.budget:
                        info["error"] = "Insufficient budget to deploy."
                    else:
                        self.budget -= cost
                        self.active_interventions.append({
                            "row": row, "col": col, "intervention_type": intervention_type, "age": -1
                        })
                        info["success"] = f"Deployed {intervention_type} at {row},{col}"
        
        else:
            info["error"] = f"Unknown action_type {action_type}"

        # ---------------------------------------------------------
        # Time Passes (Long-Horizon)
        # ---------------------------------------------------------
        self.step_count += 1
        
        if self.step_count % BUDGET_INFUSION_INTERVAL == 0:
            self.budget += BUDGET_INFUSION_AMOUNT

        # Recompute temperatures based on aging interventions
        season = self.get_season()
        is_heatwave = (season == "Summer")
        
        current_temps = self.base_temperatures.copy()
        if is_heatwave:
            current_temps += HEATWAVE_MAGNITUDE
            
        remaining_interventions = []
        for inv in self.active_interventions:
            inv["age"] += 1
            age = inv["age"]
            itype = inv["intervention_type"]
            irun_row = inv["row"]
            irun_col = inv["col"]
            
            spec = INTERVENTIONS[itype]
            radius = spec["radius"]
            max_reduction = spec["temp_reduction"]
            
            # Decay/Growth mechanics
            if itype == "green_roof":
                # Maintenance fee
                self.budget -= spec["maintenance"]
                if self.budget < 0:
                    self.budget = 0.0
                    continue # It dies, don't keep it
                reduction = max_reduction
            elif itype == "tree_canopy":
                # Grows over 12 steps
                reduction = (min(12, max(1, age)) / 12.0) * max_reduction
            elif itype == "reflective_surface":
                # Decays over 36 steps
                reduction = max(0.0, (36 - age) / 36.0) * max_reduction
                if reduction <= 0:
                    continue # Decayed completely
                    
            remaining_interventions.append(inv)
                
            # Apply reduction
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = irun_row + dr, irun_col + dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        dist = max(abs(dr), abs(dc))
                        weight = 1.0 if dist == 0 else 1.0 / (dist + 1)
                        current_temps[nr, nc] = max(
                            20.0,
                            current_temps[nr, nc] - reduction * weight,
                        )
                        
        self.active_interventions = remaining_interventions
        self.temperatures = np.round(current_temps, 2)

        # Compute Step Reward
        reward = self._compute_step_reward(task_id, current_temps)

        if self.step_count >= MAX_STEPS:
            self.done = True

        new_state = self._build_city_state()
        return new_state, reward, self.done, info

    def grade_task(self, task_id: str) -> TaskResult:
        if task_id == "reduce_avg_temp":
            return self._grade_reduce_avg_temp()
        elif task_id == "protect_dense_zones":
            return self._grade_protect_dense_zones()
        elif task_id == "full_mitigation":
            return self._grade_full_mitigation()
        else:
            return TaskResult(task_id=task_id, score=0.0, details={"error": "Unknown task"})

    def _grade_reduce_avg_temp(self) -> TaskResult:
        initial_avg = float(np.mean(self.base_temperatures))
        current_avg = float(np.mean(self.temperatures))
        actual_reduction = initial_avg - current_avg
        score = float(np.clip(actual_reduction / 2.0, 0.0, 1.0))
        return TaskResult(
            task_id="reduce_avg_temp", score=score,
            details={"initial_avg_temp": round(initial_avg, 3), "current_avg_temp": round(current_avg, 3), "actual_reduction": round(actual_reduction, 3)}
        )

    def _grade_protect_dense_zones(self) -> TaskResult:
        flat_density = self.population_density.flatten()
        flat_indices = np.argsort(flat_density)[::-1][:5]
        rows = flat_indices // GRID_SIZE
        cols = flat_indices % GRID_SIZE

        cells_cooled = 0
        for r, c in zip(rows.tolist(), cols.tolist()):
            reduction = float(self.base_temperatures[r, c] - self.temperatures[r, c])
            if reduction >= 1.5: cells_cooled += 1
        score = float(cells_cooled / 5)
        return TaskResult(task_id="protect_dense_zones", score=score, details={"cells_cooled": cells_cooled})

    def _grade_full_mitigation(self) -> TaskResult:
        initial_avg = float(np.mean(self.base_temperatures))
        current_avg = float(np.mean(self.temperatures))
        avg_reduction = initial_avg - current_avg
        avg_temp_score = float(np.clip(avg_reduction / 3.0, 0.0, 1.0))

        reductions = self.base_temperatures - self.temperatures
        cooled_mask = reductions >= 1.0
        total_pop = float(np.sum(self.population_density))
        pop_coverage_score = float(np.clip(float(np.sum(self.population_density * cooled_mask)) / total_pop, 0.0, 1.0)) if total_pop > 0 else 0.0

        # We inject budget over time. Efficiency could just be checking current avg temp
        score = float(np.clip(0.5 * avg_temp_score + 0.5 * pop_coverage_score, 0.0, 1.0))
        return TaskResult(task_id="full_mitigation", score=score, details={"avg_temp_score": round(avg_temp_score, 3), "pop_coverage_score": round(pop_coverage_score, 3)})

    def _compute_step_reward(self, task_id: str, temp_before: np.ndarray) -> float:
        # Sparse reward style. Since it is long-horizon, we just track the current temp delta over time.
        # But for RL it helps to have delta improvement.
        if task_id == "reduce_avg_temp":
            reward = float(np.mean(temp_before) - np.mean(self.temperatures))
        elif task_id == "protect_dense_zones":
            flat_density = self.population_density.flatten()
            top5_idx = np.argsort(flat_density)[::-1][:5]
            rows = top5_idx // GRID_SIZE
            cols = top5_idx % GRID_SIZE
            reward = float(np.mean(temp_before[rows, cols] - self.temperatures[rows, cols]))
        else:
            reward = float(np.mean(temp_before - self.temperatures))
        return round(max(0.0, reward), 4) # Only reward positive improvements to prevent hacking by cycling

    def _build_city_state(self) -> CityState:
        grid: list[list[CellState]] = []
        for r in range(GRID_SIZE):
            row_cells: list[CellState] = []
            for c in range(GRID_SIZE):
                row_cells.append(CellState(
                    row=r, col=c,
                    surface_type=self.surface_types[r][c],
                    temperature=round(float(self.temperatures[r, c]), 2),
                    population_density=round(float(self.population_density[r, c]), 3),
                ))
            grid.append(row_cells)
            
        active = [ActiveIntervention(**i) for i in self.active_interventions]
        props = [Proposal(**p) for p in self.proposals]
        
        # Calculate steps to next heatwave
        month = self.step_count % 12
        # Summer is 5, 6, 7. Next start of summer is 5.
        next_heatwave = 5 - month if month <= 5 else 12 - month + 5

        return CityState(
            grid=grid,
            budget=round(self.budget, 2),
            step_count=self.step_count,
            avg_temperature=round(float(np.mean(self.temperatures)), 3),
            episode_done=self.done,
            season=self.get_season(),
            next_heatwave_in=next_heatwave,
            active_interventions=active,
            proposals=props,
            zoning_queries=list(self.zoning_queries)[-5:] # Keep last 5
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "surface_types": copy.deepcopy(self.surface_types),
            "temperatures": self.temperatures.copy(),
            "base_temperatures": self.base_temperatures.copy(),
            "population_density": self.population_density.copy(),
            "budget": self.budget,
            "step_count": self.step_count,
            "done": self.done,
            "active_interventions": copy.deepcopy(self.active_interventions),
            "proposals": copy.deepcopy(self.proposals),
            "zoning_queries": copy.deepcopy(self.zoning_queries)
        }

    def restore(self, snap: dict[str, Any]) -> None:
        self.surface_types = copy.deepcopy(snap["surface_types"])
        self.temperatures = snap["temperatures"].copy()
        self.base_temperatures = snap["base_temperatures"].copy()
        self.population_density = snap["population_density"].copy()
        self.budget = snap["budget"]
        self.step_count = snap["step_count"]
        self.done = snap["done"]
        self.active_interventions = copy.deepcopy(snap["active_interventions"])
        self.proposals = copy.deepcopy(snap["proposals"])
        self.zoning_queries = copy.deepcopy(snap["zoning_queries"])
