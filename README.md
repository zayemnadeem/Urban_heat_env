---
title: Urban Heat Env (Triple Threat Edition)
emoji: 🌡️
colorFrom: red
colorTo: green
sdk: docker
pinned: false
---

# Urban Heat Island Mitigation Planner — Round 2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shoaibahmed-2005/Urban-Heat-Round-2/blob/main/train_trl.ipynb)

An intricately designed, multi-theme RL environment for the Meta × HuggingFace OpenEnv Hackathon.

## Overview

This environment perfectly encapsulates three core Hackathon themes:
1. **World Modeling (Enterprise Workflows):** The agent must navigate simulated APIs (`query_zoning`, `propose_budget`, `deploy_intervention`) to get anything built. 
2. **Multi-Agent Interactions:** Budget proposals are routed through a simulated "Mayor" actor who possesses hidden biases (e.g., rejecting projects not located in high-density areas).
3. **Long-Horizon Planning:** The simulation runs for 120 steps representing 10 years. Interventions like `tree_canopy` take 12 months to grow, while `reflective_surface` degrades over 3 years. The agent must successfully prepare for "Summer Heatwaves" that spawn every 12 months.

## Step-by-Step Guide

### 1. Define Models

The environment uses Python Pydantic models to define the State, Action, and Observation spaces (found in `models.py`).

**Action Space (API Tool Calling)**
Actions are submitted as JSON requests to the environment:
- **`query_zoning`**: `{"action_type": "query_zoning", "row": <0-7>, "col": <0-7>}`
- **`propose_budget`**: `{"action_type": "propose_budget", "intervention_type": "<type>", "row": <0-7>, "col": <0-7>}`
- **`deploy_intervention`**: `{"action_type": "deploy_intervention", "intervention_type": "<type>", "row": <0-7>, "col": <0-7>}`

**State & Observation**
The State consists of an 8x8 city grid where each cell has properties like surface type, temperature, and population density. The environment also tracks global state such as budget, step count, active interventions, and proposals.

### 2. Implement Environment & Reward Functions

The core environment is implemented in `server/environment.py` and features the primary simulation methods: `reset()`, `step()`, and `state()`. 

**Reward Functions and Tasks**

The environment supports three distinct tasks, each with its own step reward shaping and final grading mechanics:

1. **`reduce_avg_temp` (Easy)**
   - **Step Reward:** The reduction in the average grid temperature compared to the previous step (only positive improvements are rewarded). This sparse-style reward encourages continuous temperature reduction.
   - **Final Grade:** Scaled based on the total reduction of the average temperature over the baseline (max score achieved at a 2.0°C reduction).

2. **`protect_dense_zones` (Medium)**
   - **Step Reward:** The average temperature reduction across the top 5 most densely populated cells.
   - **Final Grade:** The percentage of those top 5 high-density cells that were successfully cooled by at least 1.5°C.

3. **`full_mitigation` (Hard)**
   - **Step Reward:** The reduction in the average grid temperature compared to the previous step.
   - **Final Grade:** A composite score weighting average temperature reduction (50%) and population coverage (50%). Population coverage measures the proportion of the city's total population living in cells cooled by at least 1.0°C.

**Interventions**
- `green_roof`: High immediate cooling, charges a 0.1 budget maintenance fee/step.
- `reflective_surface`: Immediate cooling, completely degrades over 36 steps (3 years).
- `tree_canopy`: Peak cooling, but starts at 0 and takes 12 steps (1 year) to fully mature.

### 3. Create FastAPI Server

The HTTP API is built using FastAPI (`server/app.py`). It exposes endpoints such as `/health`, `/reset`, `/step`, `/state`, and `/grade/{task_id}` to interact with the simulated city planner.

### 4. Define Dependencies

Required dependencies for the environment are tracked in `requirements.txt`. They include:
- `fastapi`
- `uvicorn`
- `pydantic`
- `numpy`
- `requests`

### 5. Create Dockerfile

A `Dockerfile` is provided at the root level to seamlessly containerize the FastAPI server and the environment, ensuring it runs identically across different environments and supports Hugging Face Spaces deployment out-of-the-box.

### 6. Implement Client

Clients interact with the environment by making HTTP POST/GET requests to the server's endpoints. Reference implementations of agents driving the environment can be found in our baseline LLM and RL training scripts (`inference.py` and `train_trl.py`).

## Building and Using Your Environment

### Setup and Local Execution

<<<<<<< HEAD
## OpenEnv Validation & Documentation
This environment is fully compatible with the OpenEnv standard (see `openenv.yaml` for tasks and API routing).
For Hackathon judges and mentors, please review `project_handoff.md` for a complete architecture overview and our "Triple Threat" hackathon strategy.

## Setup
=======
>>>>>>> 37bac00ae309dd9d91d8588635e31b31d3f47bda
```bash
# Install dependencies (using pip or uv)
pip install -r requirements.txt
# Copy environment variables
cp .env.example .env
```

<<<<<<< HEAD
## Running the Server & Dashboard

### Local Execution
=======
### Running the Server & Dashboard

>>>>>>> 37bac00ae309dd9d91d8588635e31b31d3f47bda
```bash
# Start the simulation backend
uvicorn server.app:app --host 0.0.0.0 --port 8000

# To view the visual dashboard, simply open dashboard.html in your browser!
```

<<<<<<< HEAD
### Docker (Hugging Face Spaces)
A `Dockerfile` is included for easy deployment. Note that the Docker image exposes port `7860`.
```bash
docker build -t urban-heat-env .
docker run -p 7860:7860 urban-heat-env
```

## Running RL Training 
To verify our ability to solve this complex space, you can run the provided Hugging Face TRL PPO training script:
=======
### Running RL Training 

To verify the ability to solve this complex space, you can run the provided Hugging Face TRL PPO training script:

>>>>>>> 37bac00ae309dd9d91d8588635e31b31d3f47bda
```bash
python train_trl.py
```
*(Note: Requires valid Hugging Face authentication if running outside of Colab)*

### Running the Baseline LLM Inference

We provide a baseline LLM inference script to interact with the environment using models like Qwen:

```bash
python inference.py
```

## Project Structure

```text
urban_heat_env/
├── README.md             # Environment documentation
├── models.py             # Action, Observation, State definitions
├── inference.py          # Baseline LLM inference script
├── train_trl.py          # RL Training using Hugging Face TRL
├── dashboard.html        # Interactive visual dashboard
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image definition
└── server/
    ├── __init__.py
    ├── environment.py    # Core CityGrid environment logic
    └── app.py            # FastAPI application
```