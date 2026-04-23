---
title: Urban Heat Env (Triple Threat Edition)
emoji: 🌡️
colorFrom: red
colorTo: green
sdk: docker
pinned: false
---

# Urban Heat Island Mitigation Planner — Round 2

An intricately designed, multi-theme RL environment for the Meta × HuggingFace OpenEnv Hackathon. 

This environment perfectly encapsulates three core Hackathon themes:
1. **World Modeling (Enterprise Workflows):** The agent must navigate simulated APIs (`query_zoning`, `propose_budget`, `deploy_intervention`) to get anything built. 
2. **Multi-Agent Interactions:** Budget proposals are routed through a simulated "Mayor" actor who possesses hidden biases (e.g., rejecting projects not located in high-density areas).
3. **Long-Horizon Planning:** The simulation runs for 120 steps representing 10 years. Interventions like `tree_canopy` take 12 months to grow, while `reflective_surface` degrades over 3 years. The agent must successfully prepare for "Summer Heatwaves" that spawn every 12 months.

## Action Space (API Tool Calling)
- `{"action_type": "query_zoning", "row": <0-7>, "col": <0-7>}`
- `{"action_type": "propose_budget", "intervention_type": "tree_canopy", "row": <0-7>, "col": <0-7>}`
- `{"action_type": "deploy_intervention", "intervention_type": "tree_canopy", "row": <0-7>, "col": <0-7>}`

## Interventions
- `green_roof`: High immediate cooling, charges a 0.1 budget maintenance fee/step.
- `reflective_surface`: Immediate cooling, completely degrades over 3 years.
- `tree_canopy`: Peak cooling, but starts at 0 and takes 1 year to fully mature.

## OpenEnv Validation & Documentation
This environment is fully compatible with the OpenEnv standard (see `openenv.yaml` for tasks and API routing).
For Hackathon judges and mentors, please review `project_handoff.md` for a complete architecture overview and our "Triple Threat" hackathon strategy.

## Setup
```bash
# Install dependencies (using pip or uv)
pip install -r requirements.txt
# Copy environment variables
cp .env.example .env
```

## Running the Server & Dashboard

### Local Execution
```bash
# Start the simulation backend
uvicorn server.app:app --host 0.0.0.0 --port 8000

# To view the visual dashboard, simply open dashboard.html in your browser!
```

### Docker (Hugging Face Spaces)
A `Dockerfile` is included for easy deployment. Note that the Docker image exposes port `7860`.
```bash
docker build -t urban-heat-env .
docker run -p 7860:7860 urban-heat-env
```

## Running RL Training 
To verify our ability to solve this complex space, you can run the provided Hugging Face TRL PPO training script:
```bash
python train_trl.py
```
*(Note: Requires valid Hugging Face authentication if running outside of Colab)*

## Running the Baseline LLM Inference
```bash
python inference.py
```