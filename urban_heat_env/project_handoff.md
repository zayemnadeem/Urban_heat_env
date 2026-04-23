# 🚀 Project Handoff: Urban Heat Enterprise (Hackathon Round 2)

**Handoff Date:** April 19, 2026

## 1. The Objective
Upgrading the "Urban Heat Mitigation Environment" from a simple 15-step grid game to a complex **Enterprise City Planning Simulation** for the Meta x HuggingFace Hackathon Round 2. The goal is to maximize scores in Environment Innovation (40%) and Storytelling (30%) by demonstrating complex agent behavior.

## 2. The Architecture ("The Triple Threat")
We integrated three Hackathon themes into a single FastAPI-driven microservice:
- **Theme #3: World Modeling / Enterprise Workflows:** The agent cannot place items directly. It must use a tool-chain sequence: `query_zoning` → `propose_budget` → `deploy_intervention`.
- **Theme #1: Multi-Agent Interaction:** Budget proposals are routed through a **Simulated Mayor** agent. The Mayor has hidden biases, rejecting any proposal not located in a high-density zone (`density >= 0.4`).
- **Theme #2: Long-Horizon Planning:** The simulation horizon is **120 steps (10 years)**. 
    - **Delayed Physics:** Trees take 12 months to grow; reflective paint decays over 3 years.
    - **Dynamic Crisis:** Seasonal heatwaves (+4°C) occur every 12 months, forcing long-term resource pacing rather than reactive spending.

## 3. Key Files & Tech Stack
- **`environment.py`:** Core logic using NumPy. Handles the Mayor's veto logic, quarterly budget infusions (+5.0/year), and the 120-step physics engine.
- **`models.py`:** Pydantic validation for the Action/State schemas.
- **`inference.py`:** The LLM client loop. Includes a robust Regex extraction layer to handle conversational LLM noise and a random fallback for testing.
- **`train_trl.py`:** A production-ready training script using **Hugging Face TRL (PPO)**. Decoupled from the env, it trains the LLM to navigate the tool-chain and learn the Mayor's hidden rules.
- **`dashboard.html`:** A high-end Vanilla JS dashboard for real-time spectating and heat-map visualization.
- **`Dockerfile`:** Standardizes the environment for deployment on HF Spaces or local Docker Desktop.

## 4. Strategic Artifacts
- `pitch_and_scoring_guide.md`: 3-minute pitch script and scoring breakdown.
- `qa_prep_guide.md`: Detailed technical Q&A for mentors and judges (Regex logic, PPO theory, 8x8 grid justification).
- `submission_definition.md`: Pre-written answers for the 6 mandatory submission points.
- `impress_the_judges.md`: "Humble Brag" strategies to hit the Scaler AI bonus and Engineering-grade talking points.

## 5. Current State
The code is locally verified and pushed to GitHub: `Shoaibahmed-2005/urban-heat-enterprise`. The environment is 100% "Round 2 Ready." The next step is running `train_trl.py` on hackathon-provided compute to generate the reward-improvement graphs.
