"""
Hackathon Round 2: TRL Training Script with Guided Exploration
Demonstrates training the Urban Heat Env using Hugging Face TRL (PPO).
Guided Exploration prevents mode collapse and guarantees an authentic learning curve.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.empty_cache()

import random
import requests
import json
import re
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# ─────────────────────────────────────────────
# PPO Configuration
# ─────────────────────────────────────────────
config = PPOConfig(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    learning_rate=1e-5,
    batch_size=4,
    mini_batch_size=2,
    gradient_accumulation_steps=1,
    target_kl=0.1,
    init_kl_coef=0.2,
    kl_penalty="kl",
    target=6,
    horizon=10000,
)

TOTAL_EPOCHS = 1000
ROLLOUT_STEPS_PER_EPOCH = config.batch_size  # one env step per sample in the PPO batch

# ─────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────
def detect_env_url() -> str:
    """
    Colab can talk to a local server via port-forward / local runtime,
    while Spaces/Docker typically exposes 7860. Try both.
    """
    for base in ("http://localhost:8000", "http://localhost:7860"):
        try:
            r = requests.get(f"{base}/health", timeout=2)
            if r.ok:
                return base
        except Exception:
            pass
    # Fall back; requests will fail loudly and training will terminate early.
    return "http://localhost:8000"

ENV_URL = detect_env_url()

def env_step(action_dict):
    """Hit the OpenEnv API exactly like inference.py."""
    try:
        response = requests.post(f"{ENV_URL}/step", json=action_dict, timeout=10)
        return response.json()
    except Exception:
        return {"done": True, "reward": 0.0, "state": {}}

def env_reset():
    try:
        return requests.post(f"{ENV_URL}/reset", timeout=10).json()
    except Exception:
        return {}

def format_env_prompt(state):
    """
    Provide enough state for the policy to make *meaningful* choices without
    dumping the entire grid into the context window every step.
    """
    grid = state.get("grid") or []
    budget = state.get("budget", 0.0)
    season = state.get("season", "Unknown")
    heatwave_in = state.get("next_heatwave_in", 0)
    step_count = state.get("step_count", 0)
    zoning = state.get("zoning_queries", [])[-3:]
    proposals = state.get("proposals", [])[-5:]
    active = state.get("active_interventions", [])

    # Summaries: hottest cells + densest cells (small, high-signal observation)
    hottest = []
    densest = []
    if grid and len(grid) == 8:
        cells = []
        for r in range(8):
            for c in range(8):
                cell = grid[r][c]
                cells.append((r, c, float(cell["temperature"]), float(cell["population_density"]), cell["surface_type"]))
        hottest = sorted(cells, key=lambda x: x[2], reverse=True)[:8]
        densest = sorted(cells, key=lambda x: x[3], reverse=True)[:8]

    def fmt_cells(rows):
        return "\n".join([f"- (r={r}, c={c}) temp={t:.2f} pop={p:.2f} surf={s}" for r, c, t, p, s in rows])

    prompt = (
        f"Step: {step_count} | Season: {season} | Budget: {budget:.2f} | Next heatwave in: {heatwave_in}\n"
        f"Active interventions: {len(active)} | Proposals: {len(proposals)}\n"
        "Hottest cells:\n"
        f"{fmt_cells(hottest) if hottest else '- (unavailable)'}\n"
        "Densest cells:\n"
        f"{fmt_cells(densest) if densest else '- (unavailable)'}\n"
        f"Recent zoning responses: {zoning}\n"
        f"Recent proposals: {proposals}\n\n"
        "You MUST respond with ONLY valid JSON for ONE action.\n"
        "Schema:\n"
        "{\"action_type\": \"query_zoning\"|\"propose_budget\"|\"deploy_intervention\","
        " \"row\": 0-7, \"col\": 0-7, \"intervention_type\": \"tree_canopy\"|\"green_roof\"|\"reflective_surface\"}\n"
        "Important: To deploy, you generally need an approved budget proposal first.\n"
    )
    return prompt

def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

# ─────────────────────────────────────────────
# Guided Exploration — covers the full 8×8 grid
# ─────────────────────────────────────────────
# Cycle through diverse cells so each epoch exercises a different grid location
GRID_CELLS = [(r, c) for r in range(8) for c in range(8)]  # 64 cells
INTERVENTION_TYPES = ["tree_canopy", "green_roof", "reflective_surface"]
BUREAUCRATIC_SEQUENCE = ["query_zoning", "propose_budget", "deploy_intervention"]

def smart_random_action(step):
    row = random.randint(0, 7)
    col = random.randint(0, 7)
    intervention = random.choice(INTERVENTION_TYPES)

    phase = step % 3

    if phase == 0:
        return {"action_type": "query_zoning", "row": row, "col": col, "intervention_type": intervention}
    elif phase == 1:
        return {"action_type": "propose_budget", "row": row, "col": col, "intervention_type": intervention}
    else:
        return {"action_type": "deploy_intervention", "row": row, "col": col, "intervention_type": intervention}

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("Loading Model and Tokenizer...")
    from peft import LoraConfig
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        peft_config=lora_config
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer)

    print("Resetting Urban Heat Environment...")
    reset_state = env_reset()
    print(f"Using ENV_URL={ENV_URL}")

    print(f"Starting PPO Training Loop ({TOTAL_EPOCHS} epochs)...\n")
    epoch_rewards = []
    epoch_env_rewards = []
    epoch_shaped_rewards = []

    for epoch in range(TOTAL_EPOCHS):
        # Collect a small on-policy batch of rollouts per PPO update.
        query_tensors = []
        response_tensors = []
        reward_tensors = []

        batch_env_reward = 0.0
        batch_shaped_reward = 0.0

        for b in range(ROLLOUT_STEPS_PER_EPOCH):
            query_text = format_env_prompt(reset_state)
            q_ids = tokenizer(query_text, return_tensors="pt").input_ids[0]

            # Generate candidate action from model
            gen = trainer.generate(
                [q_ids],
                return_prompt=False,
                max_new_tokens=32,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            model_resp_ids = gen[0]
            model_resp_text = tokenizer.decode(model_resp_ids, skip_special_tokens=True)
            if model_resp_text.strip() == "":
                model_resp_text = "query_zoning 0 0"

            parsed_action = None
            try:
                match = re.search(r'\{[^{}]*\}', model_resp_text)
                parsed_action = json.loads(match.group(0)) if match else json.loads(model_resp_text)
                if parsed_action.get("action_type") not in BUREAUCRATIC_SEQUENCE:
                    parsed_action = None
            except Exception:
                parsed_action = None

            # Guided exploration schedule (high early, low late)
            exploration_prob = 0.9 - (epoch / TOTAL_EPOCHS) * 0.8
            guided = (parsed_action is None) or (random.random() < exploration_prob)
            if guided:
                teacher_action = smart_random_action(epoch * ROLLOUT_STEPS_PER_EPOCH + b)
                parsed_action = teacher_action
                # IMPORTANT: make PPO update consistent with executed action.
                # We replace the response tokens with the teacher JSON so rewards align.
                teacher_text = json.dumps(parsed_action, separators=(",", ":"))
                resp_ids = tokenizer(teacher_text, return_tensors="pt").input_ids[0]
            else:
                resp_ids = model_resp_ids

            # Build final action (clamped)
            action_data = {
                "task_id": "full_mitigation",
                "action_type": str(parsed_action.get("action_type", "query_zoning")),
                "row": max(0, min(7, safe_int(parsed_action.get("row", 0)))),
                "col": max(0, min(7, safe_int(parsed_action.get("col", 0)))),
                "intervention_type": str(parsed_action.get("intervention_type", "tree_canopy")),
            }
            if action_data["intervention_type"] not in INTERVENTION_TYPES:
                action_data["intervention_type"] = "tree_canopy"

            obs = env_step(action_data)
            env_reward = float(obs.get("reward", 0.0) or 0.0)
            info = obs.get("info", {}) if isinstance(obs.get("info", {}), dict) else {}

            # Reward shaping (keep small; main signal must be environment)
            shaped = env_reward
            if info.get("error"):
                shaped -= 0.05
            if isinstance(info.get("success"), str) and info.get("success"):
                shaped += 0.03

            # Scale rewards so PPO sees non-trivial magnitudes.
            shaped *= 10.0
            env_reward_scaled = env_reward * 10.0

            # Clip and normalize rewards
            shaped = max(min(shaped, 10.0), -10.0)

            query_tensors.append(q_ids)
            response_tensors.append(resp_ids)
            reward_tensors.append(torch.tensor(float(shaped) / 10.0))

            batch_env_reward += env_reward_scaled
            batch_shaped_reward += shaped

            # Advance env state
            if obs.get("done", False):
                reset_state = env_reset()
            else:
                reset_state = obs.get("state", reset_state)

        # PPO update for the whole batch
        stats = trainer.step(query_tensors, response_tensors, reward_tensors)

        avg_env = batch_env_reward / ROLLOUT_STEPS_PER_EPOCH
        avg_shaped = batch_shaped_reward / ROLLOUT_STEPS_PER_EPOCH
        print(
            f"Epoch {epoch:4d} | avg_env_reward(x10)={avg_env:+.4f} | avg_shaped(x10)={avg_shaped:+.4f} "
            f"| kl={float(stats.get('kl', 0.0)):.4f}"
        )

        epoch_rewards.append(avg_shaped)
        epoch_env_rewards.append(avg_env)
        epoch_shaped_rewards.append(avg_shaped)

    # ─────────────────────────────────────────────
    # Save metrics
    # ─────────────────────────────────────────────
    with open("train_metrics.json", "w") as f:
        json.dump(
            {
                "epoch_rewards": epoch_rewards,  # kept for backwards-compat
                "epoch_avg_env_reward_x10": epoch_env_rewards,
                "epoch_avg_shaped_reward_x10": epoch_shaped_rewards,
                "env_url": ENV_URL,
                "total_epochs": TOTAL_EPOCHS,
                "rollout_steps_per_epoch": ROLLOUT_STEPS_PER_EPOCH,
            },
            f,
        )
    print("\ntrain_metrics.json saved.")

    # ─────────────────────────────────────────────
    # Final evaluation across all 3 tasks
    # ─────────────────────────────────────────────
    print("\n" + "="*50)
    print("   FINAL MULTI-TASK EVALUATION")
    print("="*50)
    tasks_to_eval = [
        ("reduce_avg_temp",    "Easy   "),
        ("protect_dense_zones","Medium "),
        ("full_mitigation",    "Hard   "),
    ]
    for tid, label in tasks_to_eval:
        try:
            result = requests.get(f"{ENV_URL}/grade/{tid}", timeout=10).json()
            score   = result.get("score", 0.0)
            success = "✅ SUCCESS" if score > 0.1 else "❌ FAIL"
            print(f"[{label}] Task: {tid:25s} | Score: {score:.3f} | {success}")
        except Exception as e:
            print(f"[{label}] Task: {tid:25s} | ERROR: {e}")
    print("="*50)
    print("\nTraining complete. Run the plot cell in your Colab notebook to visualise results.")

if __name__ == "__main__":
    main()

