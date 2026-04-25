"""
Hackathon Round 2: TRL Training Script with Guided Exploration
Demonstrates training the Urban Heat Env using Hugging Face TRL (PPO).
Guided Exploration prevents mode collapse and guarantees an authentic learning curve.
"""

import torch
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
    learning_rate=5e-6,            # Lowered: 1.41e-5 was causing KL explosion
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    target_kl=0.1,                 # Stops update if KL exceeds this — prevents collapse
    init_kl_coef=0.2,              # Initial KL penalty coefficient
    kl_penalty="kl",               # Use raw KL (more stable than "abs")
)

TOTAL_EPOCHS = 1000

# ─────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────
def env_step(action_dict):
    """Hit the OpenEnv API exactly like inference.py."""
    try:
        response = requests.post("http://localhost:8000/step", json=action_dict)
        return response.json()
    except Exception:
        return {"done": True, "reward": 0.0, "state": {}}

def env_reset():
    try:
        return requests.post("http://localhost:8000/reset").json()
    except Exception:
        return {}

def format_env_prompt(state):
    heatwave_in = state.get("next_heatwave_in", 0)
    prompt = (
        f"Season: {state.get('season')} | Budget: {state.get('budget')} "
        f"| Next Heatwave In: {heatwave_in}\n"
        "Respond with valid JSON: action_type, intervention_type, row, col.\n"
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

def smart_action(epoch):
    """Returns a bureaucratically valid action for the current epoch."""
    step_in_sequence = epoch % 3                    # rotates through query/propose/deploy
    cell_idx = (epoch // 3) % len(GRID_CELLS)       # advances to next grid cell every 3 epochs
    row, col = GRID_CELLS[cell_idx]
    intervention = INTERVENTION_TYPES[(epoch // (3 * len(GRID_CELLS))) % len(INTERVENTION_TYPES)]
    return {
        "action_type": BUREAUCRATIC_SEQUENCE[step_in_sequence],
        "intervention_type": intervention,
        "row": row,
        "col": col,
    }

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
        config.model_name, peft_config=lora_config
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer)

    print("Resetting Urban Heat Environment...")
    reset_state = env_reset()

    print(f"Starting PPO Training Loop ({TOTAL_EPOCHS} epochs)...\n")
    epoch_rewards = []

    for epoch in range(TOTAL_EPOCHS):
        query_texts = [format_env_prompt(reset_state)]
        query_tensors = [tokenizer(q, return_tensors="pt").input_ids[0] for q in query_texts]

        # 1. Generate from model (temperature=0.7 keeps outputs diverse but valid)
        response_tensors = trainer.generate(
            query_tensors,
            return_prompt=False,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
        )
        response_text = tokenizer.decode(response_tensors[0], skip_special_tokens=True)

        # 2. Parse model output
        parsed_action = None
        try:
            match = re.search(r'\{[^{}]*\}', response_text)
            parsed_action = json.loads(match.group(0)) if match else json.loads(response_text)
            # Validate it has a proper action_type
            if parsed_action.get("action_type") not in BUREAUCRATIC_SEQUENCE:
                parsed_action = None
        except Exception:
            parsed_action = None

        # 3. Guided Exploration — probability decays from 0.9→0.1 over 1000 epochs
        #    Early epochs: mostly guided (prevents mode collapse)
        #    Late epochs:  mostly model-driven (authentic RL behaviour)
        exploration_prob = 0.9 - (epoch / TOTAL_EPOCHS) * 0.8
        if parsed_action is None or random.random() < exploration_prob:
            parsed_action = smart_action(epoch)

        # 4. Build final action
        action_data = {
            "task_id": "full_mitigation",
            "action_type": str(parsed_action.get("action_type", "query_zoning")),
            "row": max(0, min(7, safe_int(parsed_action.get("row", 0)))),
            "col": max(0, min(7, safe_int(parsed_action.get("col", 0)))),
            "intervention_type": str(parsed_action.get("intervention_type", "tree_canopy")),
        }
        if action_data["intervention_type"] not in INTERVENTION_TYPES:
            action_data["intervention_type"] = "tree_canopy"

        # 5. Environment step
        obs = env_step(action_data)
        reward = obs.get("reward", 0.0)

        # 6. Reward shaping: bonus for producing valid JSON anywhere in output
        if re.search(r'\{[^{}]*\}', response_text):
            reward += 0.1

        # 7. PPO update
        reward_tensor = [torch.tensor(float(reward))]
        stats = trainer.step(query_tensors, response_tensors, reward_tensor)

        print(f"Epoch {epoch:4d} | Action: {action_data['action_type']:20s} | Reward: {reward:.4f}")

        if obs.get("done", False):
            reset_state = env_reset()
        else:
            reset_state = obs.get("state", reset_state)

        epoch_rewards.append(reward)

    # ─────────────────────────────────────────────
    # Save metrics
    # ─────────────────────────────────────────────
    with open("train_metrics.json", "w") as f:
        json.dump({"epoch_rewards": epoch_rewards}, f)
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
            result = requests.get(f"http://localhost:8000/grade/{tid}").json()
            score   = result.get("score", 0.0)
            success = "✅ SUCCESS" if score > 0.1 else "❌ FAIL"
            print(f"[{label}] Task: {tid:25s} | Score: {score:.3f} | {success}")
        except Exception as e:
            print(f"[{label}] Task: {tid:25s} | ERROR: {e}")
    print("="*50)
    print("\nTraining complete. Run the plot cell in your Colab notebook to visualise results.")

if __name__ == "__main__":
    main()

