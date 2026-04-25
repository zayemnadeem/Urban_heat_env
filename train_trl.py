"""
Hackathon Round 2: Minimal TRL Training Script
This demonstrates how to train the Urban Heat Env using Hugging Face TRL (PPO).
Ideal for running in a Colab notebook with Unsloth or standard Hugging Face PEFT.
"""

import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# Import the local environment interaction logic
import requests
import json
import re

# Set up PPO Configuration
config = PPOConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct", # Use a smaller model for Colab
    learning_rate=1.41e-5,
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=4,
)

def env_step(action_dict):
    """Hits the local OpenEnv API exactly like inference.py"""
    try:
        response = requests.post("http://localhost:8000/step", json=action_dict)
        return response.json()
    except Exception as e:
        return {"done": True, "reward": 0.0, "state": {}}

def format_env_prompt(state):
    grid = state.get('grid', [])
    heatwave_in = state.get('next_heatwave_in', 0)
    prompt = f"Season: {state.get('season')} | Budget: {state.get('budget')} | Next Heatwave In: {heatwave_in}\n"
    prompt += "Respond with valid JSON: action_type, intervention_type, row, col.\n"
    return prompt

def main():
    print("Loading Model and Tokenizer...")
    # Wrap model with Value Head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer)

    # Initialize Environment
    print("Resetting Urban Heat Environment...")
    reset_state = requests.post("http://localhost:8000/reset").json()

    print("Starting PPO Training Loop...")
    for epoch in range(10): # Example epochs
        query_texts = [format_env_prompt(reset_state)]
        query_tensors = [tokenizer(q, return_tensors="pt").input_ids[0] for q in query_texts]

        # 1. Generate Action
        response_tensors = trainer.generate(query_tensors, return_prompt=False, max_new_tokens=100)
        response_text = tokenizer.decode(response_tensors[0], skip_special_tokens=True)

        # 2. Parse Action (Fallback to query if failed)
        try:
            match = re.search(r'\{[^{}]*\}', response_text)
            parsed_action = json.loads(match.group(0)) if match else json.loads(response_text)
        except:
            parsed_action = {"action_type": "query_zoning", "row": 0, "col": 0, "intervention_type": "tree_canopy"}

        action_data = {
            "task_id": "full_mitigation",
            "action_type": parsed_action.get("action_type", "query_zoning"),
            "row": int(parsed_action.get("row", 0)),
            "col": int(parsed_action.get("col", 0)),
            "intervention_type": parsed_action.get("intervention_type", "tree_canopy")
        }

        # 3. Environment Step
        obs = env_step(action_data)
        reward = obs.get("reward", 0.0)

        # 4. PPO Update
        reward_tensor = [torch.tensor(float(reward))]
        
        print(f"Epoch {epoch}: Action: {action_data['action_type']} -> Reward: {reward}")
        
        # PPO optimization step
        stats = trainer.step(query_tensors, response_tensors, reward_tensor)
        
        if obs.get("done", False):
            reset_state = requests.post("http://localhost:8000/reset").json()
        else:
            reset_state = obs.get("state", reset_state)

    print("Training Script Dry-run Completed successfully.")

if __name__ == "__main__":
    main()
    print("Minimal TRL script ready. Run via typical python runtime when server is up.")
