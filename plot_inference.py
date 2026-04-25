import json
import matplotlib.pyplot as plt
import os

def plot_inference_metrics():
    if not os.path.exists("inference_metrics.json"):
        print("Error: inference_metrics.json not found. Run inference.py first.")
        return

    with open("inference_metrics.json", "r") as f:
        metrics = json.load(f)

    if not metrics:
        print("Error: No data found in inference_metrics.json")
        return

    plt.figure(figsize=(10, 6))
    
    # We will plot the first task's step rewards
    task_id = list(metrics.keys())[0]
    rewards = metrics[task_id]

    plt.plot(range(1, len(rewards) + 1), rewards, marker='o', linestyle='-', color='r')
    plt.title(f"Baseline (72B) Model Inference Step Rewards - {task_id}")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=1) # Zero line

    # Save and show the plot
    plt.tight_layout()
    plt.savefig("inference_plot.png")
    print("Saved inference_plot.png")
    
if __name__ == "__main__":
    plot_inference_metrics()
