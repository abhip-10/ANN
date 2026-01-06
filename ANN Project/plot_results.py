"""
Graphical Performance Analysis for RL Models
Generates four plots in ./plots/ from outputs/evaluation_summary.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

CSV_PATH = OUTPUTS_DIR / "evaluation_summary.csv"
if not CSV_PATH.exists():
    raise SystemExit(f"Missing {CSV_PATH}. Run evaluation first.")

# Load
df = pd.read_csv(CSV_PATH)
# Clean model names for plotting
df["model_short"] = df["model"].apply(lambda s: Path(s).stem)

# 1. Avg reward
plt.figure()
plt.bar(df["model_short"], df["avg_reward"], color="C0")
plt.xlabel("Model")
plt.ylabel("Average Episode Reward")
plt.title("Average Reward Comparison")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "avg_reward_comparison.png")
plt.close()

# 2. Std dev
plt.figure()
plt.bar(df["model_short"], df["std_reward"], color="C1")
plt.xlabel("Model")
plt.ylabel("Reward Standard Deviation")
plt.title("Reward Stability Comparison")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "reward_stability.png")
plt.close()

# 3. Avg episode length
plt.figure()
plt.bar(df["model_short"], df["avg_length"], color="C2")
plt.xlabel("Model")
plt.ylabel("Average Episode Length")
plt.title("Episode Length Comparison")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "avg_episode_length.png")
plt.close()

# 4. Collisions
plt.figure()
plt.bar(df["model_short"], df["collisions"], color="C3")
plt.xlabel("Model")
plt.ylabel("Number of Collisions")
plt.title("Collision Comparison")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "collision_comparison.png")
plt.close()

print("✅ All performance graphs generated in", PLOTS_DIR)
