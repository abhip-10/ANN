import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env
import imageio
import matplotlib.pyplot as plt

ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 20,
    "duration": 40,
}


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_env(render_mode=None):
    if render_mode:
        return gym.make("highway-v0", config=ENV_CONFIG, render_mode=render_mode)
    return gym.make("highway-v0", config=ENV_CONFIG)


def load_model(path, state_dim, action_dim, device="cpu"):
    model = QNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def eval_model(model_path, episodes=10, record_gif=False, gif_out=None):
    env = make_env(render_mode="rgb_array" if record_gif else None)
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.n
    model = load_model(str(model_path), state_dim, action_dim, device="cpu")

    rewards = []
    lengths = []
    collisions = 0
    frames_for_gif = None

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        length = 0
        while not done:
            state = torch.FloatTensor(obs.flatten()).unsqueeze(0)
            with torch.no_grad():
                action = model(state).argmax().item()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            length += 1
            # collision heuristic: check common info keys
            if isinstance(info, dict):
                if info.get("crashed") or info.get("collision") or info.get("crash"):
                    collisions += 1
            # fallback heuristic: large negative reward implies collision
            if reward < -5:
                collisions += 1
        rewards.append(total_reward)
        lengths.append(length)

        if record_gif and ep == 0:
            # record one episode for GIF
            obs, _ = env.reset()
            frames = []
            done = False
            while not done:
                frames.append(env.render())
                state = torch.FloatTensor(obs.flatten()).unsqueeze(0)
                with torch.no_grad():
                    action = model(state).argmax().item()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            frames_for_gif = frames

    env.close()

    if record_gif and gif_out and frames_for_gif:
        Path(gif_out).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_out, frames_for_gif, fps=10)

    return {
        "model": model_path.name,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_length": float(np.mean(lengths)),
        "collisions": int(collisions),
        "episodes": episodes,
    }


def main():
    project = Path(__file__).resolve().parents[1]
    model_dir = project / "models"
    out_dir = project / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "evaluation_results.csv"
    gif_dir = project / "videos"
    gif_dir.mkdir(parents=True, exist_ok=True)

    models = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
    if not models:
        print("No .pth models found in models/ - nothing to do.")
        return

    results = []
    for m in models:
        print("Evaluating", m.name)
        gif_path = str(gif_dir / (m.stem + "_eval.gif"))
        res = eval_model(m, episodes=10, record_gif=True, gif_out=gif_path)
        results.append(res)

    # write CSV
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "episodes", "avg_reward", "std_reward", "avg_length", "collisions"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # simple bar chart for avg_reward
    models_names = [r["model"] for r in results]
    avg_rewards = [r["avg_reward"] for r in results]

    plt.figure(figsize=(8, 4))
    plt.bar(models_names, avg_rewards, color="C0")
    plt.ylabel("Avg Reward")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_reward_comparison.png")
    plt.close()

    print("Done. Results saved to:", results_csv)


if __name__ == "__main__":
    main()
