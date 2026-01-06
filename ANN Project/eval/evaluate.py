import os
import argparse
from pathlib import Path
import imageio
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env

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


def make_env(render_mode="rgb_array"):
    env = gym.make("highway-v0", config=ENV_CONFIG, render_mode=render_mode)
    return env


def load_model(model_path, state_dim, action_dim, device="cpu"):
    model = QNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate(model, env, episodes=10, device="cpu"):
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            state = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state).argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}")
    print("Average Reward:", np.mean(rewards))
    return rewards


def record_episode(model, env, out_path, device="cpu", fps=10):
    obs, _ = env.reset()
    frames = []
    done = False
    while not done:
        frame = env.render()
        frames.append(frame)
        state = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(state).argmax().item()
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    imageio.mimsave(out_path, frames, fps=fps)


def find_default_model(project_root: Path):
    p = project_root / "models"
    if not p.exists():
        return None
    for ext in ("*.pth", "*.pt"):
        found = list(p.glob(ext))
        if found:
            return found[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model .pth", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--gif", type=str, default=None, help="Output GIF path (optional)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_default_model(project_root)
    if not model_path or not model_path.exists():
        raise FileNotFoundError(f"Model not found. Searched: {model_path}")

    env = make_env(render_mode="rgb_array")
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.n

    model = load_model(str(model_path), state_dim, action_dim, device="cpu")

    print(f"Evaluating model: {model_path}")
    evaluate(model, env, episodes=args.episodes)

    if args.gif:
        out = Path(args.gif)
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Recording one episode to {out}")
        record_episode(model, env, str(out))


if __name__ == "__main__":
    main()
