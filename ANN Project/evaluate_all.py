"""
Unified evaluator: loads SB3 DQN (.zip) and custom PyTorch .pth models,
runs evaluation episodes, saves per-episode GIFs and a CSV summary.

Usage:
    .venv\Scripts\Activate.ps1
    python evaluate_all.py
"""

import os
import glob
import csv
from pathlib import Path
import imageio
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env
import argparse

# Try to import stable-baselines3 if available
HAS_SB3 = False
try:
    from stable_baselines3 import DQN
    HAS_SB3 = True
except Exception:
    HAS_SB3 = False

# -------------------------------
# ENV CONFIG (MUST MATCH TRAINING)
# -------------------------------
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

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
VIDEOS_DIR = PROJECT_ROOT / "videos"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VIDEOS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# -------------------------------
# Q-NETWORK (custom PyTorch)
# -------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1=256, hidden2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DuelingNetwork(nn.Module):
    """Dueling Q-Network matching training architecture exactly."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

# -------------------------------
# Helpers
# -------------------------------

def make_env(render_mode="rgb_array"):
    return gym.make("highway-v0", config=ENV_CONFIG, render_mode=render_mode)


def find_sb3_model():
    # common names: *.zip
    candidates = list(MODELS_DIR.glob("*.zip"))
    return candidates[0] if candidates else None


def find_pytorch_models():
    # return .pth/.pt but filter optimizer/state dict files heuristically
    files = list(MODELS_DIR.glob("*.pth")) + list(MODELS_DIR.glob("*.pt"))
    filtered = []
    for f in files:
        name = f.name.lower()
        if "optim" in name or "optimizer" in name or "state" in name:
            continue
        filtered.append(f)
    return filtered


def load_pytorch_model(path, state_dim, action_dim, device="cpu"):
    import numpy as _np
    # Add safe globals to allow loading checkpoints with numpy types
    try:
        torch.serialization.add_safe_globals([_np.dtype, _np._core.multiarray.scalar])
    except Exception:
        pass
    raw = torch.load(path, map_location=device, weights_only=False)

    # Extract a candidate state_dict from common checkpoint formats
    sd = None
    if isinstance(raw, dict):
        # prefer explicit policy keys
        for key in ("policy_net", "policy", "policy_state_dict", "model_state_dict", "state_dict", "agent_state_dict", "q_net"):
            if key in raw and isinstance(raw[key], dict):
                sd = raw[key]
                break
    if sd is None:
        # raw itself might already be a state_dict (mapping to Tensors)
        if isinstance(raw, dict) and any(isinstance(v, torch.Tensor) for v in raw.values()):
            sd = raw
        else:
            # search nested dicts for a tensor-mapped dict
            def find_tensor_dict(d):
                if isinstance(d, dict):
                    if any(isinstance(v, torch.Tensor) for v in d.values()):
                        return d
                    for v in d.values():
                        res = find_tensor_dict(v)
                        if res is not None:
                            return res
                return None
            sd = find_tensor_dict(raw) or raw

    # normalize common prefix like 'module.'
    normalized = {}
    for k, v in sd.items():
        newk = k
        if newk.startswith("module."):
            newk = newk[len("module."):]
        normalized[newk] = v
    sd = normalized

    # detect dueling-style checkpoint by keys
    dueling_keys = [k for k in sd.keys() if k.startswith("feature.") or k.startswith("value.") or k.startswith("advantage.")]
    if dueling_keys:
        # Build DuelingNetwork with fixed architecture matching training code
        model = DuelingNetwork(state_dim, action_dim)
        try:
            model.load_state_dict(sd)
        except Exception:
            # remap keys by suffix if needed
            tgt_keys = list(model.state_dict().keys())
            remap = {}
            for k, v in sd.items():
                for tgt in tgt_keys:
                    if k.endswith(tgt):
                        remap[tgt] = v
                        break
            model.load_state_dict(remap, strict=False)
        model.to(device)
        model.eval()
        return model

    # Try loading directly into default QNetwork, otherwise attempt smart remapping by suffixes
    # First, infer hidden sizes from checkpoint weights (look for net.<idx>.weight or q_net.<idx>.weight patterns)
    import re
    layer_shapes = {}
    for k, v in sd.items():
        # Match patterns like: net.0.weight, q_net.0.weight, q_net.q_net.0.weight
        m = re.search(r"(?:q_)?net\.(\d+)\.weight$", k)
        if m and isinstance(v, torch.Tensor):
            idx = int(m.group(1))
            if idx not in layer_shapes:  # prefer first match (online net, not target)
                layer_shapes[idx] = tuple(v.shape)

    # Expecting linear layers at indices 0,2,4 for MLP architecture
    if 0 in layer_shapes and 2 in layer_shapes and 4 in layer_shapes:
        hid1 = layer_shapes[0][0]
        hid2 = layer_shapes[2][0]
        print(f"Inferred architecture: {state_dim} -> {hid1} -> {hid2} -> {action_dim}")
    else:
        hid1, hid2 = 256, 256  # fallback defaults

    model = QNetwork(state_dim, action_dim, hidden1=hid1, hidden2=hid2)
    target_keys = list(model.state_dict().keys())

    # Map checkpoint keys to model keys by suffix matching
    mapped = {}
    for k, v in sd.items():
        # Skip target network weights
        if 'target' in k.lower():
            continue
        for tgt in target_keys:
            if k.endswith(tgt):
                mapped[tgt] = v
                break

    if set(target_keys).issubset(set(mapped.keys())):
        model.load_state_dict(mapped)
        model.to(device)
        model.eval()
        return model

    # Try direct load
    try:
        model.load_state_dict(sd)
        model.to(device)
        model.eval()
        return model
    except RuntimeError as e:
        # Final fallback - raise informative error
        raise RuntimeError(f"Failed to load model: {e}")


def evaluate_pytorch(path, env, episodes=5, out_prefix=None):
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.n
    model = load_pytorch_model(path, state_dim, action_dim, device="cpu")

    rewards = []
    lengths = []
    collisions = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        length = 0
        frames = []
        while not done:
            frames.append(env.render())
            state = torch.FloatTensor(obs.flatten()).unsqueeze(0)
            with torch.no_grad():
                action = int(model(state).argmax().item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            length += 1
            # heuristic collision count
            if isinstance(info, dict) and any(k in info and info[k] for k in ("crashed", "collision", "crash")):
                collisions += 1
            if reward < -5:
                collisions += 1
        # save gif
        if out_prefix:
            gif_path = VIDEOS_DIR / f"{out_prefix}_episode_{ep+1}.gif"
            imageio.mimsave(str(gif_path), frames, fps=10)
        rewards.append(total_reward)
        lengths.append(length)
        print(f"[{path.name}] Episode {ep+1} Reward: {total_reward:.2f}")

    summary = {
        "model": path.name,
        "episodes": episodes,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_length": float(np.mean(lengths)),
        "collisions": int(collisions),
    }
    print(f"[{path.name}] Average Reward: {summary['avg_reward']:.2f}\n")
    return summary


def evaluate_sb3_model(path, env, episodes=5, out_prefix=None):
    if not HAS_SB3:
        raise RuntimeError("stable-baselines3 not installed in the environment")
    # Load SB3 model defensively: replace problematic pickled objects like schedules
    model = None
    try:
        model = DQN.load(str(path), device="cpu")
    except Exception:
        try:
            model = DQN.load(str(path), device="cpu", custom_objects={"exploration_schedule": None})
        except Exception as e:
            try:
                # broader fallback: neutralize some common serialized callables
                model = DQN.load(str(path), device="cpu", custom_objects={"exploration_schedule": None, "learning_rate": 0.0})
            except Exception as e2:
                raise RuntimeError(f"Failed to load SB3 model {path.name}: {e2}")
    rewards = []
    lengths = []
    collisions = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        length = 0
        frames = []
        while not done:
            frames.append(env.render())
            # SB3 models sometimes expect flattened observations (1D). Flatten to match.
            flat_obs = np.asarray(obs).ravel()
            action, _ = model.predict(flat_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += float(reward)
            length += 1
            if isinstance(info, dict) and any(k in info and info[k] for k in ("crashed", "collision", "crash")):
                collisions += 1
            if reward < -5:
                collisions += 1
        if out_prefix:
            gif_path = VIDEOS_DIR / f"{out_prefix}_episode_{ep+1}.gif"
            imageio.mimsave(str(gif_path), frames, fps=10)
        rewards.append(total_reward)
        lengths.append(length)
        print(f"[{path.name}] Episode {ep+1} Reward: {total_reward:.2f}")

    summary = {
        "model": path.name,
        "episodes": episodes,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_length": float(np.mean(lengths)),
        "collisions": int(collisions),
    }
    print(f"[{path.name}] Average Reward: {summary['avg_reward']:.2f}\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes per model")
    args = parser.parse_args()

    env = make_env(render_mode="rgb_array")

    results = []

    # SB3
    sb3 = find_sb3_model()
    if sb3 and HAS_SB3:
        print("Evaluating SB3 model:", sb3.name)
        try:
            res = evaluate_sb3_model(sb3, env, episodes=args.episodes, out_prefix=sb3.stem)
            results.append(res)
        except Exception as e:
            print("SB3 evaluation failed:", e)
    elif sb3:
        print("Found SB3 model", sb3.name, "but stable-baselines3 is not installed in the venv.")

    # PyTorch models
    pths = find_pytorch_models()
    for p in pths:
        print("Evaluating PyTorch model:", p.name)
        try:
            res = evaluate_pytorch(p, env, episodes=args.episodes, out_prefix=p.stem)
            results.append(res)
        except RuntimeError as e:
            print(f"Skipping {p.name}: failed to load as model (likely optimizer checkpoint): {e}")
        except Exception as e:
            print(f"Error evaluating {p.name}: {e}")

    # write CSV summary
    csv_path = OUTPUTS_DIR / "evaluation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "episodes", "avg_reward", "std_reward", "avg_length", "collisions"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    env.close()
    print("Evaluation finished. Summary:", csv_path)

if __name__ == "__main__":
    main()
