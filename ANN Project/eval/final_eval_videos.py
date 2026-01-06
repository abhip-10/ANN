"""
Generate final combined evaluation videos per model:
- eval_dqn.mp4       (SB3 .zip)
- eval_ddqn.mp4      (double_dqn .pth)
- eval_dueling.mp4   (dueling .pth)

This script runs 5 episodes per model and writes one combined MP4 per model
into `videos_mp4/`.

Run with venv activated:
    .\.venv\Scripts\Activate.ps1
    python eval\final_eval_videos.py
"""
from pathlib import Path
import imageio
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env
import numpy as np
import os

# Try to import stable-baselines3
try:
    from stable_baselines3 import DQN
    HAS_SB3 = True
except Exception:
    HAS_SB3 = False

PROJECT = Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"
VIDEOS = PROJECT / "videos"
OUT = PROJECT / "videos_mp4"
VIDEOS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

ENV_CONFIG = {
    "observation": {"type": "Kinematics", "vehicles_count": 10, "features": ["presence", "x", "y", "vx", "vy"], "absolute": False, "normalize": True},
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 20,
    "duration": 40,
}

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


def make_env():
    return gym.make("highway-v0", config=ENV_CONFIG, render_mode="rgb_array")


def load_pytorch_policy(path, state_dim, action_dim, device="cpu"):
    raw = torch.load(path, map_location=device)
    sd = None
    if isinstance(raw, dict):
        for key in ("policy_net", "policy", "policy_state_dict", "model_state_dict", "state_dict", "agent_state_dict", "q_net"):
            if key in raw and isinstance(raw[key], dict):
                sd = raw[key]
                break
    if sd is None:
        if isinstance(raw, dict) and any(isinstance(v, torch.Tensor) for v in raw.values()):
            sd = raw
    if sd is None:
        raise RuntimeError("No policy state_dict found in checkpoint")

    # strip module. prefix
    normalized = {}
    for k,v in sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        normalized[nk] = v
    sd = normalized

    # try to infer hidden sizes
    import re
    layer_shapes = {}
    for k,v in sd.items():
        m = re.search(r"net\.(\d+)\.weight$", k)
        if m and isinstance(v, torch.Tensor):
            layer_shapes[int(m.group(1))] = tuple(v.shape)
    hid1 = 256; hid2 = 256
    if 0 in layer_shapes:
        hid1 = layer_shapes[0][0]
    if 2 in layer_shapes:
        hid2 = layer_shapes[2][0]

    model = QNetwork(state_dim, action_dim, hidden1=hid1, hidden2=hid2)
    model.load_state_dict(sd, strict=False)
    model.to(device); model.eval()
    return model


def record_and_combine(model_type, model_path, out_name, episodes=5):
    env = make_env()
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.n

    # load model
    if model_type == "sb3":
        if not HAS_SB3:
            print("SB3 not installed; skipping DQN")
            return False
        # Load SB3 model without attaching env to avoid obs-space mismatch
        model = DQN.load(str(model_path))
    else:
        model = load_pytorch_policy(model_path, state_dim, action_dim)

    combined_frames = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            frame = env.render()
            combined_frames.append(frame)
            if model_type == "sb3":
                # SB3 model was likely trained on flattened observations
                obs_flat = obs.flatten()
                action, _ = model.predict(obs_flat, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(int(action))
            else:
                state = torch.FloatTensor(obs.flatten()).unsqueeze(0)
                with torch.no_grad():
                    action = int(model(state).argmax().item())
                obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()

    out_path = OUT / (out_name + ".mp4")
    imageio.mimsave(str(out_path), combined_frames, fps=10)
    print(f"Wrote {out_path}")
    return True


def main():
    # mapping: (type, filename, outname)
    # prefer files in MODELS dir; fall back to project root if present there
    candidates = [
        ("sb3", "DQN_highway_10k.zip", "eval_dqn"),
        ("pth", "double_dqn_highway.pth", "eval_ddqn"),
        # prefer extracted policy if available
        ("pth", "dueling_policy_extracted.pth", "eval_dueling"),
        ("pth", "dueling_double_dqn_highway.pth", "eval_dueling"),
    ]
    mapping = []
    for typ, name, outname in candidates:
        path_in_models = MODELS / name
        path_in_root = PROJECT / name
        if path_in_models.exists():
            mapping.append((typ, path_in_models, outname))
        elif path_in_root.exists():
            mapping.append((typ, path_in_root, outname))
        else:
            # still append the models path (so script reports missing if neither exists)
            mapping.append((typ, path_in_models, outname))

    # Clean previous outputs
    for p in (PROJECT / "videos", OUT, PROJECT / "videos_mp4"):
        # keep folders but remove files
        if p.exists():
            for f in p.glob("*"):
                try:
                    if f.is_file():
                        f.unlink()
                except Exception:
                    pass

    for typ, path, outname in mapping:
        if not path.exists():
            print(f"Model not found, skipping: {path.name}")
            continue
        print(f"Evaluating {path.name} -> {outname}.mp4")
        try:
            ok = record_and_combine("sb3" if typ=="sb3" else "pth", path, outname, episodes=5)
        except Exception as e:
            print("Failed to evaluate", path.name, e)

if __name__ == '__main__':
    main()
