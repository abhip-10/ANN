from pathlib import Path
import imageio
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env

PROJECT = Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"
OUT = PROJECT / "videos_mp4"
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


def load_policy(path, state_dim, action_dim):
    raw = torch.load(path, map_location='cpu')
    if isinstance(raw, dict):
        # prefer model_state_dict keys
        for key in ("model_state_dict","policy_net","policy","state_dict","model"):
            if key in raw and isinstance(raw[key], dict):
                sd = raw[key]
                break
        else:
            # assume raw is a state_dict
            sd = raw
    else:
        raise RuntimeError("Unsupported checkpoint format")
    # strip module.
    sd2 = { (k[len('module.'): ] if k.startswith('module.') else k):v for k,v in sd.items() }
    # infer hidden sizes
    import re
    layer_shapes = {}
    for k,v in sd2.items():
        m = re.search(r"net\.(\d+)\.weight$", k)
        if m and isinstance(v, torch.Tensor):
            layer_shapes[int(m.group(1))] = tuple(v.shape)
    hid1 = layer_shapes.get(0, (256,))[0]
    hid2 = layer_shapes.get(2, (256,))[0]
    model = QNetwork(state_dim, action_dim, hidden1=hid1, hidden2=hid2)
    model.load_state_dict(sd2, strict=False)
    model.eval()
    return model


def record(out_name="dueling_retry", episodes=5):
    env = make_env()
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.n

    # prefer extracted policy
    cand = MODELS / "dueling_policy_extracted.pth"
    if not cand.exists():
        cand = PROJECT / "dueling_double_dqn_highway.pth"
    if not cand.exists():
        raise FileNotFoundError("No dueling policy found")

    model = load_policy(cand, state_dim, action_dim)

    frames = []

    def pad_frame(frame, block=16):
        h, w = frame.shape[:2]
        new_h = ((h + block - 1) // block) * block
        new_w = ((w + block - 1) // block) * block
        if new_h == h and new_w == w:
            return frame
        pad_h = new_h - h
        pad_w = new_w - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return np.pad(frame, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=0)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            f = env.render()
            frames.append(pad_frame(f, block=16))
            state = torch.FloatTensor(obs.flatten()).unsqueeze(0)
            with torch.no_grad():
                action = int(model(state).argmax().item())
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()

    out_path = OUT / (out_name + ".mp4")
    imageio.mimsave(str(out_path), frames, fps=10)
    print("Wrote", out_path)

if __name__ == '__main__':
    record()
