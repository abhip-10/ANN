import sys
from pathlib import Path
proj = Path(__file__).resolve().parent
sys.path.insert(0, str(proj))
import evaluate_all
from pathlib import Path
import numpy as _np
import torch
import torch.nn as nn

MODEL = Path('models') / 'dueling_double_dqn_highway.pth'
if not MODEL.exists():
    print('Model not found:', MODEL)
    sys.exit(1)

# Load the training checkpoint safely
try:
    torch.serialization.add_safe_globals([_np.dtype, _np._core.multiarray.scalar])
except Exception:
    pass
raw = torch.load(str(MODEL), map_location='cpu', weights_only=False)

# Extract state_dict
sd = raw.get('model_state_dict', None)
if sd is None:
    print('No model_state_dict found in checkpoint')
    sys.exit(1)

# Build model with EXACT training architecture
class DuelingQNetwork(nn.Module):
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

state_dim = 50  # 10 vehicles * 5 features
action_dim = 5  # DiscreteMetaAction

model = DuelingQNetwork(state_dim, action_dim)
model.load_state_dict(sd)
model.eval()
print('Loaded model successfully with correct architecture')

# Evaluate
env = evaluate_all.make_env(render_mode='rgb_array')
import imageio
import numpy as np

rewards = []
lengths = []
collisions = 0

for ep in range(10):
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
        if isinstance(info, dict) and any(k in info and info[k] for k in ("crashed", "collision", "crash")):
            collisions += 1
        if reward < -5:
            collisions += 1
    # save gif
    gif_path = proj / 'videos' / f'dueling_episode_{ep+1}.gif'
    (proj / 'videos').mkdir(exist_ok=True)
    imageio.mimsave(str(gif_path), frames, fps=10)
    rewards.append(total_reward)
    lengths.append(length)
    print(f'[Dueling DDQN] Episode {ep+1} Reward: {total_reward:.2f}')

avg_reward = float(np.mean(rewards))
std_reward = float(np.std(rewards))
avg_length = float(np.mean(lengths))
print(f'[Dueling DDQN] Average Reward: {avg_reward:.2f} (std: {std_reward:.2f})')

# Write CSV
out = proj / 'outputs' / 'evaluation_summary_dueling_only.csv'
out.parent.mkdir(parents=True, exist_ok=True)
import csv
with open(out, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['model','episodes','avg_reward','std_reward','avg_length','collisions'])
    writer.writeheader()
    writer.writerow({
        'model': 'dueling_double_dqn_highway.pth',
        'episodes': 10,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'collisions': collisions
    })
print('Wrote', out)
env.close()
