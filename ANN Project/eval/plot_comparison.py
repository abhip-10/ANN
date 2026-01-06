import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CSV = PROJECT / "outputs" / "evaluation_summary.csv"
OUT = PROJECT / "outputs" / "comparison.png"

if not CSV.exists():
    print("evaluation_summary.csv not found. Run evaluation first.")
    raise SystemExit(1)

df = pd.read_csv(CSV)
if df.empty:
    print("No results in CSV.")
    raise SystemExit(1)

# normalize model names
df['model_short'] = df['model'].apply(lambda s: Path(s).stem)

fig, ax = plt.subplots(1,2, figsize=(10,4))
df.plot.bar(x='model_short', y='avg_reward', ax=ax[0], legend=False, color='C0')
ax[0].set_title('Average Reward')
ax[0].set_ylabel('Avg Reward')

df.plot.bar(x='model_short', y='collisions', ax=ax[1], legend=False, color='C1')
ax[1].set_title('Collisions')
ax[1].set_ylabel('Count')

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200)
print('Saved comparison plot to', OUT)
