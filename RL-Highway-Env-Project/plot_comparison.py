"""
Generate 4 mandatory comparison graphs for DDPG, A3C, PPO models.
Reads per-episode rewards from results/ and produces presentation-ready plots.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Model display names and colors
MODELS = ['DDPG', 'A3C', 'PPO']
COLORS = {'DDPG': '#e41a1c', 'A3C': '#377eb8', 'PPO': '#4daf4a'}

def load_rewards():
    """Load per-episode rewards for each model."""
    data = {}
    for model in MODELS:
        path = os.path.join(RESULTS_DIR, f'{model}_rewards.csv')
        if os.path.exists(path):
            rewards = np.loadtxt(path, delimiter=',')
            data[model] = rewards
            print(f'{model}: {len(rewards)} episodes, mean={rewards.mean():.2f}, std={rewards.std():.2f}')
        else:
            print(f'Warning: {path} not found')
    return data


def plot_avg_reward(data):
    """Bar chart of average reward per model with error bars."""
    out = os.path.join(PLOTS_DIR, 'avg_reward_comparison.png')
    plt.figure(figsize=(8, 5))
    
    models = list(data.keys())
    means = [data[m].mean() for m in models]
    stds = [data[m].std() for m in models]
    colors = [COLORS[m] for m in models]
    
    bars = plt.bar(models, means, yerr=stds, capsize=8, color=colors, edgecolor='black', linewidth=1.2)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Average Reward Comparison Across Models', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=10)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                 f'{mean:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Wrote {out}')


def plot_reward_stability(data):
    """Bar chart of standard deviation (lower = more stable)."""
    out = os.path.join(PLOTS_DIR, 'reward_stability.png')
    plt.figure(figsize=(8, 5))
    
    models = list(data.keys())
    stds = [data[m].std() for m in models]
    colors = [COLORS[m] for m in models]
    
    bars = plt.bar(models, stds, color=colors, edgecolor='black', linewidth=1.2)
    plt.ylabel('Std Dev of Reward (lower = more stable)', fontsize=12)
    plt.title('Reward Stability Analysis', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=10)
    
    # Add value labels
    for bar, std in zip(bars, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Wrote {out}')


def plot_reward_distribution(data):
    """Box plot showing reward distribution for each model."""
    out = os.path.join(PLOTS_DIR, 'reward_distribution.png')
    plt.figure(figsize=(8, 5))
    
    models = list(data.keys())
    rewards_list = [data[m] for m in models]
    
    bp = plt.boxplot(rewards_list, labels=models, patch_artist=True)
    for patch, model in zip(bp['boxes'], models):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.7)
    
    plt.ylabel('Total Reward per Episode', fontsize=12)
    plt.title('Reward Distribution Comparison', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Wrote {out}')


def plot_reward_over_episodes(data):
    """Line plot showing reward progression over evaluation episodes."""
    out = os.path.join(PLOTS_DIR, 'reward_vs_episode.png')
    plt.figure(figsize=(10, 5))
    
    for model in data.keys():
        rewards = data[model]
        episodes = np.arange(1, len(rewards) + 1)
        plt.plot(episodes, rewards, label=model, color=COLORS[model], alpha=0.7, linewidth=1.5)
        # Smoothed line (rolling mean)
        if len(rewards) >= 5:
            smoothed = pd.Series(rewards).rolling(window=5, min_periods=1).mean()
            plt.plot(episodes, smoothed, color=COLORS[model], linewidth=2.5, linestyle='--')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Reward vs Episode (solid=raw, dashed=smoothed)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Wrote {out}')


def plot_min_max_comparison(data):
    """Bar chart showing min and max rewards (range) per model."""
    out = os.path.join(PLOTS_DIR, 'min_max_comparison.png')
    plt.figure(figsize=(8, 5))
    
    models = list(data.keys())
    mins = [data[m].min() for m in models]
    maxs = [data[m].max() for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, mins, width, label='Min Reward', color='#fee0d2', edgecolor='black')
    plt.bar(x + width/2, maxs, width, label='Max Reward', color='#31a354', edgecolor='black')
    
    plt.ylabel('Reward', fontsize=12)
    plt.title('Min/Max Reward Range per Model', fontsize=14, fontweight='bold')
    plt.xticks(x, models, fontsize=11)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Wrote {out}')


def plot_combined_4panel(data):
    """Combined 2x2 figure with all 4 key metrics."""
    out = os.path.join(PLOTS_DIR, 'comparison_4panel.png')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = list(data.keys())
    means = [data[m].mean() for m in models]
    stds = [data[m].std() for m in models]
    mins = [data[m].min() for m in models]
    maxs = [data[m].max() for m in models]
    colors = [COLORS[m] for m in models]
    
    # 1. Average Reward
    axes[0, 0].bar(models, means, yerr=stds, capsize=5, color=colors, edgecolor='black')
    axes[0, 0].set_title('Average Reward Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Avg Reward')
    
    # 2. Reward Stability (Std Dev)
    axes[0, 1].bar(models, stds, color=colors, edgecolor='black')
    axes[0, 1].set_title('Reward Stability (Std Dev)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Std Dev (lower=better)')
    
    # 3. Box plot distribution
    bp = axes[1, 0].boxplot([data[m] for m in models], labels=models, patch_artist=True)
    for patch, model in zip(bp['boxes'], models):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.7)
    axes[1, 0].set_title('Reward Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Total Reward')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Reward over episodes (smoothed)
    for model in models:
        rewards = data[model]
        episodes = np.arange(1, len(rewards) + 1)
        smoothed = pd.Series(rewards).rolling(window=5, min_periods=1).mean()
        axes[1, 1].plot(episodes, smoothed, label=model, color=COLORS[model], linewidth=2)
    axes[1, 1].set_title('Reward vs Episode (smoothed)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Total Reward')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('RL Model Comparison: DDPG vs A3C vs PPO', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out}')


def write_summary_table(data):
    """Write a clean summary CSV with all metrics."""
    out = os.path.join(PLOTS_DIR, 'evaluation_summary.csv')
    rows = []
    for model in data.keys():
        rewards = data[model]
        rows.append({
            'model': model,
            'episodes': len(rewards),
            'avg_reward': rewards.mean(),
            'std_reward': rewards.std(),
            'min_reward': rewards.min(),
            'max_reward': rewards.max(),
            'median_reward': np.median(rewards)
        })
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f'Wrote {out}')
    print('\n=== SUMMARY TABLE ===')
    print(df.to_string(index=False))
    return df


def main():
    print('Loading reward data...')
    data = load_rewards()
    
    if not data:
        print('No reward data found. Run evaluate_models.py first.')
        return
    
    print('\nGenerating plots...')
    plot_avg_reward(data)
    plot_reward_stability(data)
    plot_reward_distribution(data)
    plot_reward_over_episodes(data)
    plot_min_max_comparison(data)
    plot_combined_4panel(data)
    
    print('\nWriting summary...')
    write_summary_table(data)
    
    print(f'\n✅ All plots saved to {PLOTS_DIR}/')


if __name__ == '__main__':
    main()
