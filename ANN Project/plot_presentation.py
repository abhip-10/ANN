import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(__file__)
CSV_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'evaluation_summary.csv')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def safe_read_csv(path):
    if not os.path.exists(path):
        print('CSV not found:', path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print('Failed to read CSV:', e)
        return pd.DataFrame()


def plot_avg_reward(df):
    out = os.path.join(PLOTS_DIR, 'avg_reward_comparison.png')
    plt.figure(figsize=(7,4))
    x = df['model']
    y = df['avg_reward']
    yerr = df['std_reward'] if 'std_reward' in df.columns else np.zeros_like(y)
    plt.bar(x, y, yerr=yerr, capsize=5, color=['#2b8cbe','#7bccc4','#a6bddb'])
    plt.ylabel('Average Reward')
    plt.title('Average Reward Comparison Across Models')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print('Wrote', out)


def plot_reward_stability(df):
    out = os.path.join(PLOTS_DIR, 'reward_stability.png')
    plt.figure(figsize=(7,4))
    x = df['model']
    y = df['std_reward'] if 'std_reward' in df.columns else np.zeros(len(df))
    plt.bar(x, y, color='#fdae61')
    plt.ylabel('Std Dev of Reward')
    plt.title('Reward Stability Analysis')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print('Wrote', out)


def plot_avg_length(df):
    out = os.path.join(PLOTS_DIR, 'avg_episode_length.png')
    plt.figure(figsize=(7,4))
    x = df['model']
    y = df['avg_length']
    plt.bar(x, y, color='#2ca25f')
    plt.ylabel('Average Episode Length')
    plt.title('Episode Length Comparison')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print('Wrote', out)


def plot_collisions(df):
    out = os.path.join(PLOTS_DIR, 'collision_comparison.png')
    plt.figure(figsize=(7,4))
    x = df['model']
    y = df['collisions']
    plt.bar(x, y, color='#de2d26')
    plt.ylabel('Collision Count')
    plt.title('Collision Analysis Across Models')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print('Wrote', out)


def plot_combined(df):
    out = os.path.join(PLOTS_DIR, 'comparison_4panel.png')
    fig, axes = plt.subplots(2,2, figsize=(12,8))
    # avg reward
    axes[0,0].bar(df['model'], df['avg_reward'], yerr=(df['std_reward'] if 'std_reward' in df.columns else np.zeros(len(df))), capsize=5)
    axes[0,0].set_title('Average Reward Comparison Across Models')
    # reward stability
    axes[0,1].bar(df['model'], (df['std_reward'] if 'std_reward' in df.columns else np.zeros(len(df))), color='#fdae61')
    axes[0,1].set_title('Reward Stability Analysis')
    # avg length
    axes[1,0].bar(df['model'], df['avg_length'], color='#2ca25f')
    axes[1,0].set_title('Episode Length Comparison')
    # collisions
    axes[1,1].bar(df['model'], df['collisions'], color='#de2d26')
    axes[1,1].set_title('Collision Analysis Across Models')
    for ax in axes.flatten():
        ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print('Wrote', out)


def main():
    df = safe_read_csv(CSV_PATH)
    if df.empty:
        print('No data available in CSV to plot. Exiting.')
        return
    # Ensure numeric columns
    for col in ['avg_reward','std_reward','avg_length','collisions']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # Reorder models sensibly if possible
    # Plot individual figures
    plot_avg_reward(df)
    plot_reward_stability(df)
    plot_avg_length(df)
    plot_collisions(df)
    plot_combined(df)
    print('All plots generated in', PLOTS_DIR)

if __name__ == '__main__':
    main()
