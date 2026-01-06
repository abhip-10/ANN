# Interpreted Evaluation Report

## Key findings

Agent ranking by mean reward:
1. **PPO** — mean=122.91, std=33.25, n=50
2. **A3C** — mean=118.56, std=35.86, n=50
3. **DDPG** — mean=40.57, std=6.63, n=50

- **Summary:** PPO achieved the highest average total reward (122.91).
  The gap between PPO and A3C is 4.35 mean reward points.

## Plots
- v2_mean_bar.png — Mean ± Std with sample counts annotated
- v2_box_swarm.png — Boxplot with per-episode points (swarm)
- v2_smoothed.png — Smoothed episode reward curves (moving average)