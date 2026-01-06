import os
import yaml
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import argparse

# Compatibility shims for older highway-env / gym code
import gym

# Monitor fallback
if not hasattr(gym, 'wrappers') or not hasattr(gym.wrappers, 'Monitor'):
    try:
        from gymnasium.wrappers import RecordVideo as Monitor  # type: ignore
    except Exception:
        def Monitor(env, *args, **kwargs):
            return env
    if hasattr(gym, 'wrappers'):
        setattr(gym.wrappers, 'Monitor', Monitor)

# GoalEnv fallback
if not hasattr(gym, 'GoalEnv'):
    class GoalEnv:
        pass
    setattr(gym, 'GoalEnv', GoalEnv)

# Make gym seeding return RandomState for compatibility
try:
    from gym.utils import seeding as _seeding
    def _np_random_rs(seed=None):
        return np.random.RandomState(seed), seed
    _seeding.np_random = _np_random_rs
except Exception:
    pass

from racetrack_env import RaceTrackEnv
import tensorflow.keras as keras
from agent.A3C import A3CAgent
from agent.DDPG import DDPGAgent
from agent.PPO import PPO


def read_params(path="config/params.yaml"):
    with open(path, 'r') as f:
        params = yaml.safe_load(f)
    params['train'] = False
    params['save_video'] = False
    params['exp_dir'] = params.get('exp_dir', f"./logs/{params.get('exp_id','eval')}")
    try:
        os.makedirs(params['exp_dir'], exist_ok=True)
    except Exception:
        pass
    return params


def eval_agent(params, agent_name, n_episodes=10, results_dir="results"):
    """Evaluate one agent in the current process; returns (rewards_array, summary_dict)."""
    params = params.copy()
    params['agent'] = agent_name
    os.makedirs(results_dir, exist_ok=True)

    # Speed optimizations
    params['save_video'] = False
    params['render_agent'] = False
    params['show_trajectories'] = False
    params['offscreen_rendering'] = True
    params['real_time_rendering'] = False
    params['policy_frequency'] = params.get('policy_frequency', params.get('simulation_frequency', 15))

    env = RaceTrackEnv(params)
    rewards = []

    if agent_name == 'DDPG':
        # Ensure loader points to the DDPG model directory
        params['load_model'] = 'models/DDPG'
        agent = DDPGAgent(params)
        obs0 = env.reset()
        agent.initialize_networks(obs0)
        if params.get('ddpg_best', False):
            agent.load_best()
        else:
            agent.load_models()

        for ep in trange(n_episodes, desc=f"{agent_name}", leave=True):
            obs = env.reset()
            done = False
            total = 0.0
            while not done:
                action = agent.select_action(np.expand_dims(obs/255, axis=0), env, test_model=True)
                obs, reward, done, _ = env.step(action)
                total += reward
            rewards.append(total)

    else:
        if agent_name == 'PPO':
            params['load_model'] = params.get('load_model', 'models/PPO.model')
        elif agent_name == 'A3C':
            params['load_model'] = params.get('load_model', 'models/A3C.model')

        model = None
        try:
            model = keras.models.load_model(params['load_model'])
        except Exception:
            try:
                import tensorflow as tf
                model = tf.saved_model.load(params['load_model'])
            except Exception as e:
                raise RuntimeError(f"Failed to load model for {agent_name}: {e}")

        import tensorflow as _tf
        for ep in trange(n_episodes, desc=f"{agent_name}", leave=True):
            obs = env.reset()
            done = False
            total = 0.0
            while not done:
                a_in = _tf.convert_to_tensor(np.expand_dims(obs, axis=0), dtype=_tf.float32)
                try:
                    action = model(a_in)[0].numpy()
                except Exception:
                    out = model(a_in)
                    if isinstance(out, dict):
                        val = list(out.values())[0]
                        action = val.numpy()[0]
                    else:
                        try:
                            action = out.numpy()[0]
                        except Exception:
                            action = _tf.convert_to_tensor(out).numpy()[0]
                obs, reward, done, _ = env.step(action)
                total += reward
            rewards.append(total)

    rewards = np.array(rewards)
    np.savetxt(os.path.join(results_dir, f"{agent_name}_rewards.csv"), rewards, delimiter=',')

    plt.figure()
    plt.plot(rewards, marker='o')
    plt.title(f"{agent_name} Rewards per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"{agent_name}_reward_vs_episode.png"))
    plt.close()

    summary = {
        'agent': agent_name,
        'n_episodes': int(len(rewards)),
        'mean_reward': float(rewards.mean()),
        'std_reward': float(rewards.std()),
        'min_reward': float(rewards.min()),
        'max_reward': float(rewards.max())
    }

    return rewards, summary


def _proc_target(agent_name, params, n_episodes, results_dir, out_list):
    try:
        rewards, summary = eval_agent(params, agent_name, n_episodes=n_episodes, results_dir=results_dir)
        out_list.append(summary)
    except Exception as e:
        out_list.append({'agent': agent_name, 'error': str(e)})


def main():
    params = read_params()
    agents = ['PPO', 'A3C', 'DDPG']
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', type=int, default=int(os.environ.get('EVAL_EPISODES', 20)),
                        help='number of episodes per agent (default: 20)')
    parser.add_argument('--agents', type=str, default=None,
                        help="comma-separated list of agents to evaluate (e.g. 'PPO,A3C,DDPG'). Default: all")
    args = parser.parse_args()
    N_EPISODES = args.n_episodes
    if args.agents:
        agents = [s.strip() for s in args.agents.split(',')]

    # If only one agent requested, run in-process to avoid multiprocessing file issues
    if len(agents) == 1:
        summaries = []
        agent = agents[0]
        try:
            rewards, summary = eval_agent(params.copy(), agent, n_episodes=N_EPISODES, results_dir=results_dir)
            summaries.append(summary)
        except Exception as e:
            summaries.append({'agent': agent, 'error': str(e)})
    else:
        ctx = multiprocessing.get_context('spawn')
        processes = []
        manager = ctx.Manager()
        summary_list = manager.list()

        # Limit concurrent worker processes
        MAX_WORKERS = 2
        i = 0
        while i < len(agents):
            group = agents[i:i+MAX_WORKERS]
            processes = []
            for agent in group:
                p = ctx.Process(target=_proc_target, args=(agent, params.copy(), N_EPISODES, results_dir, summary_list))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            i += MAX_WORKERS

        summaries = list(summary_list)
    import csv
    with open(os.path.join(results_dir, 'summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['agent','n_episodes','mean_reward','std_reward','min_reward','max_reward','error'])
        writer.writeheader()
        for s in summaries:
            row = {
                'agent': s.get('agent'),
                'n_episodes': s.get('n_episodes'),
                'mean_reward': s.get('mean_reward'),
                'std_reward': s.get('std_reward'),
                'min_reward': s.get('min_reward'),
                'max_reward': s.get('max_reward'),
                'error': s.get('error', '')
            }
            writer.writerow(row)

    print('Evaluation complete. Results saved to results/ directory')


if __name__ == '__main__':
    main()
