"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations
import sys
import glob
import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy

from pettingzoo.sisl import waterworld_v4


#Arguments
n_pursuers = 2
n_evaders = 2
n_poisons = 2
n_coop = 0
n_sensors = 10
sensor_range = 0.2
radius = 0.05
obstacle_radius = 0.2
obstacle_coord = [(0.5, 0.5)]
pursuer_max_accel = 0.01
pursuer_speed = 0.05
evader_speed = 0.05
poison_speed = 0.05
poison_reward = -1
food_reward = 1
encounter_reward = 0.01
thrust_penalty = -0.01
local_ratio = 0.5
speed_features = False
max_cycles = 1000
current_count = 0


def train_butterfly_supersuit(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn.parallel_env(**env_kwargs)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    #Do not use vectorized, only run one game at a time
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = PPO(
        PPOMlpPolicy,
        env,
        verbose=3,
        learning_rate=1e-3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in range(num_games):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for agent in env.agents:
                rewards[agent] += env.rewards[agent]
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


if __name__ == "__main__":


    env_fn = waterworld_v4
    
    
    #Alter env_fn with a subclass 
    
    
    env_kwargs = {
        "n_pursuers": n_pursuers,
        "n_evaders": n_evaders,
        "n_poisons": n_poisons,
        "n_coop": n_coop,
        "n_sensors": n_sensors,
        "sensor_range": sensor_range,
        "radius": radius,
        "obstacle_radius": obstacle_radius,
        "obstacle_coord": obstacle_coord,
        "pursuer_max_accel": pursuer_max_accel,
        "pursuer_speed": pursuer_speed,
        "evader_speed": evader_speed,
        "poison_speed": poison_speed,
        "poison_reward": poison_reward,
        "food_reward": food_reward,
        "encounter_reward": encounter_reward,
        "thrust_penalty": thrust_penalty,
        "local_ratio": local_ratio,
        "speed_features": speed_features,
        "max_cycles": max_cycles,
    }

    # Train a model (takes ~3 minutes on GPU)
    train_butterfly_supersuit(env_fn, steps=196_608, seed=0, **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    
    
    
