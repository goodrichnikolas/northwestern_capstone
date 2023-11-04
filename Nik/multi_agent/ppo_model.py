"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations
import sys
import glob
import os
import time
import pandas as pd

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from pettingzoo.test import seed_test, parallel_seed_test, test_save_obs

from pettingzoo.sisl import waterworld_v4

#Unix time
import time



#Alter env_fn with a subclass that will allow communication between agents
    


#Arguments
#Number of pursuers
n_pursuers = 2
#Food
n_evaders = 2
n_poisons = 2
#Number of agents that must touch food to get reward together
n_coop = 0
#Black lines
n_sensors = 10
#length of the black lines
sensor_range = 0.2
#archea base radius. Pursuer: radius, food: 2 x radius, poison: 3/4 x radius
radius = 0.05
#radius of obstacle object
obstacle_radius = 0.2
#Where the obstacle is located
obstacle_coord = [(0.5, 0.5)]
pursuer_max_accel = 0.01
pursuer_speed = 0.05
#0 means stationary food
evader_speed = 0.05
poison_speed = 0.05
poison_reward = -1
food_reward = 1
encounter_reward = 0.01
#scaling factor for the negative reward used to penalize large actions
thrust_penalty = -0.01
local_ratio = 0.5
#toggles whether pursuing archea (agent) sensors detect speed of other objects and archea
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


def eval(env_fn, num_games: int = 100, render_mode: str = None, current_count: int | None = None, **env_kwargs):
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
    
    possible_agents = env.possible_agents

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
    
    #create empty csv file to store rewards using pandas
    data_dict = {
        'agent': [],
        'reward': [],
        'average_reward': [],
        'current_count': [],
    }
    
    #populate data_dict with rewards
    for agent in possible_agents:
        data_dict['agent'].append(agent)
        data_dict['reward'].append(rewards[agent])
        data_dict['average_reward'].append(sum(rewards.values()) / len(rewards.values()))
        data_dict['current_count'].append(current_count + 100000)
    
    #create csv file to store rewards using pandas
    
    columns = ['agent', 'reward', 'average_reward']
    
    #create csv empty file if it does not exist
    if not os.path.exists('./Nik/multi_agent/rewards/rewards.csv'):
        #Create csv with column headers but no data
        df = pd.DataFrame(columns=columns)
        df.to_csv('./Nik/multi_agent/rewards/rewards.csv', index=False)
        
    #append data to csv file
    df = pd.read_csv('./Nik/multi_agent/rewards/rewards.csv')
    new_df = pd.DataFrame(data_dict)
    
    #concatenate new_df to df
    df = pd.concat([df, new_df])
    df.to_csv('./Nik/multi_agent/rewards/rewards.csv', index=False)

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


if __name__ == "__main__":


    env_fn = waterworld_v4
    

    
    
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
    
    
    
    total_steps = 300000    
    divider = 100000
    current_count = 0
    
    #Run train_butterfly_supersuit for a total of 10 million steps at 100k steps per run, then evaluate each 100k steps and output the average reward
    
    for i in range(total_steps//divider):
        
        train_butterfly_supersuit(env_fn, steps=divider, seed=0, **env_kwargs)
        eval(env_fn, num_games=10, render_mode=None, current_count=current_count, **env_kwargs)
        current_count += divider
        
    

    # Train a model (takes ~3 minutes on GPU)
    #train_butterfly_supersuit(env_fn, steps=100000, seed=0, **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    # eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    
