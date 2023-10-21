# gymPractice()

import gymnasium as gym
from matplotlib import pyplot as plt

# initialize the environment
env = gym.make('LunarLander-v2', render_mode="human")
observation, info = env.reset()

# loop 1000 times
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()


env.close()
