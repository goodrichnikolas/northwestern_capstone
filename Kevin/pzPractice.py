# pzPractice
"""
Testing the mpe and sisl environments to get everything running for our actual learning stuff

Args:
    - none - 

"""


# imports 
from pettingzoo.butterfly import cooperative_pong_v5

env = cooperative_pong_v5.env(render_mode = 'human')
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()


    if termination or truncation:
        action = None

    else:
        action = env.action_space(agent).sample()

    env.step(action)

env.close()
