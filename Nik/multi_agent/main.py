from pettingzoo.sisl import waterworld_v4

# Env is a list of agents
env = waterworld_v4.env(render_mode='human')

# Reset the environment
env.reset()

# Run the environment for 1000 steps
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)

env.close()