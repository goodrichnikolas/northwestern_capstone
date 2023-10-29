from pettingzoo.sisl import waterworld_v4
from deps.simple_dqn_keras import Agent


# start the environment
env = waterworld_v4.env(render_mode='human')

# get information about the action space and agents
n_agents = env.num_agents
agents = {}
for i_agent in range(n_agents):
    agents[i_agent] = Agent(gamma=0.99, epsilon=0.0, alpha=0.0)

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy

    env.step(action)
env.close()
