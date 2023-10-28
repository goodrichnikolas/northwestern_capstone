from pettingzoo.sisl import waterworld_v4

def completion_percentage(current_count, max_count):
    return round((current_count / max_count) * 100)

n_pursuers = 2
n_evaders = 2
n_poisons = 2
n_coop = 2
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

env = waterworld_v4.env(n_pursuers=n_pursuers, n_evaders=n_evaders, n_poisons=n_poisons,
                        n_coop=n_coop, n_sensors=n_sensors, sensor_range=sensor_range,
                        radius=radius, obstacle_radius=obstacle_radius, obstacle_coord=obstacle_coord,
                        pursuer_max_accel=pursuer_max_accel, pursuer_speed=pursuer_speed,
                        evader_speed=evader_speed, poison_speed=poison_speed, poison_reward=poison_reward,
                        food_reward=food_reward, encounter_reward=encounter_reward,
                        thrust_penalty=thrust_penalty, local_ratio=local_ratio,
                        speed_features=speed_features, max_cycles=max_cycles)

env.reset(seed=42)
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy

    env.step(action)
    
    #Add to count
    if agent == 'pursuer_0':
        current_count += 1
        #Print percentage complete
        print(f'{completion_percentage(current_count, max_cycles)}%')
    
env.close()