import os
import time
import csv
import logging

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from pettingzoo.test import seed_test, parallel_seed_test, test_save_obs

from pettingzoo.sisl import waterworld_v4


class waterworld_ppo():
    ''' 
    Class to store a PPO training agent, along with all of the settings
    and information about the current training etc.

    Makes it easier to store straining status and evaluate etc
    '''
    
    def __init__(self, log_dir:str = './log_dir', seed:int = 42, 
                 n_envs:int = 8, **env_kwargs):
        '''
        Initialization function. Sets up the environment and learning algorithm

        For now it just takes in optional arguments for the environment. The initialization will
        set up the PPO learner and the working directory to save checkpoints, but won't
        train anything

        Args:
            work_dir    :   where to store the checkpoints
            seed        :   seed for the environment, so we start in the same place
            n_envs      :   how many environments to run in parallel
        '''

        # initialize the environment
        env = waterworld_v4.parallel_env(**env_kwargs) # initialize environment
        env.reset(seed) # so long and thanks for all the fish

        # train N environments in parallel
        env = ss.pettingzoo_env_to_vec_env_v1(env) # vectorize that stuff
        env = ss.concat_vec_envs_v1(vec_env= env, num_vec_envs= n_envs, num_cpus = 2, base_class='stable_baselines3')

        # initialize the model
        model = PPO(
            policy = PPOMlpPolicy,
            env=env,
            tensorboard_log=os.path.join(log_dir,'tensorboard_logs'),
            learning_rate=1e-3,
            batch_size=256
        )
    
        # store parameters in instance
        self.env = env # environment
        self.model = model # model
        self.log_dir = log_dir # for storing model checkpoints and tensorboard data
        self.reward_csv_dir = log_dir+'reward_logs.csv' # to store the rewards for a csv
        self.curr_chk = None # current checkpoint file name -- to allow iterative training
        self.train_stepcount = 0 # how many steps have we trained?


    def train(self, num_steps:int = 100_000):
        '''
        train the model. Loads an existing model checkpoint if available, otherwise starts
        anew
        
        Args:
            num_steps   :   steps per training round
        '''

        if self.curr_chk is not None:
            model = PPO.load(os.path.join(self.log_dir, self.curr_chk))
        else:
            model = self.model # might be able to just do this, instead of reloading

        self.model = model.learn(total_timesteps= num_steps) # does this save in-place? seems to return a model

        # save the current model checkpoint
        self.curr_chk = f"{self.env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.model.save(os.path.join(self.log_dir,self.curr_chk)) # save in the current file name
        self.train_stepcount += num_steps # keep track of how many steps we have trained


    def eval(self, num_games:int=100, render_mode:str=None, 
              **env_kwargs):
        '''
        run the games, and see how many rewards they get for each game

        Args:
            num_games   :   number of games to run
            render_mode :   how/if to display the outputs of the games.
            **envkwargs :   keyword arguments passed to the environment to run
        '''

        # get the current checkpoint. If there isn't one, it's going to run off the randomly initialized values
        if self.curr_chk is not None:
            model = PPO.load(os.path.join(self.log_dir, self.curr_chk)) # load current checkpoint
        else:
            model = self.model

        # apparently we're running this through AEC instead of Parallel. Not sure why
        # since this is for evaluation instead of training, we aren't running multiple simultaneous
        # games so we don't wrap it in SB3. This means we create a new env
        env = waterworld_v4.env(render_mode=render_mode, **env_kwargs)

        # create a dict of agents and rewards received by each
        possible_agents = env.possible_agents # all of the agents
        rewards = {agent:0 for agent in possible_agents}

        for ii in range(num_games):
            # set a different seed for each iteration
            env.reset(seed=ii)

            # loop through the game
            for agent in env.agent_iter():
                # get the action for this iteration
                obs, reward, termination, truncation, info = env.last()

                # get the rewards per agent -- not sure why this is in its own loop
                for agent_rew in env.agents:
                    rewards[agent_rew] += env.rewards[agent_rew]

                # has the game stopped?
                if termination or truncation:
                    break
                else:
                    act = model.predict(obs, deterministic=True)[0]

                env.step(act)
            
        env.close() # stop the AEC environment

        
        # store the reward values
        # order of csv file: timestamp, step_count, rewards for each agent
        entry = [time.strftime('%Y%m%d_%H%M%S'), self.train_stepcount]
        reward_list = [reward[agent] for agent in possible_agents]
        entry = entry + reward_list

        csv_file = self.reward_csv_file

        # insert a header row if needed, otherwise just write the current row
        if not os.path.exists(csv_file):
            with open(csv_file,'w') as fid:
                writer = csv.writer(fid)
                # assemble the header row -- adjustable number of agents
                header_row = ['Timestamp','Step Count']
                agent_list = [f'agent_{agent}' for agent in possible_agents]
                header_row = header_row + agent_list
                # write to csv
                writer.writerow(header_row)
                writer.writerow(entry)
        else:
            with open(csv_file,'a') as fid: # open csv, write it
                writer = csv.writer(fid)
                writer.writerow(entry)


    def interlace_run(self, num_loops=10):
        # run a bunch of train and eval loops

        # start with an eval to see the randomly initialized model
        self.eval()

        # loop it
        for loop in range(num_loops):
            self.train()
            self.eval()