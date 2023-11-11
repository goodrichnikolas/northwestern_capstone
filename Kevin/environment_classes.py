import os
import time
import csv
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import glob
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
        self.reward_csv_file = os.path.join(log_dir,'reward_logs.csv') # to store the rewards for a csv
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
            model = PPO.load(os.path.join(self.log_dir, self.curr_chk), env=self.env)
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
        reward_list = [rewards[agent] for agent in possible_agents]
        entry = entry + reward_list

        csv_file = self.reward_csv_file
        
        #Create folder if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # insert a header row if needed, otherwise just write the current row
        if not os.path.exists(csv_file):
            
            with open(csv_file,'w') as fid:
                writer = csv.writer(fid)
                # assemble the header row -- adjustable number of agents
                header_row = ['Timestamp','StepCount']
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
        ''' 
        run a series of train and eval loops
        
        Args:
            num_loops   :   number of loops to run
        '''

        # start with an eval to see the randomly initialized model
        self.eval()

        # loop it
        for loop in range(num_loops):
            self.train()
            self.eval()


    def plot_tensorboard_info(self):
        '''
        create a plot of the rewards for the evaluation loops
        
        Data information
        train/policy_gradient_loss
        train/std
        train/value_loss
        time/fps
        time/fps
        train/approx_kl
        train/clip_fraction
        train/clip_range
        train/entropy_loss
        train/explained_variance
        train/learning_rate
        '''
        data = {
        'policy_gradient_loss': [],
        'std': [],
        'value_loss': [],
        'approx_kl': [],
        'clip_fraction': [],
        'clip_range': [],
        'entropy_loss': [],
        'explained_variance': [],
        'learning_rate': [],
        
        }
        #Load all tensorboard files from each folder PPO_1, PPO_2, etc.
        
        tensorfiles_list = glob.glob(os.path.join(self.log_dir,'tensorboard_logs','PPO_*','*'))       
        
        
        # Mapping of tags to data keys
        tag_to_key = {
            'train/policy_gradient_loss': 'policy_gradient_loss',
            'train/std': 'std',
            'train/value_loss': 'value_loss',
            'train/approx_kl': 'approx_kl',
            'train/clip_fraction': 'clip_fraction',
            'train/clip_range': 'clip_range',
            'train/entropy_loss': 'entropy_loss',
            'train/explained_variance': 'explained_variance',
            'train/learning_rate': 'learning_rate'
        }

        for file in tensorfiles_list:
            for e in tf.compat.v1.train.summary_iterator(file):
                for v in e.summary.value:
                    key = tag_to_key.get(v.tag)
                    if key:
                        data[key].append(v.simple_value)                        
        
        df = pd.DataFrame(data)

        #Plotting
        fig,ax = plt.subplots()
        ax.plot(df['policy_gradient_loss'], label='policy_gradient_loss')
        ax.plot(df['std'], label='std')
        ax.plot(df['value_loss'], label='value_loss')
        ax.plot(df['approx_kl'], label='approx_kl')
        ax.plot(df['clip_fraction'], label='clip_fraction')
        ax.plot(df['clip_range'], label='clip_range')
        ax.plot(df['entropy_loss'], label='entropy_loss')
        ax.plot(df['explained_variance'], label='explained_variance')
        ax.plot(df['learning_rate'], label='learning_rate')
        ax.legend()
        
        #Save plot in log_dir and create a folder for charts if it doesn't exist
        if not os.path.exists(os.path.join(self.log_dir,'charts')):
            os.makedirs(os.path.join(self.log_dir,'charts'))
        
        fig.savefig(os.path.join(self.log_dir,'charts','tensorboard_plot.png'))
        
        
    def plot_rewards(self):
        

        '''
        Timestamp,StepCount,agent_pursuer_0,agent_pursuer_1
        create a plot of the rewards for the evaluation loops
        '''
        # load the csv file
        df = pd.read_csv(self.reward_csv_file)
        
        for agent in df.columns[2:]:
            plt.plot(df[agent], label=agent)
        plt.legend()
        plt.savefig(os.path.join(self.log_dir,'charts','reward_plot.png'))
        plt.close()

        



if __name__ == '__main__':
    
    x = waterworld_ppo(log_dir='./log_dir', seed=42, n_envs=8)
    x.plot_rewards()
    x.interlace_run(num_loops=10)



print('Done')