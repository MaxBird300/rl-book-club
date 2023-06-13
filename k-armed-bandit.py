# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:01:25 2023

Code to simulate the k-armed bandit problem in reinforcement learning:
    - Each bandit when activated gives a reward, which has an unknown underlying probability distribution
    - Aim of the RL is, via interacting with k bandits, learn how to generate the highest score from activating the bandits

Definitions:
    - Reward (R) is the immediate reward from taking an action
    - Value (Q) is the average reward over a number of trials. 
      
Pseudo code:
    At each step, you need to track the value of each action, Q(a), and the number of times that action has been selected, N(a)
    
    for each timestep:
        - Choose action based on epsilon-greedy choice
        - Calculate reward R for chosen action using unknown underlying probability distributions
        - Update number of times the action has been selected, N(a)
        - Update the value for action a, Q(a), using the incremental implementation formula

@author: Max
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, mean: float, std_dev: float):
        self.mu = mean
        self.sigma = std_dev
        
    def generate_reward(self) -> float:
        # sample a normal distribution with mean = mu and standard deviation = sigma
        return norm.rvs(loc = self.mu, scale = self.sigma, size=1)[0]

class BanditRL:
    def __init__(self, epsilon: float, k_bandits: int, q_init: float=0.):
        self.epsilon = epsilon # probability of exploring [0,1]
        self.k_bandits = k_bandits # number of bandits
        
        # initialise N(a) and Q(a) as 0 for every bandit
        # these lists are updated after each learning step
        self.num_bandit_selections = [0 for bandit in range(k_bandits)] # N(a)
        self.bandit_values = [q_init for bandit in range(k_bandits)] # Q(a)
    
    def choose_action(self) -> int:
        """ 
        - choose greedy action with prob 1-epsilon
        - choose random action with prob epsilon
        - return which action is chosen [0, k_bandits]
        """
        greedy_bool = np.random.choice(a=[True, False], p=[1-self.epsilon, self.epsilon])
        
        if greedy_bool: # greedy choice
            # what is the max value in the bandit values
            max_value = max(self.bandit_values) 
            # which bandit(s) have this max value
            max_actions = [bandit for bandit, bandit_value in enumerate(self.bandit_values) if bandit_value == max_value]
            
            if len(max_actions) == 1: # only one action with max value
                action = max_actions[0] # greedy action
            else: # if more than one action has the max value, choose randomly between these
                action = np.random.choice(a=max_actions)
                
        else: # exploration
            action = np.random.choice(a=[x for x in range(self.k_bandits)])
            
        return action
    
    def _update_num_bandit_selections(self, action: int) -> None:
        self.num_bandit_selections[action] += 1
    
    def _update_bandit_values(self, action: int, reward: float) -> None:
        # incrementally computed sample averages (equation 2.3)
        if self.num_bandit_selections[action] != 0:
            new_bandit_val = (
                self.bandit_values[action] + 
                1/self.num_bandit_selections[action] * (reward-self.bandit_values[action])
                )
            
            self.bandit_values[action] = new_bandit_val
        
    def update_bandit_info(self, action: int, reward: float):
        self._update_bandit_values(action, reward)
        self._update_num_bandit_selections(action)


def simulate_rl_performance(k_bandits: int, epsilon: float, N_steps: int, N_runs: int, q_init: float):
    """
    Simulate the average performance of epsilon-greedy RL algorithm to learn 
    solution to k-bandits problem.
    
    k_bandits: number of bandits to consider
    epsilon: exploration parameter, 0 = greeedy, 1 = random search
    N_steps: number of consecutive actions the RL algorithm will take per run
    N_runs: number of total runs to average the RL performance over

    """
    
    all_run_rewards = []
    all_run_optimal_action_bools = []
    for run_num in range(N_runs):
        # define bandit instances
        bandit_means = norm.rvs(loc = 0, scale = 1, size=k_bandits)    
        bandits = [Bandit(mean, std_dev=1) for mean in bandit_means]
        optimal_action = bandit_means.argmax()
        # define RL instance
        bandit_rl = BanditRL(epsilon, k_bandits, q_init)
        
        rewards_for_run_n = []
        optimal_action_bool_for_run_n = []
        for j in range(N_steps):
            action = bandit_rl.choose_action() # action corresponds to which bandit you select
            reward = bandits[action].generate_reward() # sample actual reward from "unknown" bandit probability distributions
            bandit_rl.update_bandit_info(action, reward)
            rewards_for_run_n.append(reward)
            
            # store if optimal action is selected
            if action == optimal_action:
                optimal_action_bool_for_run_n.append(1)
            else:
                optimal_action_bool_for_run_n.append(0)
                
        all_run_rewards.append(rewards_for_run_n) # list of lists
        all_run_optimal_action_bools.append(optimal_action_bool_for_run_n) # list of lists
        
    run_rewards_df = pd.DataFrame(all_run_rewards).T # index is N_steps, columns are N_runs
    average_rewards = run_rewards_df.mean(axis=1)
    
    optimal_action_df = pd.DataFrame(all_run_optimal_action_bools).T
    optimal_action_percent = optimal_action_df.mean(axis=1) * 100

    
    return average_rewards, optimal_action_percent


def plot_average_rewards(average_rewards: list, epsilons: list, k_bandits: int):
    
    fig, axes = plt.subplots(1, figsize=(5,3), dpi=200)
    for y, epsilon in zip(average_rewards, epsilons):
        axes.plot(y, label=f"\u03B5 = {epsilon}")
    axes.legend()
    axes.set_ylabel("Average Reward")
    axes.set_xlabel("Steps")
    axes.set_title(f"k-bandits simulation for k={k_bandits}")
    


k_bandits = 10
N_steps = 1000
N_runs = 2000
q_init = 5
epsilons = [0, 0.01, 0.1, 0.5]
# epsilon = 0
save_folder = './rl_run_data/'


all_average_rewards = []
all_optimal_action_percents = []

for epsilon in epsilons:
    average_rewards, optimal_action_percent = simulate_rl_performance(k_bandits, epsilon, N_steps, N_runs, q_init)
    all_average_rewards.append(average_rewards)
    all_optimal_action_percents.append(optimal_action_percent)
        
plot_average_rewards(all_average_rewards, epsilons, k_bandits)

average_rewards = pd.concat(all_average_rewards, axis=1)
average_rewards.columns = epsilons

optimal_action_percent = pd.concat(all_optimal_action_percents, axis=1)
optimal_action_percent.columns = epsilons


# save data
average_rewards.to_csv(f"{save_folder}k-armed-bandit-average-rewards.csv")
optimal_action_percent.to_csv(f"{save_folder}k-armed-bandit-optimal-action-percent.csv")


