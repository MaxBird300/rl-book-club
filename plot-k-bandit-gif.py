# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:35:31 2023

@author: maxbi
"""

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation
# plt.style.use('seaborn-pastel')

def save_animation(ani: animation, save_path: str):
    # save animation as mp4
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, bitrate=1800)
    ani.save(save_path, writer=writer)
    

def animate(i, x, ys, lines):
    for y, l in zip(ys, lines):
        l.set_data(x[:i], y[:i])
    return tuple(lines)


def plot_average_rewards_animation(reward_df):
    fig, ax = plt.subplots(figsize=(5,4), dpi=200)
    
    x = [x for x in range(len(reward_df))]
    ys = [list(reward_df.loc[:,col]) for col in reward_df.columns]
    
    labels = [f"{col}" for col in reward_df.columns]
    lines = [ax.plot([], [], 'o-', label=label, markevery=[-1])[0] for label in labels]
    
    ax.legend(loc='lower right')
    ax.set_xlim(0,len(reward_df))
    ax.set_ylim(0,1.8)
    ax.set_ylabel("Average Reward")
    ax.set_xlabel("Steps")
    ax.set_title("\u03B5-greedy, 10-armed testbed")
    
    ani = animation.FuncAnimation(fig, animate, fargs=(x, ys, lines), frames=len(reward_df), interval=10, blit=True)
    
    return ani

def plot_optimal_action_percentage(optimal_action_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5,4), dpi=200)
    
    x = [x for x in range(len(optimal_action_df))]
    ys = [list(optimal_action_df.loc[:,col]) for col in optimal_action_df.columns]
    
    labels = [f"{col}" for col in optimal_action_df.columns]
    lines = [ax.plot([], [], 'o-', label=label, markevery=[-1])[0] for label in labels]
    
    ax.legend(loc='lower right')
    ax.set_xlim(0,len(optimal_action_df))
    ax.set_ylim(0,100)
    ax.set_ylabel("Optimal Action [%]")
    ax.set_xlabel("Steps")
    ax.set_title("\u03B5-greedy, 10-armed testbed")
    
    ani = animation.FuncAnimation(fig, animate, fargs=(x, ys, lines), frames=len(optimal_action_df), interval=10, blit=True)
    
    return ani

# k-bandits for different epsilon values
read_folder = './rl_run_data/'
# reward_df = pd.read_csv(f"{read_folder}k-armed-bandit-average-rewards.csv", index_col=0)
# optimal_action_df = pd.read_csv(f"{read_folder}k-armed-bandit-optimal-action-percent.csv", index_col=0)

# reward_ani = plot_average_rewards_animation(reward_df)
# optimal_action_ani = plot_optimal_action_percentage(optimal_action_df)

# save_animation(reward_ani, save_path=f"{read_folder}average_reward_animation.mp4")
# save_animation(optimal_action_ani, save_path=f"{read_folder}optimal_action_animation.mp4")


# changing initial value estimations
optimistic_init_vals_action_percent = pd.read_csv(f"{read_folder}optimistic_initial_value_comparison_optimal_action.csv", index_col=0)
optimistic_action_ani = plot_optimal_action_percentage(optimistic_init_vals_action_percent)

optimistic_intial_vals_ave_reward = pd.read_csv(f"{read_folder}optimistic_initial_value_comparison_average_reward.csv", index_col=0)
optimistic_reward_ani = plot_average_rewards_animation(optimistic_intial_vals_ave_reward)

save_animation(optimistic_action_ani, save_path=f"{read_folder}optimistic_action_animation.mp4")
save_animation(optimistic_reward_ani, save_path=f"{read_folder}optimistic_reward_animation.mp4")




