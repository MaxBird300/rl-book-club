# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:35:31 2023

@author: maxbi
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation
plt.style.use('seaborn-pastel')


raw_data = pd.read_csv("k-armed-bandit-results.csv")

x = [x for x in range(len(raw_data))]
y1 = list(raw_data.iloc[:,1])
y2 = list(raw_data.iloc[:,2])
y3 = list(raw_data.iloc[:,3])


fig, ax = plt.subplots(figsize=(5,5), dpi=300)
l1, = ax.plot([], [], 'o-', label="\u03B5 = 0", markevery=[-1])
l2, = ax.plot([], [], 'o-', label="\u03B5 = 0.01", markevery=[-1])
l3, = ax.plot([], [], 'o-', label="\u03B5 = 0.1", markevery=[-1])
ax.legend(loc='lower right')
ax.set_xlim(0,len(raw_data))
ax.set_ylim(0,1.5)
ax.set_ylabel("Average Reward")
ax.set_xlabel("Steps")


def animate(i):
    l1.set_data(x[:i], y1[:i])
    l2.set_data(x[:i], y2[:i])
    l3.set_data(x[:i], y3[:i])
    return (l1,l2,l3)

ani = animation.FuncAnimation(fig, animate, frames=len(raw_data), interval=10, blit=True)

# save animation as mp4
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, bitrate=1800)
ani.save('test.mp4', writer=writer)