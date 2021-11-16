#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Plot options (change me)
interval_size = 200 # Report performance over the last 200 training steps
max_plot_iteration = 7500 # Maximum number of training steps to report performance

# Parse session directories
parser = argparse.ArgumentParser(description='Plot performance of a session over training time.')
parser.add_argument('session_directories', metavar='N', type=str, nargs='+', help='path to session directories for which to plot performance')
args = parser.parse_args()
session_directories = args.session_directories

# Define plot colors (Tableau palette)
colors = [[078.0/255.0,121.0/255.0,167.0/255.0], # blue
          [255.0/255.0,087.0/255.0,089.0/255.0], # red
          [089.0/255.0,169.0/255.0,079.0/255.0], # green
          [237.0/255.0,201.0/255.0,072.0/255.0], # yellow
          [242.0/255.0,142.0/255.0,043.0/255.0], # orange
          [176.0/255.0,122.0/255.0,161.0/255.0], # purple
          [255.0/255.0,157.0/255.0,167.0/255.0], # pink 
          [118.0/255.0,183.0/255.0,178.0/255.0], # cyan
          [156.0/255.0,117.0/255.0,095.0/255.0], # brown
          [186.0/255.0,176.0/255.0,172.0/255.0]] # gray

# Create plot design
plt.ylim((0, 1))
plt.ylabel('Reward')
plt.xlim((0, max_plot_iteration))
plt.xlabel('Number of training steps')
plt.grid(True, linestyle='-', color=[0.8,0.8,0.8])
ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_color('#000000')
plt.rcParams.update({'font.size': 18})
plt.rcParams['mathtext.default']='regular'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
legend = []

for session_idx in range(len(session_directories)):
    session_directory = session_directories[session_idx]
    color = colors[session_idx % 10]

    # Get logged data
    transitions_directory = os.path.join(session_directory, 'transitions')
    executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
    max_iteration = min(executed_action_log.shape[0] - 2, max_plot_iteration)
    executed_action_log = executed_action_log[0:max_iteration,:]
    reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
    reward_value_log = reward_value_log[0:max_iteration]
    predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
    predicted_value_log = predicted_value_log[0:max_iteration]

    # # Compute average reward up to a certain time step
    # avg_reward = reward_value_log*0
    # for i in range(max_iteration):
    #     avg_reward[i] = np.mean(reward_value_log[:i+1])
    # # Plot average reward
    # plt.plot(range(0, max_iteration), avg_reward, color=color, linewidth=3) # color='blue', linewidth=3)

    # Compute absolute value of the difference between the predicted reward and the actual reward
    abs_diff_reward = np.abs(reward_value_log - predicted_value_log)
    # Plot absolute value of the difference between the predicted reward and the actual reward
    plt.plot(range(0, max_iteration), abs_diff_reward, color=color, linewidth=3)  # color='blue', linewidth=3)

    legend.append(session_directories[session_idx])

plt.legend(legend, loc='lower right', fontsize=18)
plt.tight_layout()
plt.show()
