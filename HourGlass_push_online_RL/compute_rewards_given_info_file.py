################################################################################
################################################################################
#HOURGLASS_PUSH is a set of programs related to the paper "An Efficient Image-to-Image
#Translation HourGlass-based Architecture for Object Pushing Policy Learning",
#presented at IROS 2021.
#
#Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#Written by Marco Ewerton <marco.ewerton@idiap.ch>
#
#This file is part of HOURGLASS_PUSH.
#
#HOURGLASS_PUSH is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License version 3 as
#published by the Free Software Foundation.
#
#HOURGLASS_PUSH is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with HOURGLASS_PUSH. If not, see <http://www.gnu.org/licenses/>.
################################################################################

import numpy as np
import os

info_directory = '<PATH TO THE DIRECTORY WHERE THE INFO ABOUT THE EXPERIENCES HAS BEEN RECORDED>'

def compute_reward_push_into_box(avg_dist_before, avg_dist_after, n_new_objs_box, n_new_objs_floor):
    reward = 0
    reward_due_to_change = 0
    reward_due_to_obj_in_box = 0

    if avg_dist_after < avg_dist_before:
        reward = avg_dist_before - avg_dist_after
        reward_due_to_change = avg_dist_before - avg_dist_after

    if n_new_objs_box > 0:
        reward += 10 * n_new_objs_box
        reward_due_to_obj_in_box += 10 * n_new_objs_box

    if n_new_objs_floor > 0:
        reward = 0
        reward_due_to_change = 0
        reward_due_to_obj_in_box = 0

    return reward, reward_due_to_change, reward_due_to_obj_in_box

fh = open(os.path.join(info_directory, 'rewards.txt'), 'w')
fh.write('reward, reward due to change, reward due to obj in box\n')
fh.close()

with open(os.path.join(info_directory, 'info_for_prediction.txt')) as filestream:
    first_line = filestream.readline()  # discard first line, which is just a header
    i = 0
    for line in filestream:
        print(i)
        i += 1
        currentline = line.split(",")
        avg_dist_before = float(currentline[6])
        avg_dist_after = float(currentline[7])
        depth_diff = float(currentline[10])
        n_new_objs_box = int(currentline[11])
        n_new_objs_floor = int(currentline[12])

        reward, reward_due_to_change, reward_due_to_obj_in_box = compute_reward_push_into_box(avg_dist_before, avg_dist_after, n_new_objs_box, n_new_objs_floor)

        fh = open(os.path.join(info_directory, 'rewards.txt'), 'a')
        fh.write(str(reward) + ', ' + str(reward_due_to_change) + ', ' + str(reward_due_to_obj_in_box) + '\n')
        fh.close()
