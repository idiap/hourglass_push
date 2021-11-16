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
import matplotlib.pyplot as plt
import user_definitions as ud
import torch
import pickle

# # MASK
# loss_hist_mask = torch.load(ud.loss_hist_mask_tensor_file).numpy()
#
# print(loss_hist_mask.shape)
#
# plt.figure()
#
# plt.plot(loss_hist_mask)
# plt.grid()
#
# plt.title('Mask Net')
# plt.xlabel('batch')
# plt.ylabel('loss')
#
# plt.savefig(ud.learning_curve_mask_folder + 'learning_curve_mask_loss.png')
# plt.savefig(ud.learning_curve_mask_folder + 'learning_curve_mask_loss.svg')
# plt.savefig(ud.learning_curve_mask_folder + 'learning_curve_mask_loss.pdf')
#
# plt.show()


# PUSH INTO BOX
loss_hist_push_into_box = torch.load(ud.loss_hist_push_into_box_tensor_file).numpy()

print(loss_hist_push_into_box.shape)

plt.figure()

plt.plot(loss_hist_push_into_box)
plt.grid()

plt.title('Push Into Box Net')
plt.xlabel('batch')
plt.ylabel('loss')

plt.savefig(ud.learning_curve_push_into_box_folder + 'learning_curve_push_into_box_loss.png')
plt.savefig(ud.learning_curve_push_into_box_folder + 'learning_curve_push_into_box_loss.svg')
plt.savefig(ud.learning_curve_push_into_box_folder + 'learning_curve_push_into_box_loss.pdf')

plt.show()


# REWARD PUSH INTO BOX

def compute_reward_push_into_box(avg_dist_before, avg_dist_after, n_new_objs_box, n_new_objs_floor):
    if n_new_objs_floor > 0:
        return 0
    elif n_new_objs_box > 0:
        return 20 * n_new_objs_box
    elif not np.isnan(avg_dist_after):
        if avg_dist_after < avg_dist_before:
            return 10
        else:
            return 5
    else:
        return 5

txt_file = '<PATH TO info_for_prediction.txt>'

scenarios = []
input_images = []
push_indexes = []
# next_images = []
# avg_dists_before_action = []
# avg_dists_after_action = []
# current_sums_True_bits = []
# next_sums_True_bits = []
depth_diffs = []
# ns_new_objects_reached_target = []
# ns_new_objects_fell_to_the_ground = []
rewards_mask = []
rewards_push_into_box = []

with open(txt_file) as filestream:
    first_line = filestream.readline()  # discard first line, which is just a header
    for line in filestream:
        currentline = line.split(",")
        scenarios.append(int(currentline[0]))
        input_images.append('%06d.0.depth.png' % int(currentline[1]))
        s1 = currentline[2][2:]
        s2 = currentline[3]
        s3 = currentline[4][:-1]
        push_indexes.append([int(s1), int(s2), int(s3)])
        # next_images.append('%06d.0.depth.png' % int(currentline[5]))
        # avg_dists_before_action.append(float(currentline[6]))
        # avg_dists_after_action.append(float(currentline[7]))
        # current_sums_True_bits.append(int(currentline[8]))
        # next_sums_True_bits.append(int(currentline[9]))
        # depth_diffs.append(float(currentline[10]))
        # ns_new_objects_reached_target.append(int(currentline[11]))
        # ns_new_objects_fell_to_the_ground.append(int(currentline[12]))

        depth_diff = float(currentline[10])

        avg_dist_before = float(currentline[6])
        avg_dist_after = float(currentline[7])
        n_new_objs_box = int(currentline[11])
        n_new_objs_floor = int(currentline[12])
        # reward_mask = compute_reward_mask(n_new_objs_floor, depth_diff)
        # rewards_mask.append(reward_mask)
        reward_push_into_box = compute_reward_push_into_box(avg_dist_before, avg_dist_after, n_new_objs_box,
                                                                 n_new_objs_floor)
        rewards_push_into_box.append(reward_push_into_box)

plt.figure()

plt.plot(rewards_push_into_box)
plt.grid()

plt.title('Push Into Box Net')
plt.xlabel('action')
plt.ylabel('reward')

plt.savefig(ud.learning_curve_push_into_box_folder + 'learning_curve_push_into_box_reward.png')
plt.savefig(ud.learning_curve_push_into_box_folder + 'learning_curve_push_into_box_reward.svg')
plt.savefig(ud.learning_curve_push_into_box_folder + 'learning_curve_push_into_box_reward.pdf')

plt.show()


