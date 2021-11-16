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

#!/usr/bin/env python
# coding: utf-8

import os
import torch
import PushingSuccessPredictor as psp
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
from tqdm.notebook import trange, tqdm
import cv2
import user_definitions as ud


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Compute forward pass through model to compute affordances/Q
def forward(depth_heightmap, mask_net, push_into_box_net):
    mask_net.eval()
    push_into_box_net.eval()

    with torch.no_grad():
        # Pass input data through models
        depth_heightmap = torch.from_numpy(depth_heightmap).float()
        depth_heightmap = depth_heightmap.view(1, 1, 224, 224).to(device)
        mask = mask_net(depth_heightmap)
        push_into_box = push_into_box_net(depth_heightmap)

        mask = mask >= ud.mask_prob_threshold  # True wherever A >= ud.mask_prob_threshold, False otherwise.
        mask = mask.float()

        if ud.use_mask:
            push_predictions = push_into_box * mask
        else:
            push_predictions = push_into_box

    return push_predictions


class PushingDataset(Dataset):
    """Pushing dataset."""

    def __init__(self, txt_file, images_directory, mask_net, push_into_box_net, transform=None):
        self.pushing_dataset = {}
        self.images_directory = images_directory
        self.transform = transform
        # We will be computing q_max_next using the current mask_net and push_into_box_net
        self.mask_net = mask_net
        self.push_into_box_net = push_into_box_net
        self.extract_data_from_txt(txt_file)

    def __len__(self):
        return len(self.pushing_dataset['scenarios'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img_name = os.path.join(self.images_directory, self.pushing_dataset['input_images'][idx])
        input_img = cv2.imread(input_img_name, -1)
        input_img = input_img.astype(
            np.float32) / 100000  # This is exactly the way images are read in the push-into-the-box code.
        # The result has heights in meters.

        # next_img_name = os.path.join(self.images_directory, self.pushing_dataset['next_images'][idx])
        # next_img = cv2.imread(next_img_name, -1)
        # next_img = next_img.astype(
        #     np.float32) / 100000  # This is exactly the way images are read in the push-into-the-box code.
        # # The result has heights in meters.

        sample = {'scenario': self.pushing_dataset['scenarios'][idx],
                  'input_image': input_img,
                  'push_indexes': self.pushing_dataset['push_indexes'][idx],
                  'reward_mask': self.pushing_dataset['rewards_mask'][idx],
                  'reward_push_into_box': self.pushing_dataset['rewards_push_into_box'][idx],
                  'q_max_next': self.pushing_dataset['qs_max_next'][idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def compute_reward_mask(self, n_new_objs_floor, depth_diff):
        if depth_diff >= 1000:
            return 1
        else:
            return 0

    def compute_reward_push_into_box(self, avg_dist_before, avg_dist_after, n_new_objs_box, n_new_objs_floor):
        reward = 0

        if avg_dist_after < avg_dist_before:
            reward = avg_dist_before - avg_dist_after

        if n_new_objs_box > 0:
            reward += 10 * n_new_objs_box

        if n_new_objs_floor > 0:
            reward = 0

        return reward

    def compute_q_max_next(self, next_image_name, avg_dist_after):

        print('Computing q_max_next')

        if np.isnan(avg_dist_after):
            print('avg_dist_after is NaN')
            return 0
        else:
            next_img_name = os.path.join(self.images_directory, next_image_name)
            next_img = cv2.imread(next_img_name, -1)
            next_img = next_img.astype(
                np.float32) / 100000

            q_max_next = torch.max(forward(next_img, self.mask_net, self.push_into_box_net))
            return q_max_next.item()

    # Extract data from the (ud.train_batch_size*ud.batch_multiplier) last lines of txt_file
    def extract_data_from_txt(self, txt_file):
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
        qs_max_next = []
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
                reward_mask = self.compute_reward_mask(n_new_objs_floor, depth_diff)
                rewards_mask.append(reward_mask)
                reward_push_into_box = self.compute_reward_push_into_box(avg_dist_before, avg_dist_after, n_new_objs_box, n_new_objs_floor)
                rewards_push_into_box.append(reward_push_into_box)

                q_max_next = self.compute_q_max_next('%06d.0.depth.png' % int(currentline[5]), avg_dist_after)
                qs_max_next.append(q_max_next)

        self.pushing_dataset['scenarios'] = scenarios
        self.pushing_dataset['input_images'] = input_images
        self.pushing_dataset['push_indexes'] = push_indexes
        # self.pushing_dataset['next_images'] = next_images
        # self.pushing_dataset['avg_dists_before_action'] = avg_dists_before_action
        # self.pushing_dataset['avg_dists_after_action'] = avg_dists_after_action
        # self.pushing_dataset['current_sums_True_bits'] = current_sums_True_bits
        # self.pushing_dataset['next_sums_True_bits'] = next_sums_True_bits
        # self.pushing_dataset['depth_diffs'] = depth_diffs
        # self.pushing_dataset['ns_new_objects_reached_target'] = ns_new_objects_reached_target
        # self.pushing_dataset['ns_new_objects_fell_to_the_ground'] = ns_new_objects_fell_to_the_ground
        self.pushing_dataset['rewards_mask'] = rewards_mask
        self.pushing_dataset['rewards_push_into_box'] = rewards_push_into_box
        self.pushing_dataset['qs_max_next'] = qs_max_next


# Transforms

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input_image = sample['input_image']

        return {'scenario': sample['scenario'],
                'input_image': torch.from_numpy(input_image).float(),
                'push_indexes': sample['push_indexes'],
                'reward_mask': sample['reward_mask'],
                'reward_push_into_box': sample['reward_push_into_box'],
                'q_max_next': sample['q_max_next']}


def plot_action(ax, action, arrow_length_in_x):
    angle = action[0]*(2*np.pi/16) - np.pi  # The number of rotations is 16
    plt.quiver(action[2], action[1], -arrow_length_in_x*np.cos(angle), arrow_length_in_x*np.sin(angle),
               color='b', alpha=1, scale=1, scale_units='x')
    ax.set_aspect('equal')


# Show for a given sample the input image, the next image and the action
def show_sample(dataset, idx):
    fig = plt.figure()

    sample = dataset[idx]

    scenario = sample['scenario']
    action = sample['push_indexes']  # action[0]: angle, action[2]: x coordinate, action[1]: y coordinate
    reward_mask = sample['reward_mask']
    reward_push_into_box = sample['reward_push_into_box']
    #     predicted_reward = sample['predicted_reward']
    print('scenario: ', scenario)
    print('action: ', action)
    print('reward_mask: ', reward_mask)
    print('reward_push_into_box: ', reward_push_into_box)
    #     print('predicted_reward: ', predicted_reward)

    # plot input image
    ax = plt.subplot(1, 2, 1)
    ax.set_title('Input image')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.imshow(sample['input_image'])
    plot_action(action)

#     # plot next image
#     ax = plt.subplot(1, 2, 2)
#     ax.set_title('Next image')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     plt.imshow(sample['next_image']);
#     plot_action(action);
#     plt.tight_layout()

    plt.show()


# Helper function to show a batch
def show_batch(ax, sample_batched):
    input_images_batch = sample_batched['input_image']
    # actions = sample_batched['push_indexes']
    # next_images_batch = sample_batched['next_image']

    batch_size = len(input_images_batch)

    for i in range(batch_size):
        # action = [int(actions[0][i]), int(actions[1][i]), int(actions[2][i])]
        # print(action)

        # plot input image

        # ax.set_title('Input image')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.imshow(input_images_batch[i]);
        # plot_action(action);

        # # plot next image
        # ax = plt.subplot(1, 2, 2)
        # ax.set_title('Next image')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # plt.imshow(next_images_batch[i]);
        # plot_action(action);
        # plt.tight_layout()

        # plt.show()