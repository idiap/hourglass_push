################################################################################
################################################################################
#HOURGLASS_PUSH is a set of programs related to the paper "An Efficient Image-to-Image
#Translation HourGlass-based Architecture for Object Pushing Policy Learning",
#presented at IROS 2021.
#
#Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#Written by Marco Ewerton <marco.ewerton@idiap.ch>
#Code adapted from Visual Pushing and Grasping Toolbox
#(Source: https://github.com/andyzeng/visual-pushing-grasping retrieved on September 27, 2021)
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

import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import CrossEntropyLoss2d
import PushingSuccessPredictor as psp
import Push_Into_Box_Net as pibn
from scipy import ndimage
import matplotlib.pyplot as plt
from numpy import linalg as LA
import user_definitions as ud
import helper
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class Trainer(object):
    def __init__(self, is_training, is_testing, load_snapshot, mask_snapshot_file, push_into_box_snapshot_file, force_cpu, robot, logger, discount_factor):

        self.model_updates_counter = 0

        self.discount_factor = discount_factor

        self.robot = robot

        self.n_objects_reached_target = 0
        self.n_objects_fell_to_the_ground = 0

        # HourGlass-based model to predict the success of pushes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        psp_params = psp.get_psp_parameters()
        self.mask_net = psp.PushingSuccessPredictor(psp_params).to(self.device)
        pibn_params = pibn.get_pibn_parameters()
        self.push_into_box_net = pibn.Push_Into_Box_Net(pibn_params).to(self.device)

        # Load pre-trained models
        if load_snapshot:
            mask_checkpoint = torch.load(mask_snapshot_file)
            self.mask_net.load_state_dict(mask_checkpoint['mask_model_state_dict'])
            print('Pre-trained mask model snapshot loaded from: %s' % (mask_snapshot_file))

            push_into_box_checkpoint = torch.load(push_into_box_snapshot_file)
            self.push_into_box_net.load_state_dict(push_into_box_checkpoint['push_into_box_model_state_dict'])
            print('Pre-trained push into box model snapshot loaded from: %s' % (push_into_box_snapshot_file))

        self.target = [168., 0.]
        self.obj_handles_to_remove = []

        # Precompute the distance between each pixel and the target
        self.resolution = 224  # Assuming number of rows = number of columns
        img_indices = np.indices((self.resolution, self.resolution))
        img_indices = np.vstack((img_indices[0].flatten(), img_indices[1].flatten())).transpose()
        self.pixel_distances = LA.norm(img_indices - self.target, axis=1)
        self.pixel_distances = self.pixel_distances.reshape((self.resolution, self.resolution))

        self.info_file = os.path.join(logger.base_directory, 'info_for_prediction_minibatch.txt')
        self.images_directory = logger.depth_heightmaps_directory
        self.train_indices = list(range(0, ud.train_batch_size * ud.batch_multiplier))
        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.criterion_mask = torch.nn.BCELoss(reduction='none').to(self.device)  # Binary Cross Entropy Loss
        self.criterion_push_into_box = torch.nn.MSELoss(reduction='none')
        self.optimizer_mask = torch.optim.Adam(self.mask_net.parameters(), lr=1e-4)
        # self.optimizer_push_into_box = torch.optim.Adam(self.mask_net.parameters(), lr=1e-3)
        self.optimizer_push_into_box = torch.optim.SGD(self.push_into_box_net.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.effective_loss_mask = 0
        self.effective_loss_push_into_box = 0
        self.loss_hist_mask = []
        self.loss_hist_push_into_box = []

    # Compute forward pass through model to compute affordances/Q
    def forward(self, depth_heightmap):

        return helper.forward(depth_heightmap, self.mask_net, self.push_into_box_net)

    def get_label_value(self, next_depth_heightmap, current_depth_heightmap):

        obj_positions = self.robot.get_obj_positions()
        obj_handles = self.robot.object_handles

        depth_diff = abs(next_depth_heightmap - current_depth_heightmap)
        depth_diff[np.isnan(depth_diff)] = 0
        depth_diff[depth_diff > 0.3] = 0
        depth_diff[depth_diff < 0.01] = 0
        depth_diff[depth_diff > 0] = 1
        depth_diff = np.sum(depth_diff)

        next_binary_map = next_depth_heightmap > 0.201  # 0.2m is the height of the table after my preprocessing.
                                                       # The additional 0.001m (1mm) is just to make sure we count
                                                       # pixels corresponding to objects and not pixels on the table.
        current_binary_map = current_depth_heightmap > 0.201
        next_sum_dists = np.sum(self.pixel_distances * next_binary_map)
        current_sum_dists = np.sum(self.pixel_distances * current_binary_map)
        next_sum_True_bits = np.sum(next_binary_map)
        current_sum_True_bits = np.sum(current_binary_map)
        avg_next_dists = next_sum_dists / next_sum_True_bits
        avg_current_dists = current_sum_dists / current_sum_True_bits

        for i in range(len(obj_positions)):
            obj_position_x = np.array(obj_positions[i][0])
            obj_position_y = np.array(obj_positions[i][1])
            obj_position_z = np.array(obj_positions[i][2])

            x_in_the_box = obj_position_x < -0.721 and obj_position_x > -0.937
            y_in_the_box = obj_position_y < 0.221 and obj_position_y > 0.005

            if x_in_the_box and y_in_the_box and obj_position_z > 0.25 and obj_position_z < 0.65:
                if obj_handles[i] not in self.obj_handles_to_remove:
                    if abs(obj_position_z) > 0.01:  # to deal with situations where [0,0,0] is returned
                        self.obj_handles_to_remove.append(obj_handles[i])
                        self.n_objects_reached_target += 1  # +1 object in the box
                    # else:
                    #     print('[0,0,0] returned')
        for i in range(len(obj_positions)):
            obj_position_x = np.array(obj_positions[i][0])
            obj_position_y = np.array(obj_positions[i][1])
            obj_position_z = np.array(obj_positions[i][2])

            # print(obj_position_x, obj_position_y, obj_position_z)

            obj_fell_0 = obj_position_x > -0.28
            obj_fell_1 = obj_position_y > 0.221
            obj_fell_2 = obj_position_x < -0.725 and obj_position_y < 0.005
            obj_fell_3 = obj_position_y < -0.22

            if (
                    obj_position_z < 0.25 or obj_fell_0 or obj_fell_1 or obj_fell_2 or obj_fell_3) and obj_position_z < 0.65:  # the object fell to the ground
                if obj_handles[i] not in self.obj_handles_to_remove:
                    if abs(obj_position_z) > 0.01:  # to deal with situations where [0,0,0] is returned
                        self.obj_handles_to_remove.append(obj_handles[i])
                        self.n_objects_fell_to_the_ground += 1  # +1 object on the ground
                        # print('Positions:')
                        # print(obj_position_x, obj_position_y, obj_position_z)
                        # input('Press Enter to continue')
                    # else:
                    #     print('[0,0,0] returned')
                    #     print(obj_position_x, obj_position_y, obj_position_z)
                    #     input('Press Enter to continue')

        return avg_current_dists, avg_next_dists, current_sum_True_bits, next_sum_True_bits, depth_diff

    def update_parameters(self, update_mask):
        transformed_pushingDataset = helper.PushingDataset(self.info_file, self.images_directory, self.mask_net, self.push_into_box_net, helper.ToTensor())
        train_loader = DataLoader(transformed_pushingDataset, batch_size=ud.train_batch_size, sampler=self.train_sampler)

        for i_batch, sample_batched in enumerate(train_loader):

            input_images_batch = sample_batched['input_image']
            actions = sample_batched['push_indexes']
            # Actual rewards for the push into box net
            rewards_push_into_box = sample_batched['reward_push_into_box']
            qs_max_next = sample_batched['q_max_next']
            # Given the discount factor, the actual rewards and the predictions for the next state,
            # compute the target values for the push into box net
            target_values_push_into_box = rewards_push_into_box.double() + self.discount_factor*qs_max_next
            loss_push_into_box = self.backprop_push_into_box(input_images_batch, actions, target_values_push_into_box, self.criterion_push_into_box)
            self.effective_loss_push_into_box += loss_push_into_box

            if update_mask:
                rewards_mask = sample_batched['reward_mask']
                loss_mask = self.backprop_mask(input_images_batch, actions, rewards_mask, self.criterion_mask)
                self.effective_loss_mask += loss_mask

        self.loss_hist_push_into_box.append(self.effective_loss_push_into_box.item())
        self.optimizer_push_into_box.step()
        self.optimizer_push_into_box.zero_grad()
        self.effective_loss_push_into_box = 0
        if self.model_updates_counter % 5 == 0:
            # save intermediate model
            torch.save(self.push_into_box_net.state_dict(), ud.path_to_new_push_into_box_model + 'pushing_success_predictor_model_%d.pth' % self.model_updates_counter)
            print('New push into box model saved')
        self.model_updates_counter += 1
        loss_hist_push_into_box_tensor = torch.Tensor(self.loss_hist_push_into_box)
        torch.save(loss_hist_push_into_box_tensor, ud.loss_hist_push_into_box_tensor_file)

        if update_mask:
            self.loss_hist_mask.append(self.effective_loss_mask.item())
            self.optimizer_mask.step()
            self.optimizer_mask.zero_grad()
            self.effective_loss_mask = 0
            # save intermediate model
            torch.save(self.mask_net.state_dict(), ud.path_to_new_mask_model)
            print('New mask model saved')
            loss_hist_mask_tensor = torch.Tensor(self.loss_hist_mask)
            torch.save(loss_hist_mask_tensor, ud.loss_hist_mask_tensor_file)

        print('Loss history saved')

    # Backpropagate mask
    def backprop_mask(self, input_images_batch, actions, rewards_mask, criterion):

        batch_size = len(input_images_batch)

        # Compute labels
        label_mask = torch.zeros(batch_size, 16, 224, 224).to(self.device)
        actions = [actions[0].to(torch.int64), actions[1].to(torch.int64), actions[2].to(torch.int64)]
        label_mask[range(batch_size), actions[0], actions[1], actions[2]] = rewards_mask.float().to(self.device)

        # Compute output mask
        out_mask = torch.zeros(batch_size, 16, 224, 224).to(self.device)
        out_mask[range(batch_size), actions[0], actions[1], actions[2]] = 1

        # Compute outputs
        input_images = input_images_batch.view(batch_size, 1, 224, 224).to(self.device)
        output_mask = self.mask_net(input_images)
        output_mask = output_mask * out_mask

        loss_mask = criterion(output_mask, label_mask)
        loss_mask = loss_mask.sum() / ud.batch_multiplier
        loss_mask.backward()

        return loss_mask

    # Backpropagate push into box
    def backprop_push_into_box(self, input_images_batch, actions, target_values_push_into_box, criterion):

        self.push_into_box_net.train()

        batch_size = len(input_images_batch)

        # Compute labels
        label_push_into_box = torch.zeros(batch_size, 16, 224, 224).to(self.device)
        actions = [actions[0].to(torch.int64), actions[1].to(torch.int64), actions[2].to(torch.int64)]
        label_push_into_box[
            range(batch_size), actions[0], actions[1], actions[2]] = target_values_push_into_box.float().to(self.device)

        # Compute output mask
        out_mask = torch.zeros(batch_size, 16, 224, 224).to(self.device)
        out_mask[range(batch_size), actions[0], actions[1], actions[2]] = 1

        # Compute outputs
        input_images = input_images_batch.view(batch_size, 1, 224, 224).to(self.device)
        output_push_into_box = self.push_into_box_net(input_images)
        output_push_into_box = output_push_into_box * out_mask

        loss_push_into_box = criterion(output_push_into_box, label_push_into_box)
        loss_push_into_box = loss_push_into_box.sum() / ud.batch_multiplier
        loss_push_into_box.backward()

        return loss_push_into_box