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

import numpy as np
import torch
import PushingSuccessPredictor as psp
import Push_Into_Box_Net as pibn
from numpy import linalg as LA


class Trainer(object):
    def __init__(self, is_training, is_testing, mask_snapshot_file, checkpoint_file, force_cpu, robot, logger, encoding_type=None):

        self.encoding_type = encoding_type

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
        mask_checkpoint = torch.load(mask_snapshot_file)
        self.mask_net.load_state_dict(mask_checkpoint['mask_model_state_dict'])
        print('Pre-trained mask model snapshot loaded from: %s' % (mask_snapshot_file))

        push_into_box_checkpoint = torch.load(checkpoint_file)
        self.push_into_box_net.load_state_dict(push_into_box_checkpoint['push_into_box_model_state_dict'])
        print('Pre-trained push into box model snapshot loaded from: %s' % (checkpoint_file))

        torch.no_grad()
        self.mask_net.eval()
        self.push_into_box_net.eval()

        self.target = [168., 0.]  # This is the position in pixel space of the middle of the edge of the box
        # in contact with the table
        self.obj_handles_to_remove = []

        # Precompute the distance between each pixel and the target
        self.resolution = 224  # Assuming number of rows = number of columns = resolution
        img_indices = np.indices((self.resolution, self.resolution))
        img_indices = np.vstack((img_indices[0].flatten(), img_indices[1].flatten())).transpose()
        self.pixel_distances = LA.norm(img_indices - self.target, axis=1)
        self.pixel_distances = self.pixel_distances.reshape((self.resolution, self.resolution))

        self.u_coords = None
        self.v_coords = None

        if self.encoding_type == 'linear':
            # Precompute the u,v normalized coordinate (between -1 and 1) of each pixel with respect to a position of interest on the box
            img_indices = np.indices((self.resolution, self.resolution))
            self.u_coords = (img_indices[0] - self.target[0]) / self.resolution
            self.v_coords = (img_indices[1] - self.target[1]) / self.resolution

    # Compute forward pass through model to compute affordances/Q
    def forward(self, depth_heightmap):

        # If using relative position encoding:
        if self.encoding_type is not None:
            copy_depth_heightmap = depth_heightmap.copy()
            copy_depth_heightmap = torch.from_numpy(copy_depth_heightmap).float()
            copy_depth_heightmap = copy_depth_heightmap.view(1, 1, 224, 224).to(self.device)
            mask = self.mask_net(copy_depth_heightmap)
            print('**********')
            print(depth_heightmap.shape)
            print(self.u_coords.shape)
            print(self.v_coords.shape)
            depth_heightmap = np.stack((depth_heightmap, self.u_coords, self.v_coords))
            print('$$$$$$$$$$')
            depth_heightmap = torch.from_numpy(depth_heightmap).float()
            depth_heightmap = depth_heightmap.view(1, 3, 224, 224).to(self.device)
            push_into_box = self.push_into_box_net(depth_heightmap)

        # If not using relative position encoding:
        if self.encoding_type == None:
            depth_heightmap = torch.from_numpy(depth_heightmap).float()
            depth_heightmap = depth_heightmap.view(1, 1, 224, 224).to(self.device)
            mask = self.mask_net(depth_heightmap)
            push_into_box = self.push_into_box_net(depth_heightmap)

        print('Mask max: ', torch.max(mask))
        print('Push into box max: ', torch.max(push_into_box))
        print('Push into box min: ', torch.min(push_into_box))

        mask = mask >= 0.14  # True wherever A >= 0.14, False otherwise.
        mask = mask.float()

        push_predictions = push_into_box * mask
        # push_predictions = push_into_box

        print('Push predictions max: ', torch.max(push_predictions))
        print('Push predictions min: ', torch.min(push_predictions))

        return push_predictions

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