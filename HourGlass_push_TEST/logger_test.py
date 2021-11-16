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

import time
import datetime
import os
import numpy as np
import cv2


# import h5py

class Logger():

    def __init__(self, logging_directory, is_testing, future_reward_discount, num_obj, scenario):

        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)

        self.base_directory = os.path.join(logging_directory, "n_objects_%d_scenario_%d" % (num_obj, scenario))
        print('Creating data logging session: %s' % (self.base_directory))

        self.color_heightmaps_directory = os.path.join(self.base_directory, 'data', 'color-heightmaps')
        self.depth_heightmaps_directory = os.path.join(self.base_directory, 'data', 'depth-heightmaps')
        self.qs_directory = os.path.join(self.base_directory, 'q_values')

        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)
        if not os.path.exists(self.color_heightmaps_directory):
            os.makedirs(self.color_heightmaps_directory)
        if not os.path.exists(self.depth_heightmaps_directory):
            os.makedirs(self.depth_heightmaps_directory)
        if not os.path.exists(self.qs_directory):
            os.makedirs(self.qs_directory)

        # File to write info about a test
        self.fh = open(os.path.join(self.base_directory, 'test_info.txt'), 'w')
        self.fh.write(
            'scenario, input_img, action, next_img, testing iteration, avg_current_dists, avg_next_dists, n_objs_reached_target, n_objects_fell_to_the_ground, n_actions, expected_reward, current_reward, reward_due_to_change, reward_due_to_obj_in_box\n')
        self.fh.close()

        # File with more details about the training or the test
        self.fh = open(os.path.join(self.base_directory, 'more_details.txt'), 'w')
        if is_testing:
            self.fh.write('Test\n\n')
        else:
            self.fh.write('Training\n\n')
        self.fh.write('Start: ' + timestamp_value.strftime('%Y-%m-%d.%H:%M:%S') + '\n\n')
        self.fh.write('Pushes in 16 different orientations' + '\n\n')
        self.fh.write('Discount factor: %f' % future_reward_discount + '\n\n')
        self.fh.close()

    def save_test_info(self, scenario, input_img, action, next_img, iteration, avg_current_dists, avg_next_dists, n_objs_reached_target, n_objects_fell_to_the_ground, n_actions,
                       expected_reward, current_reward, reward_due_to_change, reward_due_to_obj_in_box):
        self.fh = open(os.path.join(self.base_directory, 'test_info.txt'), 'a')
        self.fh.write(str(scenario) + ', ' + str(input_img) + ', ' + str(action) + ', ' + str(next_img) + ', ' + str(iteration) + ', ' + str(avg_current_dists) + ', ' + str(
            avg_next_dists) + ', ' + str(n_objs_reached_target) + ', ' + str(n_objects_fell_to_the_ground) + ', ' + str(n_actions) + ', ' + str(
            expected_reward) + ', ' + str(current_reward) + ', ' + str(reward_due_to_change) + ', ' + str(reward_due_to_obj_in_box) + '\n')
        self.fh.close()

    def save_q(self, prev_push_predictions, input_img):
        file_name = self.qs_directory + '/Q_%d.npy'%input_img
        np.save(file_name, prev_push_predictions)

    def save_more_detail(self, snapshot_file_mask, snapshot_file_push_into_box):
        self.fh = open(os.path.join(self.base_directory, 'more_details.txt'), 'a')
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        self.fh.write('Using mask model: ' + snapshot_file_mask + '\n\n')
        self.fh.write('Using push into box model: ' + snapshot_file_push_into_box + '\n\n')
        self.fh.write('End: ' + timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
        self.fh.close()

    def save_heightmaps(self, img_idx, color_heightmap, depth_heightmap, mode):
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.color_heightmaps_directory, '%06d.%s.color.png' % (img_idx, mode)), color_heightmap)
        depth_heightmap = np.round(depth_heightmap * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.depth_heightmaps_directory, '%06d.%s.depth.png' % (img_idx, mode)), depth_heightmap)
