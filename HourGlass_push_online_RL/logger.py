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
import torch 
# import h5py 

class Logger():

    def __init__(self, continue_logging, logging_directory, is_testing):

        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        self.continue_logging = continue_logging
        if self.continue_logging:
            self.base_directory = logging_directory
            print('Pre-loading data logging session: %s' % (self.base_directory))
        else:
            self.base_directory = os.path.join(logging_directory, timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
            print('Creating data logging session: %s' % (self.base_directory))
        self.info_directory = os.path.join(self.base_directory, 'info')
        self.color_images_directory = os.path.join(self.base_directory, 'data', 'color-images')
        self.depth_images_directory = os.path.join(self.base_directory, 'data', 'depth-images')
        self.color_heightmaps_directory = os.path.join(self.base_directory, 'data', 'color-heightmaps')
        self.depth_heightmaps_directory = os.path.join(self.base_directory, 'data', 'depth-heightmaps')
        self.models_directory = os.path.join(self.base_directory, 'models')
        self.visualizations_directory = os.path.join(self.base_directory, 'visualizations')
        self.recordings_directory = os.path.join(self.base_directory, 'recordings')
        self.transitions_directory = os.path.join(self.base_directory, 'transitions')

        if not os.path.exists(self.info_directory):
            os.makedirs(self.info_directory)
        if not os.path.exists(self.color_images_directory):
            os.makedirs(self.color_images_directory)
        if not os.path.exists(self.depth_images_directory):
            os.makedirs(self.depth_images_directory)
        if not os.path.exists(self.color_heightmaps_directory):
            os.makedirs(self.color_heightmaps_directory)
        if not os.path.exists(self.depth_heightmaps_directory):
            os.makedirs(self.depth_heightmaps_directory)
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)
        if not os.path.exists(self.visualizations_directory):
            os.makedirs(self.visualizations_directory)
        if not os.path.exists(self.recordings_directory):
            os.makedirs(self.recordings_directory)
        if not os.path.exists(self.transitions_directory):
            os.makedirs(os.path.join(self.transitions_directory, 'data'))

        # File to write info for prediction
        self.fh = open(os.path.join(self.base_directory, 'info_for_prediction.txt'), 'w')
        self.fh.write('scenario, input_img, action, next_img, avg_dists_before_action, avg_dists_after_action, current_sum_True_bits, next_sum_True_bits, depth_diff, n_new_objects_reached_target, n_new_objects_fell_to_the_ground\n')
        self.fh.close()
        self.fh = open(os.path.join(self.base_directory, 'info_for_prediction_minibatch.txt'), 'w')
        self.fh.write(
            'scenario, input_img, action, next_img, avg_dists_before_action, avg_dists_after_action, current_sum_True_bits, next_sum_True_bits, depth_diff, n_new_objects_reached_target, n_new_objects_fell_to_the_ground\n')
        self.fh.close()

        # File with more details about the training or the test
        self.fh = open(os.path.join(self.base_directory, 'more_details.txt'), 'w')
        if is_testing:
            self.fh.write('Test\n\n')
        else:
            self.fh.write('Training\n\n')
        self.fh.write('Start: ' + timestamp_value.strftime('%Y-%m-%d.%H:%M:%S') + '\n\n')
        self.fh.write('Pushes in 16 different orientations' + '\n\n')
        self.fh.close()

    def save_info_for_prediction(self, scenario, input_img, action, next_img, avg_dists_before_action, avg_dists_after_action, current_sum_True_bits, next_sum_True_bits, depth_diff, n_new_objects_reached_target, n_new_objects_fell_to_the_ground):
        self.fh = open(os.path.join(self.base_directory, 'info_for_prediction.txt'), 'a')
        self.fh.write(str(scenario) + ', ' + str(input_img) + ', ' + str(action) + ', ' + str(next_img) + ', ' + str(avg_dists_before_action) + ', ' + str(avg_dists_after_action) + ', ' + str(current_sum_True_bits) + ', ' + str(next_sum_True_bits) + ', ' + str(depth_diff) + ', ' + str(n_new_objects_reached_target) + ', ' + str(n_new_objects_fell_to_the_ground) + '\n')
        self.fh.close()
        self.fh = open(os.path.join(self.base_directory, 'info_for_prediction_minibatch.txt'), 'a')
        self.fh.write(str(scenario) + ', ' + str(input_img) + ', ' + str(action) + ', ' + str(next_img) + ', ' + str(
            avg_dists_before_action) + ', ' + str(avg_dists_after_action) + ', ' + str(
            current_sum_True_bits) + ', ' + str(next_sum_True_bits) + ', ' + str(depth_diff) + ', ' + str(
            n_new_objects_reached_target) + ', ' + str(n_new_objects_fell_to_the_ground) + '\n')
        self.fh.close()

    def clear_info_for_prediction_minibatch(self):
        self.fh = open(os.path.join(self.base_directory, 'info_for_prediction_minibatch.txt'), 'w')
        self.fh.write(
            'scenario, input_img, action, next_img, avg_dists_before_action, avg_dists_after_action, current_sum_True_bits, next_sum_True_bits, depth_diff, n_new_objects_reached_target, n_new_objects_fell_to_the_ground\n')
        self.fh.close()

    def save_more_detail(self, mask_snapshot_file, push_into_box_snapshot_file):
        self.fh = open(os.path.join(self.base_directory, 'more_details.txt'), 'a')
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        self.fh.write('Using mask model: ' + mask_snapshot_file + '\n\n')
        self.fh.write('Using push into box model: ' + push_into_box_snapshot_file + '\n\n')
        self.fh.write('End: ' + timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
        self.fh.close()

    def save_elapsed_time(self, total_time_elapsed):
        self.fh = open(os.path.join(self.base_directory, 'total_elapsed_time.txt'), 'w')
        self.fh.write(str(total_time_elapsed))
        self.fh.close()

    def save_camera_info(self, intrinsics, pose, depth_scale):
        np.savetxt(os.path.join(self.info_directory, 'camera-intrinsics.txt'), intrinsics, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'camera-pose.txt'), pose, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'camera-depth-scale.txt'), [depth_scale], delimiter=' ')

    def save_heightmap_info(self, boundaries, resolution):
        np.savetxt(os.path.join(self.info_directory, 'heightmap-boundaries.txt'), boundaries, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'heightmap-resolution.txt'), [resolution], delimiter=' ')

    def save_images(self, img_idx, color_image, depth_image, mode):
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.color_images_directory, '%06d.%s.color.png' % (img_idx, mode)), color_image)
        depth_image = np.round(depth_image * 10000).astype(np.uint16) # Save depth in 1e-4 meters
        cv2.imwrite(os.path.join(self.depth_images_directory, '%06d.%s.depth.png' % (img_idx, mode)), depth_image)
    
    def save_heightmaps(self, img_idx, color_heightmap, depth_heightmap, mode):
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.color_heightmaps_directory, '%06d.%s.color.png' % (img_idx, mode)), color_heightmap)
        depth_heightmap = np.round(depth_heightmap * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.depth_heightmaps_directory, '%06d.%s.depth.png' % (img_idx, mode)), depth_heightmap)
    
    def write_to_log(self, log_name, log):
        np.savetxt(os.path.join(self.transitions_directory, '%s.log.txt' % log_name), log, delimiter=' ')
