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

#!/usr/bin/env python

import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
import torch  
from torch.autograd import Variable
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import user_definitions as ud


def main(args, base_directory):

    # --------------- Setup options ---------------
    is_sim = args.is_sim # Run in simulation?
    obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj = args.num_obj if is_sim else None # Number of objects to add to simulation
    tcp_host_ip = args.tcp_host_ip if not is_sim else None # IP and port to robot arm as TCP client (UR5)
    tcp_port = args.tcp_port if not is_sim else None
    rtc_host_ip = args.rtc_host_ip if not is_sim else None # IP and port to robot arm as real-time client (UR5)
    rtc_port = args.rtc_port if not is_sim else None
    if is_sim:
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001+0.7, 0.4+0.7]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    else:
        workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    heightmap_resolution = args.heightmap_resolution # Meters per pixel of heightmap
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # ------------- Algorithm options -------------
    push_rewards = args.push_rewards  # Use immediate rewards (from change detection) for pushing?
    future_reward_discount = args.future_reward_discount
    experience_replay = args.experience_replay # Use prioritized experience replay?
    explore_rate_decay = args.explore_rate_decay

    # -------------- Training options --------------
    is_training = args.is_training
    training_preset_cases = args.training_preset_cases
    training_preset_folder = args.training_preset_folder

    # -------------- Testing options --------------
    is_testing = args.is_testing
    max_test_trials = args.max_test_trials # Maximum number of test runs per case/scenario
    test_preset_cases = args.test_preset_cases
    test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
    mask_snapshot_file = os.path.abspath(args.mask_snapshot_file) if load_snapshot else None
    push_into_box_snapshot_file = os.path.abspath(args.push_into_box_snapshot_file) if load_snapshot else None
    continue_logging = args.continue_logging # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath(base_directory+'logs/')
    save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True

    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Initialize system (camera and robot)
    robot = Robot(is_sim, base_directory, obj_mesh_dir, num_obj, workspace_limits,
                  tcp_host_ip, tcp_port, rtc_host_ip, rtc_port, is_training, training_preset_cases, training_preset_folder,
                  is_testing, test_preset_cases, test_preset_file)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory, is_testing)

    # Initialize trainer
    trainer = Trainer(is_training, is_testing, load_snapshot, mask_snapshot_file, push_into_box_snapshot_file,
                      force_cpu, robot, logger, ud.discount_factor)

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action' : False,
                          'chosen_pix_ind' : None,
                          'push_success' : False}

    scenario = 0
    n_actions = 0

    n_lines_in_info_for_prediction = 0

    img_idx = 0

    avg_current_dists = np.nan
    avg_next_dists = np.nan

    current_sum_True_bits = np.nan
    next_sum_True_bits = np.nan

    depth_diff = np.nan

    prev_valid_depth_heightmap = None
    prev_chosen_pix_ind = None

    n_objects_on_the_floor_before_action = np.nan
    n_objects_on_the_floor_after_action = np.nan

    n_objects_in_the_box_before_action = np.nan
    n_objects_in_the_box_after_action = np.nan

    n_objects_on_the_table = num_obj

    no_change_counter = 0

    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:

                # action_indexes = utils.select_action_at_random_given_mask(push_predictions)
                action_indexes = np.unravel_index(np.argmax(push_predictions),
                                                                      push_predictions.shape)

                # Get pixel location and rotation corresponding to action_indexes
                nonlocal_variables['chosen_pix_ind'] = action_indexes

                # Compute 3D position of pixel
                print('Action: %s at (%d, %d, %d)' % ('push', nonlocal_variables['chosen_pix_ind'][0], nonlocal_variables['chosen_pix_ind'][1], nonlocal_variables['chosen_pix_ind'][2]))
                chosen_rotation_angle = np.deg2rad(nonlocal_variables['chosen_pix_ind'][0]*(360.0/16))
                chosen_pix_x = nonlocal_variables['chosen_pix_ind'][2]
                chosen_pix_y = nonlocal_variables['chosen_pix_ind'][1]
                primitive_position = [chosen_pix_x * heightmap_resolution + workspace_limits[0][0], chosen_pix_y * heightmap_resolution + workspace_limits[1][0], valid_depth_heightmap[chosen_pix_y][chosen_pix_x] + workspace_limits[2][0]]
                primitive_position[2] -= 0.2  # because the table is 20cm above the floor in the preprocessed depth heightmap

                # Adjust start position, and make sure z value is safe and not too low
                finger_width = 0.02  # finger_width = 2cm
                safe_kernel_width = int(np.round((finger_width/2)/heightmap_resolution))  # safe_kernel_width = 5 pixels
                                                                                          # This accounts for the width of the finger.
                local_region = valid_depth_heightmap[max(chosen_pix_y - safe_kernel_width, 0):min(chosen_pix_y + safe_kernel_width + 1, valid_depth_heightmap.shape[0]), max(chosen_pix_x - safe_kernel_width, 0):min(chosen_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[1])]
                # local_region gives the heights of the pixels covered by the finger when it goes down
                # one needs to find the maximum height in this region to tell the robot how low it should go
                local_region -= 0.2  # because the table is 20cm above the floor in the preprocessed depth heightmap
                if local_region.size == 0:
                    safe_z_position = workspace_limits[2][0]
                else:
                    safe_z_position = np.max(local_region) + workspace_limits[2][0]
                primitive_position[2] = safe_z_position

                # Execute primitive
                nonlocal_variables['push_success'] = robot.push(primitive_position, chosen_rotation_angle, workspace_limits)
                print('Push successful: %r' % (nonlocal_variables['push_success']))

                nonlocal_variables['executing_action'] = False

            # time.sleep(0.01)
            time.sleep(1)
            # time.sleep(2)

    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False
    # -------------------------------------------------------------
    # -------------------------------------------------------------


    # Start main training/testing loop

    while True:

        iteration_time_0 = time.time()

        # # Make sure simulation is still stable (if not, reset simulation)
        # if is_sim: robot.check_sim()

        try:
            # Get latest RGB-D image
            color_img, depth_img = robot.get_camera_data()
            depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration
        except:
            continue

        try:
            # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
            color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics,
                                                                   robot.cam_pose, workspace_limits, heightmap_resolution)
        except:
            continue

        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[valid_depth_heightmap < 0.7] = 0.7  # 0.7 is the height of the table
        # Occlusions are automatically 0.
        # If there is some occlusion or very low point (below the table), just assign it first to the height of the table.

        # # Change depth_heightmap such that it's clear where the floor is
        # # (I am doing this in this way because I don't need the actual height)
        # # (It might be good to observe just a range of heights that actually matter)
        # valid_depth_heightmap[112:224, 0:2] = 0.7
        valid_depth_heightmap[0:2, :] = 0.5
        valid_depth_heightmap[222:224, :] = 0.5
        valid_depth_heightmap[0:112, 0:2] = 0.5
        valid_depth_heightmap[:, 222:224] = 0.5

        valid_depth_heightmap = valid_depth_heightmap - 0.5  # Shift everything such that the lowest point(the floor) is 0

        # Save RGB-D images and RGB-D heightmaps
        # logger.save_images(img_idx, color_img, depth_img, '0')
        logger.save_heightmaps(img_idx, color_heightmap, valid_depth_heightmap, '0')

        if not exit_called:

            # Run forward pass with network to get affordances
            push_predictions = trainer.forward(valid_depth_heightmap)
            push_predictions = push_predictions.view(16, 224, 224)
            push_predictions = push_predictions.detach().cpu().numpy()

            # Execute primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True
            n_actions += 1

        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)

        # Run training iteration in current thread (aka training thread)
        if prev_valid_depth_heightmap is not None:

            # The following numbers are with respect to the previous action, not the one that has just been executed.
            n_objects_in_the_box_after_action = trainer.n_objects_reached_target
            n_objects_on_the_floor_after_action = trainer.n_objects_fell_to_the_ground
            n_objects_on_the_table = num_obj - (n_objects_in_the_box_after_action + n_objects_on_the_floor_after_action)

            # Compute training labels
            avg_current_dists, avg_next_dists, current_sum_True_bits, next_sum_True_bits, depth_diff = trainer.get_label_value(valid_depth_heightmap, prev_valid_depth_heightmap)
            if depth_diff < 450:
                no_change_counter += 1

            # Save info in info_for_prediction if no part of the info is NaN
            array_sum = np.sum(np.array([avg_current_dists, n_objects_in_the_box_after_action-n_objects_in_the_box_before_action, n_objects_on_the_floor_after_action-n_objects_on_the_floor_before_action]))
            array_has_nan = np.isnan(array_sum)
            if not array_has_nan:
                # Save info for prediction
                logger.save_info_for_prediction(scenario, img_idx - 1, prev_chosen_pix_ind,
                                                img_idx, avg_current_dists, avg_next_dists, current_sum_True_bits, next_sum_True_bits, depth_diff,
                                                n_objects_in_the_box_after_action-n_objects_in_the_box_before_action, n_objects_on_the_floor_after_action-n_objects_on_the_floor_before_action)

                n_lines_in_info_for_prediction += 1
                print("Numer of lines in info for prediction = %d" % n_lines_in_info_for_prediction)

                if n_lines_in_info_for_prediction % (ud.train_batch_size*ud.batch_multiplier) == 0:
                    trainer.update_parameters(ud.update_mask)
                    logger.clear_info_for_prediction_minibatch()

            print(
                '****** ' + str(scenario) + ', ' + str(
                    img_idx - 1) + ', ' + str(
                    prev_chosen_pix_ind) + ', ' + str(
                    img_idx) + ', ' + str(avg_current_dists) + ', ' + str(avg_next_dists) +
                    ', ' + str(current_sum_True_bits) + ', ' + str(next_sum_True_bits) + ', ' + str(depth_diff) +
                    ', ' + str(n_objects_in_the_box_after_action-n_objects_in_the_box_before_action) + ', ' + str(n_objects_on_the_floor_after_action-n_objects_on_the_floor_before_action))

            print('Number of objects on the table = ', n_objects_on_the_table)

        if exit_called:
            logger.save_more_detail(mask_snapshot_file, push_into_box_snapshot_file)
            break

        # Save information for next training step
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_chosen_pix_ind = nonlocal_variables['chosen_pix_ind']
        n_objects_on_the_floor_before_action = n_objects_on_the_floor_after_action
        n_objects_in_the_box_before_action = n_objects_in_the_box_after_action

        img_idx += 1

        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))

        total_time_now = time.time()
        logger.save_elapsed_time(total_time_now - total_time_start)

        # input('Press Enter to continue.\n')

        # Reset simulation or pause real-world training if table is empty
        # n_actions >= 3 is to enter at least one line in info_for_prediction.txt about the current scenario
        # because each line is actually about the previous action, not the one that has just been executed
        if (n_objects_on_the_table == 0 and n_actions >= 3) or no_change_counter >=5 or n_actions >= 60:  # if there is not enough stuff or the number of actions is greater than or equal 60 (to move on if there are glitches)
            if is_sim:
                print('Not enough objects in view! Repositioning objects.')
                scenario += 1
                n_actions = 0
                no_change_counter = 0
                robot.restart_sim()
                robot.add_objects()
                trainer.n_objects_reached_target = 0
                trainer.n_objects_fell_to_the_ground = 0
                trainer.obj_handles_to_remove = []
                prev_valid_depth_heightmap = None
                prev_chosen_pix_ind = None
                avg_current_dists = np.nan
                avg_next_dists = np.nan
                current_sum_True_bits = np.nan
                next_sum_True_bits = np.nan
                depth_diff = np.nan
                n_objects_on_the_floor_before_action = np.nan
                n_objects_on_the_floor_after_action = np.nan
                n_objects_in_the_box_before_action = np.nan
                n_objects_in_the_box_after_action = np.nan
                n_objects_on_the_table = num_obj
            else:
                print('Not enough stuff on the table! Flipping over bin of objects...')
                robot.restart_real()

            continue


if __name__ == '__main__':

    total_time_start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to push several objects towards a desired position with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False,                                    help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default=ud.base_directory+'objects/blocks',   help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,                                help='number of objects to add to simulation')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30002,                           help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,                           help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,                      help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument('--push_rewards', dest='push_rewards', action='store_true', default=False,                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=ud.gamma)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,              help='use prioritized experience replay?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)

    # -------------- Training options --------------
    parser.add_argument('--is_training', dest='is_training', action='store_true', default=False)
    parser.add_argument('--training_preset_cases', dest='training_preset_cases', action='store_true', default=False)
    parser.add_argument('--training_preset_folder', dest='training_preset_folder', action='store',
                        default=ud.train_preset_folder)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,                help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default=ud.base_directory+'simulation/test-cases/test-10-obj-01.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--mask_snapshot_file', dest='mask_snapshot_file', action='store', default=ud.mask_snapshot_file)
    parser.add_argument('--push_into_box_snapshot_file', dest='push_into_box_snapshot_file', action='store',
                        default=ud.push_into_box_snapshot_file)
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')
    
    # Run main program with specified arguments
    args = parser.parse_args()
    main(args, ud.base_directory)
