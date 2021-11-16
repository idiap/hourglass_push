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
import threading
import numpy as np
import torch
from robot import Robot
from trainer import Trainer
import user_definitions as ud
from logger_test import Logger
import utils

############################################################################
def main(base_directory, trained_mask_model_folder, checkpoints_folder, num_obj, scenario):

    base_directory = base_directory

    # --------------- Setup options ---------------
    is_sim = True  # Run in simulation?
    obj_mesh_dir = os.path.abspath(base_directory+'objects/blocks') if is_sim else None # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj = num_obj if is_sim else None # Number of objects to add to simulation
    tcp_host_ip = None # IP and port to robot arm as TCP client (UR5)
    tcp_port = None
    rtc_host_ip = None # IP and port to robot arm as real-time client (UR5)
    rtc_port = None
    if is_sim:
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001+0.7, 0.4+0.7]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    else:
        workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    heightmap_resolution = 0.002 # Meters per pixel of heightmap
    random_seed = 1234
    force_cpu = False

    # ------------- Algorithm options -------------
    push_rewards = True  # Use immediate rewards (from change detection) for pushing?
    future_reward_discount = 0.0

    # -------------- Training options --------------
    is_training = False
    training_preset_cases = False
    training_preset_folder = None

    # -------------- Testing options --------------
    is_testing = True
    test_preset_cases = True
    test_preset_file_str = ud.test_preset_folder + 'my_test_case_%d.txt' % scenario
    test_preset_file = os.path.abspath(test_preset_file_str) if test_preset_cases else None

    snapshot_file_mask = os.path.abspath(ud.snapshot_file_str_mask)

############################################################################

    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Initialize system (camera and robot)
    robot = Robot(is_sim, base_directory, obj_mesh_dir, num_obj, workspace_limits,
                  tcp_host_ip, tcp_port, rtc_host_ip, rtc_port, is_training, training_preset_cases, training_preset_folder,
                  is_testing, test_preset_cases, test_preset_file)

    # Initialize data logger
    logger = Logger(ud.logging_directory, is_testing, future_reward_discount, num_obj, scenario)

    # Initialize trainer
    trainer = Trainer(is_training, is_testing, ud.snapshot_file_str_mask, ud.snapshot_file_str_push_into_box, force_cpu, robot, logger, ud.pos_encoding)

    no_change_count = 0

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action': False,
                          'chosen_pix_ind': None,
                          'push_success': False,
                          'change_detected': False}


    # scenario = 0

    n_actions = 0

    img_idx = 0

    test_iteration = 0

    change_threshold = 100

    avg_current_dists = np.nan
    avg_next_dists = np.nan

    prev_depth_heightmap = None
    prev_valid_depth_heightmap = None

    prev_chosen_pix_ind = None

    prev_push_predictions = None

    previous_5_actions = []


    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:

                # print("push_predictions shape = %s" % str(push_predictions.shape))
                # best_push_conf = np.max(push_predictions)
                # print('Primitive confidence scores: %f (push)' % best_push_conf)

                # All the Values corresponding to the previous 5 actions are 0
                for element in previous_5_actions:
                    print(element)
                    push_predictions[element] = 0.0

                # Get pixel location and rotation with highest affordance prediction
                nonlocal_variables['chosen_pix_ind'] = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
                print('Max push_predictions: ', np.max(push_predictions))
                # input('Press Enter to continue')

                # Put new element in previous_5_actions
                if len(previous_5_actions) < 5:
                    previous_5_actions.append(nonlocal_variables['chosen_pix_ind'])
                else:
                    del previous_5_actions[0]
                    previous_5_actions.append(nonlocal_variables['chosen_pix_ind'])


                # Compute 3D position of pixel
                print('Action: %s at (%d, %d, %d)' % (
                'push', nonlocal_variables['chosen_pix_ind'][0], nonlocal_variables['chosen_pix_ind'][1],
                nonlocal_variables['chosen_pix_ind'][2]))
                chosen_rotation_angle = np.deg2rad(nonlocal_variables['chosen_pix_ind'][0] * (360.0 / 16))
                chosen_pix_x = nonlocal_variables['chosen_pix_ind'][2]
                chosen_pix_y = nonlocal_variables['chosen_pix_ind'][1]
                primitive_position = [chosen_pix_x * heightmap_resolution + workspace_limits[0][0],
                                      chosen_pix_y * heightmap_resolution + workspace_limits[1][0],
                                      valid_depth_heightmap[chosen_pix_y][chosen_pix_x] + workspace_limits[2][0]]
                primitive_position[
                    2] -= 0.2  # because the table is 20cm above the floor in the preprocessed depth heightmap

                # Adjust start position, and make sure z value is safe and not too low
                finger_width = 0.02  # finger_width = 2cm
                safe_kernel_width = int(
                    np.round((finger_width / 2) / heightmap_resolution))  # safe_kernel_width = 5 pixels
                # This accounts for the width of the finger.
                local_region = valid_depth_heightmap[
                               max(chosen_pix_y - safe_kernel_width, 0):min(chosen_pix_y + safe_kernel_width + 1,
                                                                            valid_depth_heightmap.shape[0]),
                               max(chosen_pix_x - safe_kernel_width, 0):min(chosen_pix_x + safe_kernel_width + 1,
                                                                            valid_depth_heightmap.shape[1])]
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

            time.sleep(1)
    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False
    # -------------------------------------------------------------
    # -------------------------------------------------------------


    # Start main training/testing loop
    while True:
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', test_iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim: robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale  # Apply depth scale from calibration

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)

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

        # Reset simulation or pause real-world training if table is empty
        if len(trainer.obj_handles_to_remove) == num_obj or is_sim and no_change_count >= 5 or n_actions >= 60:  # if there is not enough stuff or nothing changed after a certain number of actions
            no_change_count = 0
            if is_sim:
                print('Not enough objects in view! Repositioning objects.')
                logger.save_test_info(scenario, img_idx-1, prev_chosen_pix_ind, img_idx, test_iteration, np.nan, np.nan, trainer.n_objects_reached_target, trainer.n_objects_fell_to_the_ground, n_actions, np.nan, np.nan, np.nan, np.nan)
                print(
                    '****** ' + str(scenario) + ', ' + str(img_idx-1) + ', ' + str(prev_chosen_pix_ind) + ', ' + str(img_idx) + ', ' + str(test_iteration) + ', ' + str(
                        np.nan) + ', ' + str(
                        np.nan) + ', ' + str(trainer.n_objects_reached_target) + ', ' + str(trainer.n_objects_fell_to_the_ground) + ', ' + str(n_actions) + ', ' + str(np.nan) + ', ' + str(np.nan) + ', ' + str(np.nan) + ', ' + str(np.nan))
                scenario += 1
                n_actions = 0
                # robot.restart_sim()
                # robot.add_objects()
                trainer.n_objects_reached_target = 0
                trainer.obj_handles_to_remove = []
                prev_valid_depth_heightmap = None

            else:
                print('Not enough stuff on the table! Flipping over bin of objects...')
                robot.restart_real()

            if is_testing:
                exit_called = True
            continue

        logger.save_heightmaps(img_idx, color_heightmap, valid_depth_heightmap, '0')

        if not exit_called:

            # Run forward pass with network to get affordances
            push_predictions = trainer.forward(valid_depth_heightmap)
            push_predictions = push_predictions.view(16, 224, 224)
            push_predictions = push_predictions.detach().cpu().numpy()

            # Execute best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True
            n_actions += 1

        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)

        # Run training iteration in current thread (aka training thread)
        if prev_valid_depth_heightmap is not None:

            # Detect changes
            depth_diff = abs(valid_depth_heightmap - prev_valid_depth_heightmap)
            depth_diff[np.isnan(depth_diff)] = 0
            depth_diff[depth_diff > 0.3] = 0
            depth_diff[depth_diff < 0.01] = 0
            depth_diff[depth_diff > 0] = 1
            change_value = np.sum(depth_diff)
            nonlocal_variables['change_detected'] = change_value > change_threshold
            change_detected = change_value > change_threshold
            print('Change detected: %r (value: %d)' % (change_detected, change_value))

            if change_detected:
                no_change_count = 0
            else:
                no_change_count += 1

            # Compute training labels
            avg_current_dists, avg_next_dists, current_sum_True_bits, next_sum_True_bits, depth_diff = trainer.get_label_value(valid_depth_heightmap, prev_valid_depth_heightmap)

            # Save info about the test for comparisons
            logger.save_test_info(scenario, img_idx-1, prev_chosen_pix_ind, img_idx, test_iteration, avg_current_dists, avg_next_dists, trainer.n_objects_reached_target, trainer.n_objects_fell_to_the_ground, n_actions, 0, 0, 0, 0)
            print(
                '****** ' + str(scenario) + ', ' + str(img_idx-1) + ', ' + str(prev_chosen_pix_ind) + ', ' + str(img_idx) + ', ' + str(test_iteration) + ', ' + str(avg_current_dists) + ', ' + str(
                    avg_next_dists) + ', ' + str(trainer.n_objects_reached_target) + ', ' + str(trainer.n_objects_fell_to_the_ground) + ', ' + str(n_actions) + ', ' + str(0) + ', ' + str(0) + ', ' + str(0) + ', ' + str(0))
            logger.save_q(prev_push_predictions, img_idx-1)

            test_iteration += 1

        if exit_called:
            logger.save_more_detail(snapshot_file_mask, ud.snapshot_file_str_push_into_box)
            break

        # Save information for next step
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_chosen_pix_ind = nonlocal_variables['chosen_pix_ind']
        prev_push_predictions = push_predictions

        img_idx += 1

        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))