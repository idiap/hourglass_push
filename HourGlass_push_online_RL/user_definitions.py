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

base_directory = '/idiap/temp/mewerton/Idiap_2020/HOURGLASS_PUSH/HOURGLASS_PUSH_CODE/HourGlass_push_online_RL/'

train_batch_size = 5
batch_multiplier = 20

path_to_new_mask_model = ''  # The mask model is not being updated at the moment but it can be update as well.
path_to_new_push_into_box_model = base_directory + 'trained_models/new/push_into_box/'

loss_hist_mask_tensor_file = base_directory + 'loss_history/loss_hist_mask_tensor.pt'
loss_hist_push_into_box_tensor_file = base_directory + 'loss_history/loss_hist_push_into_box_tensor.pt'

learning_curve_mask_folder = base_directory + 'learning_curves/mask/'
learning_curve_push_into_box_folder = base_directory + 'learning_curves/push_into_box/'

# This is a boolean to decide whether we want to refine the mask in RL or not
update_mask = False

#######################
discount_factor = 0.0
use_mask = True
vrep_port = 19994
vrep_ip = '172.20.21.7'
#######################
# Mask Probability Threshold
mask_prob_threshold = 0.14

gamma = 0.0

# Models used as initialization when running online RL
mask_snapshot_file = base_directory + '/trained_models/init/mask/mask_model.pt'
push_into_box_snapshot_file = base_directory + '/trained_models/init/push_into_box/push_into_box_model.pt'

train_preset_folder = '/idiap/temp/mewerton/Idiap_2020/HOURGLASS_PUSH/DATA/training_scenarios/10_objects/'
