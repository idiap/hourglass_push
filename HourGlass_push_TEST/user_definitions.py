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

vrep_port = 19994
vrep_ip = '172.20.21.7'

base_directory = '<PATH TO THE FOLDER HourGlass_push_TEST>'

trained_mask_model_folder = base_directory + 'init_mask_model/'
snapshot_file_str_mask = trained_mask_model_folder + 'mask_model.pt'

trained_push_into_box_model_folder = base_directory + 'init_push_into_box_model/'
snapshot_file_str_push_into_box = trained_push_into_box_model_folder + 'push_into_box_model.pt'

logging_directory = base_directory + 'TESTS/'

test_preset_folder = '<PATH TO THE FOLDER containing test cases>'

pos_encoding = None
# pos_encoding = 'linear'
