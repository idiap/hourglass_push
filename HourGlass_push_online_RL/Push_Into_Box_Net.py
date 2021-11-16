################################################################################
################################################################################
#HOURGLASS_PUSH is a set of programs related to the paper "An Efficient Image-to-Image
#Translation HourGlass-based Architecture for Object Pushing Policy Learning",
#presented at IROS 2021.
#
#Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#Written by Angel Martinez-Gonzalez <angel.martinez@idiap.ch>,
#Modified by Marco Ewerton <marco.ewerton@idiap.ch>
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
import sys
import cv2
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


# body: input --> 1x1 Conv2d --> BatchNorm2d --> ReLU --> 3x3 Conv2d --> BatchNorm2d --> ReLU --> 1x1 Conv2d --> BatchNorm2d --> output
# shortcut: input --> 1x1 Conv2d --> output
# The ResidualBlock is illustrated in Fig. 4 (left) of the paper "Stacked Hourglass Networks for Human Pose Estimation" by Newell et al.
class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):  # input_channels is the number of input_channels
                                                          # output_channels is the number of output channels
        super(ResidualBlock, self).__init__()

        # // means "floor" division. It rounds down to the nearest whole number.
        # In this implementation, mid_channels is roughly the half of input_channels
        self.mid_channels = input_channels//2 if input_channels > 1 else output_channels

        layers=[nn.ReplicationPad2d(0),
          nn.Conv2d(input_channels, self.mid_channels, kernel_size=1, stride=1),
          nn.BatchNorm2d(self.mid_channels),
          nn.ReLU(inplace=True),

          nn.ReplicationPad2d(1),
          nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1),
          nn.BatchNorm2d(self.mid_channels),
          nn.ReLU(inplace=True),

          nn.ReplicationPad2d(0),
          nn.Conv2d(self.mid_channels, output_channels, kernel_size=1, stride=1),
          nn.BatchNorm2d(output_channels)]

        self.bn= nn.BatchNorm2d(output_channels)
        self.relu= nn.ReLU(inplace=True)

        self.body=nn.Sequential(*layers)  # the asterisk takes the elements inside of layers out of brackets.
                                          # In other words, the elements are being passed here as arguments,
                                          # not the list.

        self.shortcut = None
        # If input_channels != output_channels, a 1x1 conv is used to make sure the residual
        # and the main signal have the same number of channels.
        if input_channels != output_channels:
            self.shortcut = nn.Conv2d(in_channels=input_channels,\
                                      out_channels=output_channels,\
                                      kernel_size=1, \
                                      stride=1, \
                                      padding=0)

    def forward(self, x):
        res=x
        out= self.body(res)

        if self.shortcut is not None:
            res = self.shortcut(res)

        y= out+res  # elementwise addition of out and res

        # bn and relu have been defined such that we can combine the residual with the output before BatchNorm and ReLU
        y= self.bn(y)
        y= self.relu(y)

        # print('output of ResidualBlock: ', y.size())
        
        return y


class HourGlass(nn.Module):
    def __init__(self, params):
        super(HourGlass, self).__init__()

        self.params = params
        thisname = self.__class__.__name__
#         for k, v in self.params.items():
#             print('[INFO] ({}) {}: {}'.format(thisname, k, v))

        self.residual_block= ResidualBlock

        # This part implements the sequence of three bottlenecks with the lowest resolution.
        self.latent= nn.Sequential(*[self.residual_block(self.params['hg_across_channels'], self.params['hg_across_channels']),
                                     self.residual_block(self.params['hg_across_channels'], self.params['hg_across_channels']),
                                     self.residual_block(self.params['hg_across_channels'], self.params['hg_across_channels'])])

        self.front = nn.ModuleList()  # list of fronts (groups of three residual blocks)
        self.maxpools = nn.ModuleList()  # list of maxpools
        self.skip_connections = nn.ModuleList()  # list of skip_connections

        self.back = nn.ModuleList()  # list of backs (groups of one residual block)
        
        self.make_maxpoolings()
        self.make_fronts()
        self.make_skip_connections()
        
        self.make_backs()

    def make_maxpoolings(self):  # creates 4 maxpooling layers and put them in a list of maxpoolings
        for i in range(4):
            self.maxpools.append(nn.MaxPool2d(kernel_size=2, stride=2))

    def make_front_(self, n_input, n_output):  # creates a sequence with three residual blocks
                                               # because three residual blocks are used after each maxpooling
        layers= [self.residual_block(n_input, n_output),
                 self.residual_block(n_output, n_output),
                 self.residual_block(n_output, n_output)]

        return nn.Sequential(*layers)
    
    def make_back_(self, n_input, n_output):  # creates one residual block
                                              # because one residual block is used after each upsampling + residual addition
        layers= [self.residual_block(n_input, n_output)]

        return nn.Sequential(*layers)

    def make_fronts(self):  # creates the 4 groups of residual blocks
                            # 1 block for each maxpooling
        n_output= self.params['hg_across_channels']
        for i in range(4):
            n_input= self.params['hg_input_channels'] if i == 0 else self.params['hg_across_channels']
            self.front.append(self.make_front_(n_input, n_output))
            # n_input *= 2
            
    def make_backs(self):  # creates the 4 groups of 1 residual block
                            # 1 block for each upsampling + residual addition
        n_output= self.params['hg_across_channels']
        for i in range(4):
            n_input = self.params['hg_across_channels']
            self.back.append(self.make_back_(n_input, n_output))
            
    def make_skip_connection_(self, n_input):  # before each maxpooling, there is a branch
                                               # that goes to 2 residual blocks
        layers= [self.residual_block(n_input, n_input),
                 self.residual_block(n_input, n_input)]

        return nn.Sequential(*layers)

    def make_skip_connections(self):  # creates the 4 skip connections
        n_input= self.params['hg_across_channels']

        for i in range(4):
            # print('[INFO] Input size skip 1', n_input)
            self.skip_connections.append(self.make_skip_connection_(n_input))

    def forward(self, x):  # x: input image
        #### downsamplings
        skip_inputs = []
        out = x
        for i in range(4):
            out = self.front[i](out)
            skip_inputs.append(out)
            out = self.maxpools[i](out)
            # print('Down {} {}'.format(i,out.size()))

        ### them lowest resolution
        Z = self.latent(out)
        # print('Latent {}'.format(Z.size()))

        skip_outputs = []
        for i in range(4):
            skip = self.skip_connections[i](skip_inputs[i])
            skip_outputs.append(skip)
            # print('skip {} {}'.format(i,skip.size()))

        #### upsamplings
        up1 = F.interpolate(Z, scale_factor=2, mode='bilinear', align_corners=True)  # By default, mode='nearest'.
        # print(up1.size(), skip_outputs[-1].size())
        up = up1+skip_outputs[-1]
        
        # Apply a Residual Block
        up = self.back[-1](up)

        j = 2
        for i in range(3):
            up_ = F.interpolate(up, scale_factor=2, mode='bilinear', align_corners=True)
            up__ = skip_outputs[j]
            # print("{} {} ".format(up_.size(), up__.size()))
            up = up_ + up__
            # Apply a Residual Block
            up = self.back[j](up)
            j -= 1

        #print('output of HourGlass: ', up.size())
            
        return up

    
class Push_Into_Box_Net(nn.Module):
    def __init__(self, params):
        super(Push_Into_Box_Net, self).__init__()
        thisname = self.__class__.__name__
        self.params = params

#         for k, v in self.params.items():
#             print('[INFO] ({}) {}: {}'.format(thisname, k, v))

        ####
        self.residual_block = ResidualBlock

        # input_channels = 1
        # front_channels = 64
        # hg_input_channels = 128
        self.front = [nn.ReplicationPad2d(3),
                     nn.Conv2d(self.params['input_channels'], self.params['front_channels'], kernel_size=7, stride=1),
                     nn.BatchNorm2d(self.params['front_channels']),
                     nn.ReLU(inplace=True),
#                      nn.MaxPool2d(kernel_size=2, stride=2),
                     self.residual_block(self.params['front_channels'], self.params['front_channels']),
                     self.residual_block(self.params['front_channels'], self.params['front_channels']),
                     self.residual_block(self.params['front_channels'], self.params['hg_input_channels'])]

        self.back = [nn.ReplicationPad2d(3),
                     nn.Conv2d(self.params['hg_across_channels'], self.params['output_channels'], kernel_size=7, stride=1),
                     nn.BatchNorm2d(self.params['output_channels']),
                     nn.ReLU(), 
                     nn.ReplicationPad2d(3), 
                     nn.Conv2d(self.params['output_channels'], self.params['output_channels'], kernel_size=7, stride=1),                                  nn.BatchNorm2d(self.params['output_channels']), 
                     nn.ReLU()]
        
        self.front = nn.Sequential(*self.front)
        
        self.hg = nn.ModuleList()
        
        for i in range(self.params['n_stages']):
            self.hg.append(HourGlass(params))

        self.back = nn.Sequential(*self.back)
            
    def forward(self, x):
        # print('[INFO] Size 1 ', x.size())     nSamples x nChannels x Height x Width
        in_features = self.front(x)
        
        # print('in_features: ', in_features.size())
        
        for i in range(self.params['n_stages']):
               out = self.hg[i](in_features)
                
        out = self.back(out)
        
        # print('output of Push_Into_Box_Net: ', out.size())
        
        return out

    
def get_pibn_parameters():
    """
    Returns the default parameters used to generate the HG architecture.
    """
    # return {# channels produced by the first layers before HG
    #         'front_channels': 64,
    #         # channels input to the HG blocks
    #         'hg_input_channels': 128,
    #         # channels produced across the HG
    #         'hg_across_channels': 256,
    #         # how many hourglass modules
    #         'n_stages': 1,
    #         # channels of the image
    #         'input_channels':1,
    #         # output channels
    #         'output_channels':16}

    return {  # channels produced by the first layers before HG
        'front_channels': 32,
        # channels input to the HG blocks
        'hg_input_channels': 64,
        # channels produced across the HG
        'hg_across_channels': 128,
        # how many hourglass modules
        'n_stages': 1,
        # channels of the image
        'input_channels': 1,
        # output channels
        'output_channels': 16}
