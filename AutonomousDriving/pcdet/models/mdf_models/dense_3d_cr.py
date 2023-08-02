import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.utils import common_utils

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SEBlock(nn.Module):
    """ Squeeze-and-excitation block """
    def __init__(self, channels, r=16):
        super(SEBlock, self).__init__()
        self.r = r 
        self.squeeze = nn.Sequential(nn.Linear(channels, channels//self.r),
                                     nn.ReLU(),
                                     nn.Linear(channels//self.r, channels),
                                     nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.size()
        squeeze = self.squeeze(torch.mean(x, dim=(2,3))).view(B,C,1,1)
        return torch.mul(x, squeeze)

class DENSE_3D_DT(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.source_one_name = self.model_cfg.SOURCE_ONE_NAME
        self.per_task_channels = self.model_cfg.INPUT_CONV_CHANNEL

        # SEBlock
        self.se_s1 = SEBlock(self.per_task_channels)
        self.se_s2 = SEBlock(self.per_task_channels)

    def forward(self, data_dict):
        # Get shared representation
        split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, data_dict)
        spatial_features_2d = data_dict['spatial_features'] 

        spatial_features_2d_s1 = spatial_features_2d[split_tag_s1,:,:,:]
        spatial_features_2d_s2 = spatial_features_2d[split_tag_s2,:,:,:]

        out_s1 = self.se_s1(spatial_features_2d_s1)
        out_s2 = self.se_s2(spatial_features_2d_s2)

        concat_f = torch.cat([out_s1, out_s2], 0)

        data_dict['spatial_features'] = concat_f

        return data_dict