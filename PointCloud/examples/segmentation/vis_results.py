#!/usr/bin/env python
# coding: utf-8
import __init__
import os
from openpoints.dataset.vis3d import read_obj, vis_multi_points


# --------------------------------
idx = 0 
data_dir = "pretrained/pix4point/mae-s/visualization"
dataset_name = 's3dissphere'
roof_height = 3
# --------------------------------

method_names = ['input', 'pix4point', 'gt']
file_list = []
colors_list = []
for i, method_name in enumerate(method_names):
    file_path = os.path.join(data_dir, f'{method_name}-{dataset_name}-{idx}.obj')
    points, colors =read_obj(file_path)
    if i == 0: # input
        # remove roof
        valid_idx = points[:, 2] < roof_height 
        input_points = points[valid_idx] 
        colors_list.append(colors[valid_idx]/255.) 
    else:
        colors_list.append(colors[valid_idx]) 

points_list = [input_points] * len(method_names)
vis_multi_points(points_list, colors_list)