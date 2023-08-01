import os
import copy
import pickle
from collections import defaultdict
import json

import numpy as np
from pathlib import Path
import argparse
import torch

from utils import Visualizer, LabelLUT

from utils.base_dataset import DataCollect
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu




def sequence_visualize3d(**infos):
    data_collect = DataCollect(color_attr=[
                                          "class",
                                        #   "id"
                                          ], 
                               text_attr=[
                                        #   "class", 
                                        #   "id",
                                        #   "score",
                                           ], 
                               show_text=True)
    
    data_collect.offline_process_infos(**infos)

    lut = LabelLUT()
    lut_labels = {
        "track": [1., 1., 1.],
        "gt": [1., 0., 0.],
        "detect": [0., 1., 0.],
        "detect_pro": [0.7, 0.2, 0.7],
    }
    lut_labels = {
        "gt_Car": [0., 1., 0.], # once
        "gt_Truck": [0., 1., 0.],
        "gt_Bus": [0., 1., 0.],
        "gt_Pedestrian": [0., 0., 1.],
        "gt_Cyclist": [1., 0.0, 0.0],
        "gt_car": [0., 1., 0.], # nuscenes
        "gt_traffic_cone": [1.0, 1.0, 0.25],
        "gt_truck": [0., 1., 0.],
        "gt_pedestrian": [0., 0., 1.0],
        "gt_construction_vehicle": [0., 1., 0.],
        "gt_bus": [0., 1., 0.],
        "gt_trailer": [0., 0.68627451, 0.],
        "gt_motorcycle": [1., 0., 0.],
        "gt_bicycle": [1., 0., 0.],
        "gt_barrier": [0.19607843, 0.47058824, 1.],
    }
    for key, val in lut_labels.items():
        lut.add_label(key, key, val)
    # lut = None
    _3dal_vis = Visualizer(fps=4)
    _3dal_vis.visualize_dataset(data_collect, prefix="frame id", lut=lut)


def load_once(data_path, seq_id):
    info_path = os.path.join(data_path, seq_id)
    annos_path = os.path.join(info_path, seq_id + '.json')

    frame_ids_list = list()
    pts_list = list()
    pts_label_list = list()
    gt_list = list()

    with open(annos_path, 'r') as f:
        annos = json.load(f)
        frames = annos['frames'][:3]  # We only put three once frames here as an example.
        for frame in frames:
            if 'annos' in frame.keys():
                sequence_id = frame['sequence_id']
                frame_id = frame['frame_id']
                pose = frame['pose']
                annos = frame['annos']
                names = annos['names']
                boxes_3d = np.array(annos['boxes_3d'])

                frame_ids_list.append(frame_id)
                bin_path = os.path.join(info_path, 'lidar_roof', '{}.bin'.format(frame_id))
                points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
                pts_list.append(points)

                gt_list.append(
                {
                    "bbox": boxes_3d,
                    "class": names,
                })

                box_idxs = points_in_boxes_gpu(
                    torch.from_numpy(points).unsqueeze(dim=0).float().cuda(),
                    torch.from_numpy(boxes_3d).unsqueeze(dim=0).float().cuda()
                ).long().squeeze(dim=0).cpu().numpy()
                pts_label_list.append(box_idxs) 

    info = {
        "idx_names": frame_ids_list,
        "pts": pts_list,
        "pts_label": pts_label_list,
        "gt": gt_list,
    }
    return info

def load_nuscenes(data_path, seq_id):
    info_path = os.path.join(data_path, 'nuscenes_infos_10sweeps_train.pkl')
    annos = pickle.load(open(info_path, "rb"))
    frame_ids_list = list()
    pts_list = list()
    pts_label_list = list()
    gt_list = list()
    for anno in annos:
        lidar_path = anno['lidar_path']
        cur_seq_name = lidar_path.split("__LIDAR_TOP__")[0].split("LIDAR_TOP/")[-1]
        if cur_seq_name != seq_id:
            continue

        gt_names = anno['gt_names']
        gt_boxes = anno['gt_boxes'][:,:7]
        frame_id = lidar_path.split('_')[-1].strip('.pcd.bin')
        
        bin_path = os.path.join(data_path, lidar_path)
        points = np.fromfile(bin_path, dtype=np.float32).reshape([-1, 5])[:, :3]
        print(points.shape)
        
        boxes_3d = []
        names = []
        for box, name in zip(gt_boxes, gt_names):
            if name != 'ignore':
                boxes_3d.append(box)
                names.append(name)
        boxes_3d = np.array(boxes_3d)

        if len(points) and len(boxes_3d):
            box_idxs = points_in_boxes_gpu(
                torch.from_numpy(points).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(boxes_3d).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()
        else:
            # box_idxs = np.zeros(len(points)) - 1
            continue
        
        gt_list.append(
        {
            "bbox": boxes_3d,
            "class": names,
        })
        pts_list.append(points)
        frame_ids_list.append(frame_id)
        pts_label_list.append(box_idxs) 

    info = {
        "idx_names": frame_ids_list,
        "pts": pts_list,
        "pts_label": pts_label_list,
        "gt": gt_list,
    }
    return info

if __name__ == '__main__':

    np.set_printoptions(precision=3, linewidth=500,
                        threshold=np.inf, suppress=True)
    
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_file', type=str, default="once_data", help='the data path')
    parser.add_argument('--seq_id', type=str, default="000076", help='the sequence id')
    # parser.add_argument('--data_file', type=str, default="nuscenes_data", help='the data path of nuscenes')
    # parser.add_argument('--seq_id', type=str, default="n015-2018-07-18-11-07-57+0800", help='the sequence id of nuscenes')
    parser.add_argument('--func', type=str, default='once', help='choose the data')

    args = parser.parse_args()

    if args.func == 'once':
        info = load_once(args.data_file, args.seq_id)
    elif args.func == 'nuscenes':
        info = load_nuscenes(args.data_file, args.seq_id)

    sequence_visualize3d(**info)
