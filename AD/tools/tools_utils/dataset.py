from ast import arg
# from http.client import _DataType
import os
import matplotlib.pyplot as plt
import boto3
import io
import pickle
import numpy as np
import argparse
import pickle
import os
from collections import defaultdict
import time, copy
import numpy as np
import torch
import open3d as o3d
import open3d
import matplotlib
from open3d import geometry
import pickle
from itertools import groupby
import open3d_vis_utils as V
import calibration_kitti


class Dataset():
    def __init__(self, args):
        super().__init__()
        self.dataset_name = args.dataset_name
        self.data_root = args.data_root
        if args.bucket_name is not None:
            self.client = boto3.client(service_name='s3', endpoint_url='')

    def get_data(self, args, info):
        if self.dataset_name == "kitti":
            lidar_idx = info['point_cloud']['lidar_idx']
            # get image shape
            img_shape = info['image']['image_shape']
            print(lidar_idx)
            pointcloud = self.get_lidar_kitti(args, lidar_idx)[:, :4]

            calib = self.get_calib(args, lidar_idx)
            pts_rect = calib.lidar_to_rect(pointcloud[:, 0:3])

            # FOV_only 
            if args.fov:
                pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
                val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
                val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
                val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
                pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
                pointcloud = pointcloud[pts_valid_flag]


            annos = info['annos']
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes = self.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            object_idx = []
            for item in info['annos']['name']:
                if item in args.visualize_categories:
                    object_idx.append(True)
                else:
                    object_idx.append(False)
            gt_boxes = gt_boxes[object_idx, :]

        elif self.dataset_name == "nuscenes":
            pointcloud = self.get_lidar_with_sweeps(args, info)[:, :3]

            object_idx = []
            for item in info['gt_names']:
                if item in args.visualize_categories:
                    object_idx.append(True)
                else:
                    object_idx.append(False)
            gt_boxes = info['gt_boxes'][object_idx, :7]

        elif self.dataset_name == "waymo":
            pc_info = info['point_cloud']
            pointcloud = self.get_lidar_waymo(args, pc_info)[:, :3]
            object_idx = []
            for item in info['annos']['name']:
                if item in args.visualize_categories:
                    object_idx.append(True)
                else:
                    object_idx.append(False)

            gt_boxes = info['annos']['gt_boxes_lidar'][object_idx, :7]

        elif self.dataset_name == "once":
            frame_id = info['frame_id']
            sequence_id = info['sequence_id']
            pointcloud = self.get_lidar_once(args, sequence_id, frame_id)
            
            object_idx = []
            for item in info['annos']['name']:
                if item in args.visualize_categories:
                    object_idx.append(True)
                else:
                    object_idx.append(False)

            gt_boxes = info['annos']['boxes_3d'][object_idx, :]

        return pointcloud, gt_boxes

    def get_lidar_once(self, args, seq_id, frame_id):
        if args.bucket_name is not None:
            bin_path = os.path.join("dataset/once/data", seq_id, 'lidar_roof', '{}.bin'.format(frame_id))
            obj = self.client.get_object(Bucket=args.bucket_name, Key=bin_path)
            points = np.frombuffer(io.BytesIO(obj['Body'].read()).read(), dtype=np.float32).reshape(-1, 4).copy()
        else:
            bin_path = os.path.join(self.data_root, seq_id, 'lidar_roof', '{}.bin'.format(frame_id))
            points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points
    
    def get_lidar_kitti(self, args, idx):
        if args.bucket_name is not None:
            lidar_file = os.path.join("dataset", args.dataset_name, "training", 'velodyne', '%s.bin' % idx)
            obj = self.client.get_object(Bucket=args.bucket_name, Key=lidar_file)
            lidar_points = np.frombuffer(io.BytesIO(obj['Body'].read()).read(), dtype=np.float32).reshape(-1, 4).copy()
        else:
            lidar_file = os.path.join(self.data_root, 'training/velodyne', '%s.bin' % idx)
            lidar_points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        return lidar_points

    def get_sweep(self, args, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]
        if args.bucket_name is not None:
            lidar_path = os.path.join("", sweep_info['lidar_path'])
            obj = self.client.get_object(Bucket=args.bucket_name, Key=lidar_path)
            points_sweep = np.frombuffer(io.BytesIO(obj['Body'].read()).read(), count=-1).reshape([-1, 5])[:, :4].copy()
        else:
            lidar_path = os.path.join(self.data_root, sweep_info['lidar_path'])
            points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, args, info):
        if args.bucket_name is not None:
            lidar_path = os.path.join("dataset/nuScenes/original_raw_data/v1.0-trainval", info['lidar_path'])
            obj = self.client.get_object(Bucket=args.bucket_name, Key=lidar_path)
            points_pre = np.frombuffer(io.BytesIO(obj['Body'].read()).read(), dtype=np.float32, count=-1).reshape([-1, 5]).copy()
            points = points_pre[:, :4]
        else:
            lidar_path = os.path.join(self.data_root, info['lidar_path'])
            points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
            

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), 1 - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def get_lidar_waymo(self, args, pc_info):
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        if args.bucket_name is not None:
            lidar_file = os.path.join("dataset/waymo_0.5.0/waymo_processed_data_v0_5_0", sequence_name,  ('%04d.npy' % sample_idx))
            obj = self.client.get_object(Bucket=args.bucket_name, Key=lidar_file)
            lidar_points = np.load(io.BytesIO(obj['Body'].read())).copy()
        else:
            lidar_file = os.path.join(self.data_root, sequence_name, ('%04d.npy' % sample_idx))
            lidar_points = np.load(lidar_file)
        points_all, NLZ_flag = lidar_points[:, 0:5], lidar_points[:, 5]
        points_all = points_all[NLZ_flag == -1]
        points_all[:, 3] = np.tanh(points_all[:, 3])
        return points_all

    def boxes3d_kitti_camera_to_lidar(self, boxes3d_camera, calib):
        """
        Args:
            boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
            calib:

        Returns:
            boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

        """
        boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
        xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
        l, h, w = boxes3d_camera_copy[:, 3:4], boxes3d_camera_copy[:, 4:5], boxes3d_camera_copy[:, 5:6]

        xyz_lidar = calib.rect_to_lidar(xyz_camera)
        xyz_lidar[:, 2] += h[:, 0] / 2
        return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)

    def get_calib(self, args, idx):
        if args.bucket_name is not None:
            calib_file = os.path.join("dataset", args.dataset_name, "training", "calib", ('%s.txt' % idx))
            text_bytes = self.client.get_object(Bucket=args.bucket_name, Key=calib_file)
            text_bytes = text_bytes['Body'].read().decode('utf-8')
            calibrated_res = calibration_kitti.Calibration(io.StringIO(text_bytes), True)
        else:
            calib_file = os.path.join(self.data_root, 'calib', ('%s.txt' % idx))
            calibrated_res = calibration_kitti.Calibration(calib_file, False)
        return calibrated_res