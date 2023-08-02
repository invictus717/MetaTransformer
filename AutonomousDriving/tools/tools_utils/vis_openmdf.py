import os
import boto3
import io
import numpy as np
import argparse
import pickle
import os
import pickle
import open3d_vis_utils as V
from dataset import Dataset


def read_s3_pkl(bucket_name, pkl_path):

    obj = client.get_object(Bucket=bucket_name, Key=pkl_path)
    infos = pickle.load(io.BytesIO(obj['Body'].read()))
    return infos

def check_annos(info):
    return 'annos' in info

def vis_scene(args):
    DATA = Dataset(args)
    if args.val_pkl_path is not None:
        try:
            infos_val = read_s3_pkl(args.bucket_name, args.val_pkl_path)
        except:
            with open(args.val_pkl_path, 'rb') as f:
                infos_val = pickle.load(f)

    if args.dataset_name == 'once' and args.vis_gt:
        infos_val = list(filter(check_annos, infos_val))
    
    if args.res_path is not None:
        pkl_z = pickle.load(open(args.res_path, 'rb'))
    else:
        pkl_z = None

    if not os.path.exists(args.dataset_name):
        os.mkdir(args.dataset_name)

    for idx, info in enumerate(infos_val):
        print(idx)
        if idx < 730:
            continue
        pointcloud, gt_boxes = DATA.get_data(args, info)

        if args.vis_gt == False:
            gt_boxes = None

        if pkl_z is None or args.vis_result_box == False:
            box3d = None
        elif args.dataset_name == 'once':
            box3d = pkl_z[idx]['boxes_3d']
        else:
            box3d = pkl_z[idx]['boxes_lidar']
        V.draw_scenes(points=pointcloud, gt_boxes=gt_boxes, ref_boxes=box3d,)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--bucket_name', type=str, default=None) 
    parser.add_argument('--dataset_name', type=str, default="kitti") #kitti, waymo, nuscenes. once
    parser.add_argument('--val_pkl_path',  type=str, default=None)
    parser.add_argument('--result_file', type=str, default=None)
    parser.add_argument('--visualize_categories', type=list, default=['Pedestrian', 'Vehicle', 'Cyclist'])
    parser.add_argument('--vis_gt', type=bool, default=True)
    parser.add_argument('--vis_result_box', type=bool, default=False)
    parser.add_argument('--fov', type=bool, default=True)
    parser.add_argument('--data_root', type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ =='__main__':
    args = parse_config()
    if args.bucket_name is not None:
        client = client = boto3.client(service_name='s3', endpoint_url='')
    vis_scene(args)