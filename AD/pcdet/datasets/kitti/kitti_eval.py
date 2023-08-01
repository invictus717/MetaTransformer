import pickle
import argparse
from .kitti_object_eval_python import eval as kitti_eval
import copy
import numpy as np
from . import kitti_utils


def filter_by_range(infos, gt_key, range_min=0, range_max=80, is_pred=False, dataset='kitti'):
    infos = copy.deepcopy(infos)
    total_objs = 0
    for i, info in enumerate(infos):
        if is_pred:
            info.pop('truncated', None)
            info.pop('occluded', None)

        location = info['location']
        range_distance = np.linalg.norm(location[:, [0, 2]], axis=-1)
        mask = (range_distance >= range_min) & (range_distance <= range_max)
        total_objs += mask.sum()
        for key, val in info.items():
            if isinstance(val, np.ndarray):
                if key == gt_key:
                    info[key] = val[mask[:val.shape[0]]]  # ignore the Don't Care mask
                elif key in ['car_from_global', 'fov_gt_flag', 'gt_boxes_velocity', 'gt_boxes_token', 'cam_intrinsic',
                             'ref_from_car', 'gt_boxes', 'num_lidar_pts', 'num_radar_pts']:
                    continue
                else:
                    try:
                        info[key] = val[mask]
                    except:
                        import ipdb; ipdb.set_trace(context=20)

    return infos, total_objs


def transform_to_kitti_format(pred_infos, gt_annos, dataset, fakelidar):
    if dataset == 'waymo':
        map_name_to_kitti = {
            'Vehicle': 'Car',
            'Pedestrian': 'Pedestrian',
            'Cyclist': 'Cyclist',
            'Sign': 'Sign',
            'Car': 'Car'
        }
    elif dataset in ['lyft', 'nuscenes']:
        map_name_to_kitti = {
            'car': 'Car',
            'pedestrian': 'Pedestrian',
            'truck': 'Truck',
        }
    else:
        raise NotImplementedError

    kwargs = {
        'is_gt': True,
        'GT_FILTER': True,
        'FOV_FILTER': True,
        'FOV_DEGREE': 90,
        'FOV_ANGLE': 0,
        'RANGE_FILTER': [0, -40, -10, 70.4, 40, 10]
    }

    kitti_utils.transform_annotations_to_kitti_format(pred_infos, map_name_to_kitti=map_name_to_kitti)
    kitti_utils.transform_annotations_to_kitti_format(
        gt_annos, map_name_to_kitti=map_name_to_kitti,
        info_with_fakelidar=fakelidar, **kwargs
    )


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Car'], help='')
    parser.add_argument('--dataset', type=str, default='kitti', help='')
    parser.add_argument('--fakelidar', type=bool, default=False, help='')
    args = parser.parse_args()

    pred_infos = pickle.load(open(args.pred_infos, 'rb'))
    gt_infos = pickle.load(open(args.gt_infos, 'rb'))

    if args.dataset in ['kitti']:
        gt_annos = [info['annos'] for info in gt_infos]
    else:
        gt_annos = gt_infos

    gt_keys = {
        'kitti': ['gt_boxes_lidar'],
        'lyft': 'gt_boxes_lidar',
        'nuscenes': 'gt_boxes_lidar'
    }

    # For other datasets
    if args.dataset != 'kitti':
        transform_to_kitti_format(pred_infos, gt_annos, args.dataset, args.fakelidar)

    print('------------------Start to eval------------------------')

    range_list = [[0, 1000], [0, 30], [30, 50], [50, 80]]
    for cur_range in range_list:
        cur_pred_info, num_pred_objs = filter_by_range(
            pred_infos, gt_keys[args.dataset], range_min=cur_range[0], range_max=cur_range[1],
            is_pred=True, dataset=args.dataset
        )
        cur_gt_annos, num_gt_objs = filter_by_range(
            gt_annos, gt_keys[args.dataset], range_min=cur_range[0], range_max=cur_range[1], dataset=args.dataset
        )

        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            cur_gt_annos, cur_pred_info, current_classes=['Car']
        )
        print(f'----------Range={cur_range}, avg_pred_objs={num_pred_objs / len(pred_infos)}, '
              f'avg_gt_objs={num_gt_objs / len(gt_infos)}-------------')
        print(ap_result_str)


if __name__ == '__main__':
    main()
