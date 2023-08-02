import copy
import pickle
from pathlib import Path
import os
import io

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..semi_dataset import SemiDatasetTemplate

def split_nuscenes_semi_data(dataset_cfg, info_paths, data_splits, root_path, labeled_ratio, logger):
    oss_path = dataset_cfg.OSS_PATH if 'OSS_PATH' in dataset_cfg else None
    if oss_path:
        from petrel_client.client import Client
        client = Client('~/.petreloss.conf')

    nuscenes_pretrain_infos = []
    nuscenes_test_infos = []
    nuscenes_labeled_infos = []
    nuscenes_unlabeled_infos = []
    
    def check_annos(info):
        return 'annos' in info
    
    if dataset_cfg.get('RANDOM_SAMPLE_ID_PATH', None):
        root_path = Path(root_path)
        logger.info('Loading NuScenes dataset')
        nuscenes_infos = {"train":[], "test":[]}

        for info_path in dataset_cfg.INFO_PATH["train"]:
            if oss_path is None:
                info_path = root_path / info_path
                if not info_path.exists():
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    nuscenes_infos["train"].extend(infos)
            else:
                info_path = os.path.join(oss_path, info_path)
                #pkl_bytes = self.client.get(info_path)
                pkl_bytes = client.get(info_path, update_cache=True)
                infos = pickle.load(io.BytesIO(pkl_bytes))
                nuscenes_infos["train"].extend(infos)

        for info_path in dataset_cfg.INFO_PATH["test"]:
            if oss_path is None:
                info_path = root_path / info_path
                if not info_path.exists():
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    nuscenes_infos["test"].extend(infos)
            else:
                info_path = os.path.join(oss_path, info_path)
                #pkl_bytes = self.client.get(info_path)
                pkl_bytes = client.get(info_path, update_cache=True)
                infos = pickle.load(io.BytesIO(pkl_bytes))
                nuscenes_infos["test"].extend(infos)

        sampled_id = np.load(dataset_cfg.RANDOM_SAMPLE_ID_PATH)
        nuscenes_pretrain_infos = [nuscenes_infos["train"][i] for i in sampled_id]
        nuscenes_labeled_infos = [nuscenes_infos["train"][i] for i in sampled_id]
        if dataset_cfg.get('RANDOM_SAMPLE_ID_PATH_UNLABEL', None):
            sampled_id_unlabel = np.load(dataset_cfg.RANDOM_SAMPLE_ID_PATH_UNLABEL)
            nuscenes_unlabeled_infos = [nuscenes_infos["train"][i] for i in sampled_id_unlabel if i not in sampled_id]
        else:
            nuscenes_unlabeled_infos = [nuscenes_infos["train"][i] for i in range(len(nuscenes_infos["train"])) if i not in sampled_id]
        nuscenes_test_infos = nuscenes_infos["test"]

    else:
        root_path = Path(root_path)

        train_split = data_splits['train']
        for info_path in info_paths[train_split]:
            if oss_path is None:
                info_path = root_path / info_path
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    infos = list(filter(check_annos, infos))
                    nuscenes_pretrain_infos.extend(copy.deepcopy(infos))
                    nuscenes_labeled_infos.extend(copy.deepcopy(infos))
            else:
                info_path = os.path.join(oss_path, info_path)
                pkl_bytes = client.get(info_path, update_cache=True)
                infos = pickle.load(io.BytesIO(pkl_bytes))
                # infos = list(filter(check_annos, infos))
                nuscenes_pretrain_infos.extend(copy.deepcopy(infos))
                nuscenes_labeled_infos.extend(copy.deepcopy(infos))


        test_split = data_splits['test']
        for info_path in info_paths[test_split]:
            if oss_path is None:
                info_path = root_path / info_path
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    infos = list(filter(check_annos, infos))
                    nuscenes_test_infos.extend(copy.deepcopy(infos))
            else:
                info_path = os.path.join(oss_path, info_path)
                pkl_bytes = client.get(info_path, update_cache=True)
                infos = pickle.load(io.BytesIO(pkl_bytes))
                # infos = list(filter(check_annos, infos))
                nuscenes_test_infos.extend(copy.deepcopy(infos))

        raw_split = data_splits['raw']
        for info_path in info_paths[raw_split]:
            if oss_path is None:
                info_path = root_path / info_path
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    nuscenes_unlabeled_infos.extend(copy.deepcopy(infos))
            else:
                info_path = os.path.join(oss_path, info_path)
                pkl_bytes = client.get(info_path, update_cache=True)
                infos = pickle.load(io.BytesIO(pkl_bytes))
                nuscenes_unlabeled_infos.extend(copy.deepcopy(infos))

    logger.info('Total samples for nuscenes pre-training dataset: %d' % (len(nuscenes_pretrain_infos)))
    logger.info('Total samples for nuscenes testing dataset: %d' % (len(nuscenes_test_infos)))
    logger.info('Total samples for nuscenes labeled dataset: %d' % (len(nuscenes_labeled_infos)))
    logger.info('Total samples for nuscenes unlabeled dataset: %d' % (len(nuscenes_unlabeled_infos)))

    return nuscenes_pretrain_infos, nuscenes_test_infos, nuscenes_labeled_infos, nuscenes_unlabeled_infos


class NuScenesSemiDataset(SemiDatasetTemplate):
    """Petrel Ceph storage backend.
        3DTrans supports the reading and writing data from Ceph
        Usage:
        self.oss_path = 's3://path/of/nuScenes'
        '~/.petreloss.conf': A config file of Ceph, saving the KEY/ACCESS_KEY of S3 Ceph
    """
    def __init__(self, dataset_cfg, class_names,infos=None, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        if self.oss_path is not None:
            from petrel_client.client import Client
            self.client = Client('~/.petreloss.conf')
            # self.oss_data_list = self.list_oss_dir(self.oss_path, with_info=False) 
            # zhangbo: for OSS format, list the nuScenes dataset will cause a Bug, 
            # due to OSS cannot load too many objects
        self.nuscenes_infos = infos

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        if self.oss_path is None:
            lidar_path = self.root_path / sweep_info['lidar_path']
            points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        else:
            lidar_path = os.path.join(self.oss_path, sweep_info['lidar_path'])
            #sdk_local_bytes = self.client.get(lidar_path)
            sdk_local_bytes = self.client.get(lidar_path, update_cache=True)
            points_sweep = np.frombuffer(sdk_local_bytes, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.nuscenes_infos[index]

        if self.oss_path is None:
            lidar_path = self.root_path / info['lidar_path']
            points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        else:
            lidar_path = os.path.join(self.oss_path, info['lidar_path'])
            #sdk_local_bytes = self.client.get(lidar_path)
            sdk_local_bytes = self.client.get(lidar_path, update_cache=True)
            points_pre = np.frombuffer(sdk_local_bytes, dtype=np.float32, count=-1).reshape([-1, 5]).copy()
            points = points_pre[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.nuscenes_infos) * self.total_epochs

        return len(self.nuscenes_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.nuscenes_infos)

        info = copy.deepcopy(self.nuscenes_infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'db_flag': "nusc",
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']},
            'index_id': index
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })
            
            if self.dataset_cfg.get('SHIFT_COOR', None):
                input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR
            
            if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
                input_dict['gt_boxes'] = None
            
            # for debug only
            # gt_boxes_mask = np.array([n in self.class_names for n in input_dict['gt_names']], dtype=np.bool_)
            # debug_dict = {'gt_boxes': copy.deepcopy(input_dict['gt_boxes'][gt_boxes_mask])}
        
        if self.dataset_cfg.get('FOV_POINTS_ONLY', None):
            input_dict['points'] = self.extract_fov_data(
                input_dict['points'], self.dataset_cfg.FOV_DEGREE, self.dataset_cfg.FOV_ANGLE
            )
            if input_dict['gt_boxes'] is not None:
                fov_gt_flag = self.extract_fov_gt(
                    input_dict['gt_boxes'], self.dataset_cfg.FOV_DEGREE, self.dataset_cfg.FOV_ANGLE
                )
                input_dict.update({
                    'gt_names': input_dict['gt_names'][fov_gt_flag],
                    'gt_boxes': input_dict['gt_boxes'][fov_gt_flag],
                })
        
        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict

    #@staticmethod
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            if self.dataset_cfg.get('SHIFT_COOR', None):
                #print ("*******WARNING FOR SHIFT_COOR:", self.dataset_cfg.SHIFT_COOR)
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        map_name_to_kitti = {
            'car': 'Car',
            'pedestrian': 'Pedestrian',
            'truck': 'Truck',
            'bicycle': 'Cyclist',
        }

        def transform_to_kitti_format(annos, info_with_fakelidar=False, is_gt=False):
            for anno in annos:
                if 'name' not in anno:
                    anno['name'] = anno['gt_names']
                    anno.pop('gt_names')

                for k in range(anno['name'].shape[0]):
                    if anno['name'][k] in map_name_to_kitti:
                        anno['name'][k] = map_name_to_kitti[anno['name'][k]]
                    else:
                        anno['name'][k] = 'Person_sitting'

                if 'boxes_lidar' in anno:
                    gt_boxes_lidar = anno['boxes_lidar'].copy()
                else:
                    gt_boxes_lidar = anno['gt_boxes'].copy()

                # filter by fov
                if is_gt and self.dataset_cfg.get('GT_FILTER', None):
                    if self.dataset_cfg.GT_FILTER.get('FOV_FILTER', None):
                        fov_gt_flag = self.extract_fov_gt(
                            gt_boxes_lidar, self.dataset_cfg['FOV_DEGREE'], self.dataset_cfg['FOV_ANGLE']
                        )
                        gt_boxes_lidar = gt_boxes_lidar[fov_gt_flag]
                        anno['name'] = anno['name'][fov_gt_flag]

                anno['bbox'] = np.zeros((len(anno['name']), 4))
                anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
                anno['truncated'] = np.zeros(len(anno['name']))
                anno['occluded'] = np.zeros(len(anno['name']))

                if len(gt_boxes_lidar) > 0:
                    if info_with_fakelidar:
                        gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

                    gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
                    anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
                    anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
                    anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
                    anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
                    dxdydz = gt_boxes_lidar[:, 3:6]
                    anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
                    anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
                    anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
                else:
                    anno['location'] = anno['dimensions'] = np.zeros((0, 3))
                    anno['rotation_y'] = anno['alpha'] = np.zeros(0)

        transform_to_kitti_format(eval_det_annos)
        transform_to_kitti_format(eval_gt_annos, info_with_fakelidar=False, is_gt=True)

        kitti_class_names = []
        for x in class_names:
            if x in map_name_to_kitti:
                kitti_class_names.append(map_name_to_kitti[x])
            else:
                kitti_class_names.append('Person_sitting')
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )
        return ap_result_str, ap_dict

    def nuscene_eval(self, det_annos, class_names, **kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import nuscenes_utils
        nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            eval_set=eval_set_map[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if kwargs['eval_metric'] == 'kitti':
            eval_det_annos = copy.deepcopy(det_annos)
            eval_gt_annos = copy.deepcopy(self.nuscenes_infos)
            return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
        elif kwargs['eval_metric'] == 'nuscenes':
            return self.nuscene_eval(det_annos, class_names, **kwargs)
        else:
            raise NotImplementedError


class NuScenesPretrainDataset(NuScenesSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.nuscenes_infos)

        info = copy.deepcopy(self.nuscenes_infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'db_flag': "nusc",
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']},
            'index_id': index
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })
            
            if self.dataset_cfg.get('SHIFT_COOR', None):
                input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict

class NuScenesLabeledDataset(NuScenesSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )
        self.labeled_data_for = dataset_cfg.LABELED_DATA_FOR

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.nuscenes_infos)

        info = copy.deepcopy(self.nuscenes_infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'db_flag': "nusc",
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']},
            'index_id': index
        }

        assert 'gt_boxes' in info
        if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
            mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
        else:
            mask = None

        input_dict.update({
            'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
            'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
        })
        
        if self.dataset_cfg.get('SHIFT_COOR', None):
            input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

        teacher_dict, student_dict = self.prepare_data_ssl(input_dict, output_dicts=self.labeled_data_for)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
            if teacher_dict is not None:
                gt_boxes = teacher_dict['gt_boxes']
                gt_boxes[np.isnan(gt_boxes)] = 0
                teacher_dict['gt_boxes'] = gt_boxes

            gt_boxes = student_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            student_dict['gt_boxes'] = gt_boxes
        
        if teacher_dict is not None and not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in teacher_dict and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
            teacher_dict['gt_boxes'] = teacher_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in student_dict and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
            student_dict['gt_boxes'] = student_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return tuple([teacher_dict, student_dict])


class NuScenesUnlabeledDataset(NuScenesSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )
        self.unlabeled_data_for = dataset_cfg.UNLABELED_DATA_FOR

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.nuscenes_infos)

        info = copy.deepcopy(self.nuscenes_infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'db_flag': "nusc",
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']},
            'index_id': index
        }

        teacher_dict, student_dict = self.prepare_data_ssl(input_dict, output_dicts=self.unlabeled_data_for)

        return tuple([teacher_dict, student_dict])

class NuScenesTestDataset(NuScenesSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=False, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is False
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.nuscenes_infos)

        info = copy.deepcopy(self.nuscenes_infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'db_flag': "nusc",
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']},
            'index_id': index
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })
            
            if self.dataset_cfg.get('SHIFT_COOR', None):
                input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

        data_dict = self.prepare_data(data_dict=input_dict)
                    
        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
            
        return data_dict