import copy
import pickle
from pathlib import Path
from . import kitti_utils
import io
import os
import numpy as np
from tqdm import tqdm
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..semi_dataset import SemiDatasetTemplate

def split_kitti_semi_data(dataset_cfg, info_paths, data_splits, root_path, labeled_ratio, logger):
    oss_path = dataset_cfg.OSS_PATH if 'OSS_PATH' in dataset_cfg else None
    if oss_path:
        from petrel_client.client import Client
        client = Client('~/.petreloss.conf')

    kitti_pretrain_infos = []
    kitti_test_infos = []
    kitti_labeled_infos = []
    kitti_unlabeled_infos = []
    
    def check_annos(info):
        return 'annos' in info
    
    if dataset_cfg.get('RANDOM_SAMPLE_ID_PATH', None):
        root_path = Path(root_path)
        logger.info('Loading kitti dataset')
        kitti_infos = {"train":[], "test":[]}

        for info_path in dataset_cfg.INFO_PATH["train"]:
            if oss_path is None:
                info_path = root_path / info_path
                if not info_path.exists():
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    kitti_infos["train"].extend(infos)
            else:
                info_path = os.path.join(oss_path, info_path)
                #pkl_bytes = self.client.get(info_path)
                pkl_bytes = client.get(info_path, update_cache=True)
                infos = pickle.load(io.BytesIO(pkl_bytes))
                kitti_infos["train"].extend(infos)

        for info_path in dataset_cfg.INFO_PATH["test"]:
            if oss_path is None:
                info_path = root_path / info_path
                if not info_path.exists():
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    kitti_infos["test"].extend(infos)
            else:
                info_path = os.path.join(oss_path, info_path)
                #pkl_bytes = self.client.get(info_path)
                pkl_bytes = client.get(info_path, update_cache=True)
                infos = pickle.load(io.BytesIO(pkl_bytes))
                kitti_infos["test"].extend(infos)

        sampled_id = np.load(dataset_cfg.RANDOM_SAMPLE_ID_PATH)
        kitti_pretrain_infos = [kitti_infos["train"][i] for i in sampled_id]
        kitti_labeled_infos = [kitti_infos["train"][i] for i in sampled_id]
        if dataset_cfg.get('RANDOM_SAMPLE_ID_PATH_UNLABEL', None):
            sampled_id_unlabel = np.load(dataset_cfg.RANDOM_SAMPLE_ID_PATH_UNLABEL)
            kitti_unlabeled_infos = [kitti_infos["train"][i] for i in sampled_id_unlabel if i not in sampled_id]
        else:
            kitti_unlabeled_infos = [kitti_infos["train"][i] for i in range(len(kitti_infos["train"])) if i not in sampled_id]
        kitti_test_infos = kitti_infos["test"]

    else:
        root_path = Path(root_path)

        train_split = data_splits['train']
        for info_path in info_paths[train_split]:
            if oss_path is None:
                info_path = root_path / info_path
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    infos = list(filter(check_annos, infos))
                    kitti_pretrain_infos.extend(copy.deepcopy(infos))
                    kitti_labeled_infos.extend(copy.deepcopy(infos))
            else:
                info_path = os.path.join(oss_path, info_path)
                pkl_bytes = client.get(info_path, update_cache=True)
                infos = pickle.load(io.BytesIO(pkl_bytes))
                # infos = list(filter(check_annos, infos))
                kitti_pretrain_infos.extend(copy.deepcopy(infos))
                kitti_labeled_infos.extend(copy.deepcopy(infos))

        test_split = data_splits['test']
        for info_path in info_paths[test_split]:
            if oss_path is None:
                info_path = root_path / info_path
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    infos = list(filter(check_annos, infos))
                    kitti_test_infos.extend(copy.deepcopy(infos))
            else:
                info_path = os.path.join(oss_path, info_path)
                pkl_bytes = client.get(info_path, update_cache=True)
                infos = pickle.load(io.BytesIO(pkl_bytes))
                # infos = list(filter(check_annos, infos))
                kitti_test_infos.extend(copy.deepcopy(infos))

        raw_split = data_splits['raw']
        for info_path in info_paths[raw_split]:
            if oss_path is None:
                info_path = root_path / info_path
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    kitti_unlabeled_infos.extend(copy.deepcopy(infos))
            else:
                info_path = os.path.join(oss_path, info_path)
                pkl_bytes = client.get(info_path, update_cache=True)
                infos = pickle.load(io.BytesIO(pkl_bytes))
                kitti_unlabeled_infos.extend(copy.deepcopy(infos))

    logger.info('Total samples for kitti pre-training dataset: %d' % (len(kitti_pretrain_infos)))
    logger.info('Total samples for kitti testing dataset: %d' % (len(kitti_test_infos)))
    logger.info('Total samples for kitti labeled dataset: %d' % (len(kitti_labeled_infos)))
    logger.info('Total samples for kitti unlabeled dataset: %d' % (len(kitti_unlabeled_infos)))

    return kitti_pretrain_infos, kitti_test_infos, kitti_labeled_infos, kitti_unlabeled_infos


class KittiSemiDataset(SemiDatasetTemplate):
    """Petrel Ceph storage backend.
        3DTrans supports the reading and writing data from Ceph
        Usage:
        self.oss_path = 's3://path/of/KITTI'
        '~/.petreloss.conf': A config file of Ceph, saving the KEY/ACCESS_KEY of S3 Ceph
    """
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        if self.oss_path is not None:
            from petrel_client.client import Client
            self.client = Client('~/.petreloss.conf')
            if self.split != 'test':
                self.root_split_path = os.path.join(self.oss_path, 'training')
            else:
                self.root_split_path = os.path.join(self.oss_path, 'testing')
        else:
            self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = infos

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        if self.oss_path is None:
            lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
            assert lidar_file.exists()
            points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        else:
            lidar_file = os.path.join(self.root_split_path, 'velodyne', ('%s.bin' % idx))
            sdk_local_bytes = self.client.get(lidar_file, update_cache=True)
            points = np.frombuffer(sdk_local_bytes, dtype=np.float32).reshape(-1, 4).copy()

        return points

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        from skimage import io
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth
    
    def get_calib(self, idx):
        if self.oss_path is None:
            calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
            assert calib_file.exists()
            calibrated_res = calibration_kitti.Calibration(calib_file, False)
        else:
            calib_file = os.path.join(self.root_split_path, 'calib', ('%s.txt' % idx))
            text_bytes = self.client.get(calib_file, update_cache=True)
            text_bytes = text_bytes.decode('utf-8')
            calibrated_res = calibration_kitti.Calibration(io.StringIO(text_bytes), True)
        return calibrated_res
        
    def get_road_plane(self, idx):
        if self.oss_path is None:
            plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
            if not plane_file.exists():
                return None

            with open(plane_file, 'r') as f:
                lines = f.readlines()
        else:
            plane_file = os.path.join(self.root_split_path, 'planes', ('%s.txt' % idx))
            text_bytes = self.client.get(plane_file, update_cache=True)
            text_bytes = text_bytes.decode('utf-8')
            lines = io.StringIO(text_bytes).readlines()
            
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane
        
    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib, margin=0):
        """
        Args:
            pts_rect:
            img_shape:
            calib:
            margin:
        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0 - margin, pts_img[:, 0] < img_shape[1] + margin)
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0 - margin, pts_img[:, 1] < img_shape[0] + margin)
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

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
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()

            if self.dataset_cfg.get('SHIFT_COOR', None):
                #print ("*******WARNING FOR SHIFT_COOR:", self.dataset_cfg.SHIFT_COOR)
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR

            # BOX FILTER
            if self.dataset_cfg.get('TEST', None) and self.dataset_cfg.TEST.BOX_FILTER['FOV_FILTER']:
                box_preds_lidar_center = pred_boxes[:, 0:3]
                pts_rect = calib.lidar_to_rect(box_preds_lidar_center)
                fov_flag = self.get_fov_flag(pts_rect, image_shape, calib, margin=5)
                pred_boxes = pred_boxes[fov_flag]
                pred_labels = pred_labels[fov_flag]
                pred_scores = pred_scores[fov_flag]
            
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'db_flag': "kitti",
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            
            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]
            if self.dataset_cfg.get('REMOVE_ORIGIN_GTS', None) and self.training:
                input_dict['points'] = box_utils.remove_points_in_boxes3d(input_dict['points'], input_dict['gt_boxes'])
                mask = np.zeros(gt_boxes_lidar.shape[0], dtype=np.bool_)
                input_dict['gt_boxes'] = input_dict['gt_boxes'][mask]
                input_dict['gt_names'] = input_dict['gt_names'][mask]

            if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
                input_dict['gt_boxes'] = None

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane
                
            # for debug only
            # gt_boxes_mask = np.array([n in self.class_names for n in input_dict['gt_names']], dtype=np.bool_)
            # debug_dict = {'gt_boxes': copy.deepcopy(gt_boxes_lidar[gt_boxes_mask])}
            
        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            
            img_shape = info['image']['image_shape']
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            if self.dataset_cfg.get('SHIFT_COOR', None):
                points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        # load saved pseudo label for unlabel data
        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)
            
        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict


class KittiPretrainDataset(KittiSemiDataset):
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
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'db_flag': "kitti",
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            
            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]
            if self.dataset_cfg.get('REMOVE_ORIGIN_GTS', None) and self.training:
                input_dict['points'] = box_utils.remove_points_in_boxes3d(input_dict['points'], input_dict['gt_boxes'])
                mask = np.zeros(gt_boxes_lidar.shape[0], dtype=np.bool_)
                input_dict['gt_boxes'] = input_dict['gt_boxes'][mask]
                input_dict['gt_names'] = input_dict['gt_names'][mask]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane
                
        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            
            img_shape = info['image']['image_shape']
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            if self.dataset_cfg.get('SHIFT_COOR', None):
                points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)
            
        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict

class KittiLabeledDataset(KittiSemiDataset):
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
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])        

        input_dict = {
            'db_flag': "kitti",
            'frame_id': sample_idx,
            'calib': calib,
        }

        assert 'annos' in info
        annos = info['annos']
        annos = common_utils.drop_info_with_name(annos, name='DontCare')
        loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
        gt_names = annos['name']
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
        
        if self.dataset_cfg.get('SHIFT_COOR', None):
            gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR

        input_dict.update({
            'gt_names': gt_names,
            'gt_boxes': gt_boxes_lidar
        })
        if "gt_boxes2d" in get_item_list:
            input_dict['gt_boxes2d'] = annos["bbox"]
        if self.dataset_cfg.get('REMOVE_ORIGIN_GTS', None) and self.training:
            input_dict['points'] = box_utils.remove_points_in_boxes3d(input_dict['points'], input_dict['gt_boxes'])
            mask = np.zeros(gt_boxes_lidar.shape[0], dtype=np.bool_)
            input_dict['gt_boxes'] = input_dict['gt_boxes'][mask]
            input_dict['gt_names'] = input_dict['gt_names'][mask]

        road_plane = self.get_road_plane(sample_idx)
        if road_plane is not None:
            input_dict['road_plane'] = road_plane
                
        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            
            img_shape = info['image']['image_shape']
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            if self.dataset_cfg.get('SHIFT_COOR', None):
                points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        teacher_dict, student_dict = self.prepare_data_ssl(input_dict, output_dicts=self.labeled_data_for)
        
        if teacher_dict is not None :
            teacher_dict['image_shape'] = img_shape
        if student_dict is not None:
            student_dict['image_shape'] = img_shape

        return tuple([teacher_dict, student_dict])


class KittiUnlabeledDataset(KittiSemiDataset):
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
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'db_flag': "kitti",
            'frame_id': sample_idx,
            'calib': calib,
        }
                
        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            
            img_shape = info['image']['image_shape']
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            if self.dataset_cfg.get('SHIFT_COOR', None):
                points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        teacher_dict, student_dict = self.prepare_data_ssl(input_dict, output_dicts=self.unlabeled_data_for)
        
        if teacher_dict is not None :
            teacher_dict['image_shape'] = img_shape
        if student_dict is not None:
            student_dict['image_shape'] = img_shape

        return tuple([teacher_dict, student_dict])


class KittiTestDataset(KittiSemiDataset):
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
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'db_flag': "kitti",
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            
            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]
            if self.dataset_cfg.get('REMOVE_ORIGIN_GTS', None) and self.training:
                input_dict['points'] = box_utils.remove_points_in_boxes3d(input_dict['points'], input_dict['gt_boxes'])
                mask = np.zeros(gt_boxes_lidar.shape[0], dtype=np.bool_)
                input_dict['gt_boxes'] = input_dict['gt_boxes'][mask]
                input_dict['gt_names'] = input_dict['gt_names'][mask]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane
                
        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            
            img_shape = info['image']['image_shape']
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            if self.dataset_cfg.get('SHIFT_COOR', None):
                points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)
            
        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict