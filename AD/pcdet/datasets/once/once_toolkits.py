import json
import os.path as osp
from collections import defaultdict
import cv2
import numpy as np

class Octopus(object):
    """
    dataset structure:
    - data_root
        - train_split.txt
        - val_split.txt
        - test_split.txt
        -
    """
    camera_names = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
    camera_tags = ['top', 'top2', 'left_back', 'left_front', 'right_front', 'right_back', 'back']

    def __init__(self, dataset_root, oss_path=None, client=None):
        self.dataset_root = dataset_root
        self.oss_path = oss_path
        self.data_root = osp.join(self.dataset_root, 'data')
        if self.oss_path is not None:
            self.oss_root = osp.join(self.oss_path, 'data')
            self.client = client
        self._collect_basic_infos()

    @property
    def train_split_list(self):
        if not osp.isfile(osp.join(self.dataset_root, 'ImageSets', 'train_set.txt')):
            train_split_list = None
        else:
            train_split_list = set(map(lambda x: x.strip(),
                                       open(osp.join(self.data_root, 'train_set.txt')).readlines()))
        return train_split_list

    @property
    def val_split_list(self):
        if not osp.isfile(osp.join(self.dataset_root, 'ImageSets', 'val_set.txt')):
            val_split_list = None
        else:
            val_split_list = set(map(lambda x: x.strip(),
                                     open(osp.join(self.data_root, 'val_set.txt')).readlines()))
        return val_split_list

    @property
    def test_split_list(self):
        if not osp.isfile(osp.join(self.dataset_root, 'ImageSets', 'test_set.txt')):
            test_split_list = None
        else:
            test_split_list = set(map(lambda x: x.strip(),
                                       open(osp.join(self.data_root, 'test_set.txt')).readlines()))
        return test_split_list

    @property
    def raw_split_list(self):
        if not osp.isfile(osp.join(self.dataset_root, 'ImageSets', 'raw_set.txt')):
            raw_split_list = None
        else:
            raw_split_list = set(map(lambda x: x.strip(),
                                       open(osp.join(self.data_root, 'raw_set.txt')).readlines()))
        return raw_split_list

    def _find_split_name(self, seq_id):
        if seq_id in self.raw_split_list:
            return 'raw'
        if seq_id in self.train_split_list:
            return 'train'
        if seq_id in self.test_split_list:
            return 'test'
        if seq_id in self.val_split_list:
            return 'val'
        print("sequence id {} corresponding to no split".format(seq_id))
        raise NotImplementedError

    def _collect_basic_infos(self):
        self.train_info = defaultdict(dict)
        if self.train_split_list is not None:
            for train_seq in self.train_split_list:
                if self.oss_path is None:
                    anno_file_path = osp.join(self.data_root, train_seq, '{}.json'.format(train_seq))
                    if not osp.isfile(anno_file_path):
                        print("no annotation file for sequence {}".format(train_seq))
                        raise FileNotFoundError
                    anno_file = json.load(open(anno_file_path, 'r'))
                else:
                    anno_file_path = osp.join(self.oss_root, train_seq, '{}.json'.format(train_seq))
                    text_bytes = self.client.get(anno_file_path, update_cache=True)
                    text_bytes = text_bytes.decode('utf-8')
                    anno_file = json.load(io.StringIO(text_bytes))

                for frame_anno in anno_file['frames']:
                    self.train_info[train_seq][frame_anno['frame_id']] = {
                        'pose': frame_anno['pose'],
                        'calib': anno_file['calib'],
                    }

    def get_frame_anno(self, seq_id, frame_id):
        split_name = self._find_split_name(seq_id)
        frame_info = getattr(self, '{}_info'.format(split_name))[seq_id][frame_id]
        if 'anno' in frame_info:
            return frame_info['anno']
        return None

    def load_point_cloud(self, seq_id, frame_id):
        if self.oss_path is None:
            bin_path = osp.join(self.data_root, seq_id, 'lidar_roof', '{}.bin'.format(frame_id))
            points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        else:
            bin_path = osp.join(self.oss_root, seq_id, 'lidar_roof', '{}.bin'.format(frame_id))
            sdk_local_bytes = self.client.get(bin_path, update_cache=True)
            points = np.frombuffer(sdk_local_bytes, dtype=np.float32).reshape(-1, 4).copy()
        return points

    def load_image(self, seq_id, frame_id, cam_name):
        if self.oss_path is None:
            cam_path = osp.join(self.data_root, seq_id, cam_name, '{}.jpg'.format(frame_id))
            img_buf = cv2.cvtColor(cv2.imread(cam_path), cv2.COLOR_BGR2RGB)
        else:
            cam_path = osp.join(self.oss_root, seq_id, cam_name, '{}.jpg'.format(frame_id))
            sdk_local_bytes = self.client.get(cam_path, update_cache=True)
            img_buf = cv2.cvtColor(np.frombuffer(sdk_local_bytes, dtype='uint8'), cv2.COLOR_BGR2RGB)
        return img_buf

    def project_lidar_to_image(self, seq_id, frame_id):
        points = self.load_point_cloud(seq_id, frame_id)

        split_name = self._find_split_name(seq_id)
        frame_info = getattr(self, '{}_info'.format(split_name))[seq_id][frame_id]
        points_img_dict = dict()
        for cam_name in self.__class__.camera_names:
            calib_info = frame_info['calib'][cam_name]
            cam_2_velo = calib_info['cam_to_velo']
            cam_intri = calib_info['cam_intrinsic']
            point_xyz = points[:, :3]
            points_homo = np.hstack(
                [point_xyz, np.ones(point_xyz.shape[0], dtype=np.float32).reshape((-1, 1))])
            points_lidar = np.dot(points_homo, np.linalg.inv(cam_2_velo).T)
            mask = points_lidar[:, 2] > 0
            points_lidar = points_lidar[mask]
            points_img = np.dot(points_lidar, cam_intri.T)
            points_img_dict[cam_name] = points_img
        return points_img_dict

    def undistort_image(self, seq_id, frame_id):
        pass
