'''
Borrowed from PointBERT 
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
from ..build import DATASETS
import torch
import logging
warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


@DATASETS.register_module()
class ModelNet(Dataset):
    def __init__(self,
                 data_dir, num_points, num_classes,
                 use_normals=False,
                 split='train',
                 transform=None
                 ):
        self.root = os.path.join(data_dir, 'modelnet40_normal_resampled')
        self.npoints = num_points
        self.use_normals = use_normals
        self.num_category = num_classes
        split = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.split = split

        if self.num_category == 10:
            self.catfile = os.path.join(
                self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(
                self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        logging.info('The size of %s data is %d' % (split, len(self.datapath)))
        self.transform = transform

    def __len__(self):
        return len(self.datapath)

    @property
    def num_classes(self):
        return self.list_of_labels.max() + 1

    def _get_item(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        label = np.array([cls]).astype(np.int64)
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        return point_set[:, 0:3], point_set[:, 3:], label[0]

    def __getitem__(self, index):
        points, feats, label = self._get_item(index)
        if self.split == 'train':
            np.random.shuffle(points)
        data = {'pos': points,
                'y': label
                }
        if self.use_normals:
            data['x'] = feats
        if self.transform is not None:
            data = self.transform(data)

        if self.use_normals:
            data['x'] = torch.cat((data['pos'], data['x']), dim=1)
        if 'heights' in data.keys():
            data['x'] = torch.cat((data['x'], data['heights']), dim=1)
        return data
