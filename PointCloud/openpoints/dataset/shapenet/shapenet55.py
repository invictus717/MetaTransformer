import os
import torch
import numpy as np
import torch.utils.data as data
from ..data_util import IO
from ..build import DATASETS
import logging


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self,
                 data_dir,
                 n_points,
                 split='train',
                 transform=None
                 ):
        self.data_root = data_dir
        self.pc_path = os.path.join(data_dir, 'shapenet_pc')
        self.subset = 'train' if split == 'train' else 'test'
        self.npoints = n_points
        self.transform = transform

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        logging.info(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        # self.whole = config.get('whole')
        # if self.whole:
        #     test_data_list_file = os.path.join(self.data_root, 'test.txt')
        #     with open(test_data_list_file, 'r') as f:
        #         test_lines = f.readlines()
        #     logging.info(f'[DATASET] Open file {test_data_list_file}')
        #     lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        logging.info(f'[DATASET] {len(self.file_list)} instances were loaded')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        # np.random.shuffle(pc.shape[0])
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        # data = self.random_sample(data, self.sample_points)
        data = self.pc_norm(data).astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)
        data = {'pos': data}
        return data
        # return data, sample['taxonomy_id'], sample['model_id']

    def __len__(self):
        return len(self.file_list)
