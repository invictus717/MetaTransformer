"""_summary_
"""


import os
import numpy as np
import pickle
import logging
import pathlib
import glob
import random
import h5py
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from ..build import DATASETS
from openpoints.models.layers import fps
import warnings
warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


@DATASETS.register_module()
class MP40(Dataset):
    classes = ["wall",
               "floor",
               "chair",
               "door"
               "table",
               "picture",
               "cabinet",
               "cushion",
               "window",
               "sofa"
               "bed"
               "curtain",
               "chest_of_drawers",
               "plant",
               "sink"
               "stairs",
               "ceiling",
               "toilet",
               "stool",
               "towel",
               "mirror",
               "tv_monitor",
               "shower",
               "column",
               "bathtub",
               "counter",
               "fireplace",
               "lighting",
               "beam"
               "railing",
               "shelving",
               "blinds",
               "gym_equipment",
               "seating",
               "board_panel",
               "furniture",
               "appliances",
               "clothes",
               "objects",
               "misc"
               ]

    def __init__(self,
                 data_dir,
                 num_points=1024,
                 split='train',
                 transform=None,
                 use_normal=False,
                 **kwargs
                 ):
        self.npoints = num_points
        self.preprocess = True
        self.uniform = True
        self.split = split
        self.use_normal = use_normal
        self.transform = transform

        root_dir = pathlib.Path(data_dir)
        data_dir = root_dir / 'raw'
        data_list = root_dir / f'mattportobject_{split}_list.txt'

        if not data_list.exists():
            all_files = [i.name for i in data_dir.glob('*.npy')]
            random.shuffle(all_files)
            n_files = len(all_files)
            n_train = int(0.8 * n_files)
            n_val = int(0.1 * n_files)
            train_files = all_files[:n_train]
            val_files = all_files[n_train:n_train+n_val]
            test_files = all_files[n_train+n_val:]
            train_list = str(root_dir / (f'mattportobject_train_list.txt'))
            val_list = str(root_dir / (f'mattportobject_val_list.txt'))
            test_list = str(root_dir / (f'mattportobject_test_list.txt'))
            with open(train_list, 'w') as f:
                f.writelines('\n'.join(train_files))
            with open(val_list, 'w') as f:
                f.writelines('\n'.join(val_files))
            with open(test_list, 'w') as f:
                f.writelines('\n'.join(test_files))
        with open(data_list, 'r') as f:
            lines = f.read().splitlines()
        self.datapath = [data_dir / l for l in lines]

        if self.uniform:
            self.save_path = os.path.join(
                root_dir, 'matterport3dobjects_%s_%dpts_fps.h5' % (split, 2048))
        else:
            self.save_path = os.path.join(
                root_dir, 'matterport3dobjects_%s_%dpts.h5' % (split, 2048))

        npoints = 2048
        if not os.path.exists(self.save_path):
            logging.info(
                'Processing data %s (only running in the first time)...' % self.save_path)
            list_of_points = []
            list_of_labels = []

            for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                fn = self.datapath[index]
                data = np.load(fn, allow_pickle=True).item()
                cls = data['label']
                point_set = data['points'].astype(np.float32)
                if self.uniform:
                    point_set = fps(torch.from_numpy(point_set).unsqueeze(
                        0).cuda(), npoints).squeeze(0).cpu().numpy()
                else:
                    point_set = point_set[:npoints, :]

                list_of_points.append(point_set)
                list_of_labels.append(cls)

            all_points = np.stack(list_of_points)
            all_labels = np.stack(list_of_labels)
            all_data = {'data': all_points, 'label': all_labels}
            hf = h5py.File(self.save_path, 'w')
            data = hf.create_group('data')
            for k, v in all_data.items():
                data[k] = v
            hf.close()
        logging.info('Load processed data from %s...' % self.save_path)
        with h5py.File(self.save_path, 'r') as f:
            data = f['data']['data'][:].astype('float32')
            label = f['data']['label'][:].astype('int32')

        # TODO: make this in preprocessing step. not here.
        cls_mapping = pd.read_csv(str(pathlib.Path(__file__).parent / "category_mapping.tsv"),
                                  skiprows=1, header=None, sep='\t', usecols=[0, 16]).values.astype(int)

        # step 1 remove negative label (useless)
        idx = np.argwhere(label > 0).squeeze()
        data = data[idx]
        label = label[idx]

        # step 2 mapping label to mat40
        # cls_mapping is a np array. row 0 = wall = label(wall) --> mpcat40
        label = cls_mapping[label-1][:, 1]

        # step 3, remove the label not in mat40 (0, and 41)
        label_bigger_0 = label > 0
        label_smaller_41 = label < 41
        idx = []
        for i in range(len(label)):
            if label_bigger_0[i] and label_smaller_41[i]:
                idx.append(i)
        self.data = data[idx]
        self.label = label[idx] - 1

    def __len__(self):
        return len(self.data)

    @property
    def num_classes(self):
        return self.label.max() + 1

    def __getitem__(self, index):
        points, label = self.data[index][:self.npoints], self.label[index]
        if self.split == 'train':
            np.random.shuffle(points)
        data = {'pos': points[:, :3],
                'x': points[:, 3:6 + 3 * self.use_normal],
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)
        """_summary_
        from openpoints.dataset.vis3d import vis_points

        for i, idx in enumerate(np.where(self.label==22)[0]):
            if i >5:
                break
            vis_points(self.data[idx, :, :3], self.data[idx, :, 3:6]/255.)
        """
        if 'heights' in data.keys():
            data['x'] = torch.cat(
                (data['pos'], data['heights'], data['x']), dim=1)
        else:
            data['x'] = torch.cat((data['pos'], data['x']), dim=1)
        return data
