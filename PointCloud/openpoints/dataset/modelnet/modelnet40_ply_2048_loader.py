"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""
import os
import glob
import h5py
import numpy as np
import pickle
import logging
import ssl
import urllib
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive, check_integrity
from ..build import DATASETS


def download_and_extract_archive(url, path, md5=None):
    # Works when the SSL certificate is expired for the link
    path = Path(path)
    extract_path = path
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / Path(url).name
        if not file_path.exists() or not check_integrity(file_path, md5):
            print(f'{file_path} not found or corrupted')
            print(f'downloading from {url}')
            context = ssl.SSLContext()
            with urllib.request.urlopen(url, context=context) as response:
                with tqdm(total=response.length) as pbar:
                    with open(file_path, 'wb') as file:
                        chunk_size = 1024
                        chunks = iter(lambda: response.read(chunk_size), '')
                        for chunk in chunks:
                            if not chunk:
                                break
                            pbar.update(chunk_size)
                            file.write(chunk)
            extract_archive(str(file_path), str(extract_path))
    return extract_path


def load_data(data_dir, partition, url):
    download_and_extract_archive(url, data_dir)
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0).squeeze(-1)
    return all_data, all_label


@DATASETS.register_module()
class ModelNet40Ply2048(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.
    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    dir_name = 'modelnet40_ply_hdf5_2048'
    md5 = 'c9ab8e6dfb16f67afdab25e155c79e59'
    url = f'https://shapenet.cs.stanford.edu/media/{dir_name}.zip'
    classes = ['airplane',
               'bathtub',
               'bed',
               'bench',
               'bookshelf',
               'bottle',
               'bowl',
               'car',
               'chair',
               'cone',
               'cup',
               'curtain',
               'desk',
               'door',
               'dresser',
               'flower_pot',
               'glass_box',
               'guitar',
               'keyboard',
               'lamp',
               'laptop',
               'mantel',
               'monitor',
               'night_stand',
               'person',
               'piano',
               'plant',
               'radio',
               'range_hood',
               'sink',
               'sofa',
               'stairs',
               'stool',
               'table',
               'tent',
               'toilet',
               'tv_stand',
               'vase',
               'wardrobe',
               'xbox']

    def __init__(self,
                 num_points=1024,
                 data_dir="./data/ModelNet40Ply2048",
                 split='train',
                 transform=None
                 ):
        data_dir = os.path.join(
            os.getcwd(), data_dir) if data_dir.startswith('.') else data_dir
        self.partition = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.data, self.label = load_data(data_dir, self.partition, self.url)
        self.num_points = num_points
        logging.info(f'==> sucessfully loaded {self.partition} data')
        self.transform = transform

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        if self.partition == 'train':
            np.random.shuffle(pointcloud)
        data = {'pos': pointcloud,
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = data['pos']
        return data

    def __len__(self):
        return self.data.shape[0]

    @property
    def num_classes(self):
        return np.max(self.label) + 1

    """ for visulalization
    from openpoints.dataset import vis_multi_points
    import copy
    old_points = copy.deepcopy(data['pos'])
    if self.transform is not None:
        data = self.transform(data)
    new_points = copy.deepcopy(data['pos'])
    vis_multi_points([old_points, new_points.numpy()])
    End of visulization """