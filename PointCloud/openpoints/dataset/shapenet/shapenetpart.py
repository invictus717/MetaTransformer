import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# change this to your data root
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def download_shapenetpart(DATA_DIR):
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR)):
        os.mkdir(os.path.join(DATA_DIR))
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], os.path.join(DATA_DIR)))
        os.system('rm %s' % (zipfile))


def load_data_partseg(partition, DATA_DIR):
    download_shapenetpart(DATA_DIR)
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, '*%s*.h5' % partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(rotation_matrix)  # random rotation (x,z)
    return pointcloud


class ShapeNetPart(Dataset):
    def __init__(self, data_root='data/shapenetpart', num_points=2048, split='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(split, data_root)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = split
        self.class_choice = class_choice

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]

        data = {'pos': pointcloud, 'y': label,
                    'offsets': seg}

        return data

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ShapeNetPart(num_points=2048, split='trainval')

    test = ShapeNetPart(num_points=2048, split='test')
    for dict in train:
        for i in dict:
            print(dict[i].shape)
        # print(dd)
