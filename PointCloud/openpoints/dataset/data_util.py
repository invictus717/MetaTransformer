from curses import keyname
import numpy as np
import torch
import os
import os.path as osp
import ssl
import sys
import urllib
import h5py
from typing import Optional


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
    # # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # # Support PCD files without compression ONLY!
    # @classmethod
    # def _read_pcd(cls, file_path):
    #     pc = open3d.io.read_point_cloud(file_path)
    #     ptcloud = np.array(pc.points)
    #     return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]


# download
def download_url(url: str, folder: str, log: bool = True,
                 filename: Optional[str] = None):
    r"""Downloads the content of an URL to a specific folder. 
    Borrowed from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/download.py
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    os.makedirs(folder, exist_ok=True)
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * \
        np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, voxel_idx, count = np.unique(key_sort, return_counts=True, return_inverse=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[
                               0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, voxel_idx, count


def crop_pc(coord, feat, label, split='train',
            voxel_size=0.04, voxel_max=None,
            downsample=True, variable=True, shuffle=True):
    if voxel_size and downsample:
        # Is this shifting a must? I borrow it from Stratified Transformer and Point Transformer. 
        coord -= coord.min(0) 
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx] if feat is not None else None, label[uniq_idx] if label is not None else None
    if voxel_max is not None:
        crop_idx = None
        N = len(label)  # the number of points
        if N >= voxel_max:
            init_idx = np.random.randint(N) if 'train' in split else N // 2
            crop_idx = np.argsort(
                np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        elif not variable:
            # fill more points for non-variable case (batched data)
            cur_num_points = N
            query_inds = np.arange(cur_num_points)
            padding_choice = np.random.choice(
                cur_num_points, voxel_max - cur_num_points)
            crop_idx = np.hstack([query_inds, query_inds[padding_choice]])
        crop_idx = np.arange(coord.shape[0]) if crop_idx is None else crop_idx
        if shuffle:
            shuffle_choice = np.random.permutation(np.arange(len(crop_idx)))
            crop_idx = crop_idx[shuffle_choice]
        coord, feat, label = coord[crop_idx], feat[crop_idx] if feat is not None else None, label[crop_idx] if label is not None else None
    coord -= coord.min(0) 
    return coord.astype(np.float32), feat.astype(np.float32) if feat is not None else None , label.astype(np.long) if label is not None else None


def get_features_by_keys(data, keys='pos,x'):
    key_list = keys.split(',')
    if len(key_list) == 1:
        return data[keys].transpose(1,2).contiguous()
    else:
        return torch.cat([data[key] for key in keys.split(',')], -1).transpose(1,2).contiguous()


def get_class_weights(num_per_class, normalize=False):
    weight = num_per_class / float(sum(num_per_class))
    ce_label_weight = 1 / (weight + 0.02)

    if normalize:
        ce_label_weight = (ce_label_weight *
                           len(ce_label_weight)) / ce_label_weight.sum()
    return torch.from_numpy(ce_label_weight.astype(np.float32))
