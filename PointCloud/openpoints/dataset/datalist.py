
import torch
from torch.utils.data import dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np


class DataList(dataset.Dataset):
    def __init__(self, 
                 dataset_name,
                 split, 
                 data_list,  
                 **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.data_list = data_list 
        self.split = split
        
    def load_data(self, data_path):
        if 's3dis' in self.dataset_name:
            data = np.load(data_path)  # xyzrgbl, N*7
            coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
            feat = np.clip(feat / 255., 0, 1).astype(np.float32)
        elif 'scannet' in self.dataset_name:
            data = torch.load(data_path)  # xyzrgbl, N*7
            if self.split != 'test':
                coord, feat, label = data[0], data[1], data[2]
            else:
                coord, feat, label = data[0], data[1], None
            feat = np.clip((feat + 1) / 2., 0, 1).astype(np.float32)
        elif 'semantickitti' in self.dataset_name: 
            points = self.load_pc_kitti(pc_path)
            labels = self.load_label_kitti(label_path, self.remap_lut_read)
            
        coord -= coord.min(0)

        idx_points = []
        voxel_size = cfg.dataset.common.get('voxel_size', None)
        if voxel_size is not None:
            idx_sort, count = voxelize(coord, voxel_size, mode=1)
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
        else:
            idx_points.append(np.arange(label.shape[0]))
        return coord, feat, label, idx_points


    
    def __len__(self):
        return len(self.record_tokens)
    
    def __getitem__(self, index):
        token  = self.record_tokens[index]
        try:
            return self._records[token]
        except AttributeError:
            record = self.read_record(token)
            self._records = {token:record}
            return record
        except KeyError:
            record = self.read_record(token)
            self._records[token] = record
            return record
