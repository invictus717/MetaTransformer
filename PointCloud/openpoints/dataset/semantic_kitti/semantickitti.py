import pickle
import os
from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
from ..build import DATASETS
from ..data_util import crop_pc 
from ...transforms.point_transform_cpu import PointsToTensor


def load_pc_kitti(pc_path):
    scan = np.fromfile(pc_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, 0:3]  # get xyz, what about the fourth? 
    return points

def load_label_kitti(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    # inst_label = label >> 16  # instance id in upper half
    sem_label = remap_lut[sem_label] -1 
    return sem_label.astype(np.int32)


def get_semantickitti_file_list(dataset_path, test_seq_num):
    seq_list = np.sort(os.listdir(dataset_path))

    train_file_list = []
    test_file_list = []
    val_file_list = []
    for seq_id in seq_list:
        seq_path = join(dataset_path, seq_id)
        label_path = join(seq_path, 'labels')
        pc_path = join(seq_path, 'velodyne')
        path_list =  [[join(pc_path, f), join(label_path, f.replace('bin', 'label'))] for f in np.sort(os.listdir(pc_path))]
        
        if seq_id == '08':
            val_file_list.append(path_list)
            if seq_id == test_seq_num:
                test_file_list.append(path_list)
        elif int(seq_id) >= 11 and seq_id == test_seq_num:
            print("\n\n\n Loading test seq_id ", test_seq_num)
            test_file_list.append(path_list)
        elif seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
            train_file_list.append(path_list)

    train_file_list = np.concatenate(train_file_list, axis=0)
    val_file_list = np.concatenate(val_file_list, axis=0)

    if test_seq_num != 'None':
        test_file_list = np.concatenate(test_file_list, axis=0)
    else:
        test_file_list = None
    return train_file_list, val_file_list, test_file_list


remap_lut_write = np.array([
    0, 10, 11, 15, 18, 20, 30, 31, 32, 40, 44, 48, 49, 50, 51, 70, 71,
    72, 80, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
], dtype=np.int32)
remap_lut_read = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 5, 0, 3, 5, 0, 4, 0, 5, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0,
    10, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 15, 16, 17, 0, 0, 0, 0, 0, 0, 0, 18, 19, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 7, 6, 8, 5, 5, 4, 5, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0
], dtype=np.int32)


@DATASETS.register_module()
class SemanticKITTI(Dataset):
    label_to_names = {
        -1: 'unlabeled', 
        0: 'car',
        1: 'bicycle',
        2: 'motorcycle',
        3: 'truck',
        4: 'other-vehicle',
        5: 'person',
        6: 'bicyclist',
        7: 'motorcyclist',
        8: 'road',
        9: 'parking',
        10: 'sidewalk',
        11: 'other-ground',
        12: 'building',
        13: 'fence',
        14: 'vegetation',
        15: 'trunk',
        16: 'terrain',
        17: 'pole',
        18: 'traffic-sign'
    }
    classes = [item for item in label_to_names.values()]
    del classes[0]
    num_classes = 19
    ignored_labels = [0]
    num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                              240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                              9833174, 129609852, 4506626, 1168181])
    gravity_dim = 2
    
    # number of points after 0.06m  
    # npoints.mean() 79763.77941537705
    # npoints.std() 8013.625502719855
    def __init__(
            self,
            split,
            test_id=None,
            data_root=None,
            voxel_max=45056,  # 4096 * 11, same as RandLANet, but why?
            voxel_size=0.06,
            loop=1, presample=False,  variable=False, 
            transform=None):
        self.data_root = data_root
        self.voxel_max = voxel_max
        self.loop = loop 
        self.voxel_size = voxel_size
        self.presample = presample  # voxel subsample all pointclouds before training
        self.variable = variable
        self.transform = transform 
        self.class_weights = self.get_class_weights()
        self.pipe_transform = PointsToTensor() 

        label_values = np.sort([k for k, v in self.label_to_names.items()])
        label_to_idx = {l: i for i, l in enumerate(label_values)}
        self.ignored_label_inds = [label_to_idx[ign_label]
                                   for ign_label in self.ignored_labels]

        raw_root = join(data_root, 'sequences')
        processed_root = join(data_root, 'processed')
        self.seq_list = np.sort(os.listdir(raw_root))
        if split == 'test' and test_id is not None:
            test_id += 11  # Seq 11 is where the test starts
            self.test_seq_num = str(test_id)
        else:
            test_id = None
        self.split = split
        train_list, val_list, test_list = get_semantickitti_file_list(raw_root, str(test_id))
        if split == 'train':
            self.data_list = train_list
        elif split in ['val', 'validation']:
            self.data_list = val_list
        elif split == 'test':
            self.data_list = test_list

        filename = os.path.join(processed_root, f'semantickitti_{split}_{voxel_size:.3f}.pkl')
        if presample:
            if not os.path.exists(filename):
                self.data = []
                np.random.seed(0) 
                for (pc_path, label_path) in tqdm(self.data_list, desc=f'Loading SemanticKITTI {split} split'):
                    points = load_pc_kitti(pc_path)
                    labels = load_label_kitti(label_path, remap_lut_read)  
                    points, _, labels = crop_pc(points, None, labels, self.split, 
                                            self.voxel_size, self.voxel_max, 
                                            downsample=self.presample, 
                                            variable=self.variable
                                            )
                    cdata = np.hstack([points, np.expand_dims(labels, -1).astype(np.float32)])
                    self.data.append(cdata)
                os.makedirs(processed_root, exist_ok=True)
                with open(filename, 'wb') as f:
                    pickle.dump(self.data, f)
                    print(f"{filename} saved successfully")
            else:
                with open(filename, 'rb') as f:
                    self.data = pickle.load(f)
                    print(f"{filename} load successfully")
        
    def __len__(self):
        return len(self.data_list) * self.loop



    def get_class_weights(self):
        weight = self.num_per_class / float(sum(self.num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)
    
    def __getitem__(self, item):
        cloud_ind = item % len(self.data_list)
        
        if self.presample:
            points, labels = np.split(self.data[cloud_ind], [3], axis=1)
        else:
            pc_path, label_path = self.data_list[cloud_ind]
            points = load_pc_kitti(pc_path)
            labels = load_label_kitti(label_path, remap_lut_read)
        """Vis points
        data = {'pos': points,  'y': labels}
        if self.transform is not None:
            data = self.transform(data)
        from openpoints.dataset.vis3d import vis_multi_points
        vis_multi_points([points, data['pos'].cpu().numpy()], labels=[labels, labels])    
        """
        data = {'pos': points.astype(np.float32),  'y': labels.squeeze().astype(np.long)}
        if self.transform is not None:
            data = self.transform(data)

        if not self.presample: 
            data['pos'], _, data['y'] = crop_pc(data['pos'], None, data['y'], self.split, 
                                        self.voxel_size, self.voxel_max, 
                                        downsample=not self.presample, 
                                        variable=self.variable
                                        )
        data = self.pipe_transform(data)
        if 'heights' not in data.keys():
            data['heights'] =  data['pos'][:, self.gravity_dim:self.gravity_dim+1] - data['pos'][:, self.gravity_dim:self.gravity_dim+1].min()
        return data