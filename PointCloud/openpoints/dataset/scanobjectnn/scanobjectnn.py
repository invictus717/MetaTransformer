import os, sys, h5py, pickle, numpy as np, logging, os.path as osp
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from openpoints.models.layers import fps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


@DATASETS.register_module()
class ScanObjectNNHardest(Dataset):
    """The hardest variant of ScanObjectNN. 
    The data we use is: `training_objectdataset_augmentedrot_scale75.h5`[1], 
    where there are 2048 points in training and testing.
    The number of training samples is: 11416, and the number of testing samples is 2882. 
    Args:
    """
    classes = [
        "bag",
        "bin",
        "box",
        "cabinet",
        "chair",
        "desk",
        "display",
        "door",
        "shelf",
        "table",
        "bed",
        "pillow",
        "sink",
        "sofa",
        "toilet",
    ]
    num_classes = 15
    gravity_dim = 1

    def __init__(self, data_dir, split,
                 num_points=2048,
                 uniform_sample=True,
                 transform=None,
                 **kwargs):
        super().__init__()
        self.partition = split
        self.transform = transform
        self.num_points = num_points
        slit_name = 'training' if split == 'train' else 'test'
        h5_name = os.path.join(
            data_dir, f'{slit_name}_objectdataset_augmentedrot_scale75.h5')

        if not osp.isfile(h5_name):
            raise FileExistsError(
                f'{h5_name} does not exist, please download dataset at first')
        with h5py.File(h5_name, 'r') as f:
            self.points = np.array(f['data']).astype(np.float32)
            self.labels = np.array(f['label']).astype(int)

        if slit_name == 'test' and uniform_sample:
            precomputed_path = os.path.join(
                data_dir, f'{slit_name}_objectdataset_augmentedrot_scale75_1024_fps.pkl')
            if not os.path.exists(precomputed_path):
                points = torch.from_numpy(self.points).to(torch.float32).cuda()
                self.points = fps(points, 1024).cpu().numpy()
                with open(precomputed_path, 'wb') as f:
                    pickle.dump(self.points, f)
                    print(f"{precomputed_path} saved successfully")
            else:
                with open(precomputed_path, 'rb') as f:
                    self.points = pickle.load(f)
                    print(f"{precomputed_path} load successfully")
        logging.info(f'Successfully load ScanObjectNN {split} '
                     f'size: {self.points.shape}, num_classes: {self.labels.max()+1}')

    @property
    def num_classes(self):
        return self.labels.max() + 1

    def __getitem__(self, idx):
        current_points = self.points[idx][:self.num_points]
        label = self.labels[idx]
        if self.partition == 'train':
            np.random.shuffle(current_points)
        data = {'pos': current_points,
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)

        # height appending. @KPConv
        # TODO: remove pos here, and only use heights. 
        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = torch.cat((data['pos'],
                                   torch.from_numpy(current_points[:, self.gravity_dim:self.gravity_dim+1] - current_points[:, self.gravity_dim:self.gravity_dim+1].min())), dim=1)
        return data

    def __len__(self):
        return self.points.shape[0]

    """ for visulalization
    from openpoints.dataset import vis_multi_points
    import copy
    old_points = copy.deepcopy(data['pos'])
    if self.transform is not None:
        data = self.transform(data)
    new_points = copy.deepcopy(data['pos'])
    vis_multi_points([old_points, new_points.numpy()])
    End of visulization """