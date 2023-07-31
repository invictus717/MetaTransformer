import torch
import torch.utils.data as data
import numpy as np
import os
import pickle
from sklearn.neighbors import KDTree
from ..build import DATASETS
from ..data_util import crop_pc, voxelize
from tqdm import tqdm
import logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@DATASETS.register_module()
class S3DISSphere(data.Dataset):
    label_to_names = {0: 'ceiling',
                      1: 'floor',
                      2: 'wall',
                      3: 'beam',
                      4: 'column',
                      5: 'window',
                      6: 'door',
                      7: 'chair',
                      8: 'table',
                      9: 'bookcase',
                      10: 'sofa',
                      11: 'board',
                      12: 'clutter'}
    name_to_label = {v: i for i, v in enumerate(label_to_names.values())}
    classes = list(label_to_names.values())
    color_mean = np.array([0.5136457, 0.49523646, 0.44921124])
    color_std = np.array([0.18308958, 0.18415008, 0.19252081])
    num_classes = 13
    class2color = {'ceiling':     [0, 255, 0],
                   'floor':       [0, 0, 255],
                   'wall':        [0, 255, 255],
                   'beam':        [255, 255, 0],
                   'column':      [255, 0, 255],
                   'window':      [100, 100, 255],
                   'door':        [200, 200, 100],
                   'chair':       [255, 0, 0],
                   'table':       [170, 120, 200],
                   'bookcase':    [10, 200, 100],
                   'sofa':        [200, 100, 100],
                   'board':       [200, 200, 200],
                   'clutter':     [50, 50, 50]}
    cmap = np.array([*class2color.values()]).astype(np.uint8)
    gravity_dim = 2
    """S3DIS dataset for scene segmentation task.
    Args:
        voxel_size: grid length for pre-subsampling point clouds.
        in_radius: radius of each input spheres.
        num_points: max number of points for the input spheres.
        num_steps: number of spheres for one training epoch.
        num_epochs: total epochs.
        data_root: root path for data.
        transform: data transformations.
        split: dataset split name.
    """
    def __init__(self, voxel_size,
                 in_radius, num_points, num_steps, num_epochs,
                 data_root=None,
                 transform=None,
                 split='train',
                 centering=False,
                 **kwargs):

        super().__init__()
        self.epoch = 0
        self.transform = transform
        self.voxel_size = voxel_size
        self.in_radius = in_radius
        self.num_points = num_points
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.centering = centering
        self.data_root = data_root
        self.train_clouds = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
        self.val_clouds = ['Area_5']
        self.split = split

        if split == 'train':
            cloud_names = self.train_clouds
        else:
            cloud_names = self.val_clouds

        processed_dir = os.path.join(data_root, 'processed')
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        # prepare data
        filename = os.path.join(
            processed_dir, f'{split}_{voxel_size:.3f}_data.pkl')
        if not os.path.exists(filename):
            cloud_points_list, cloud_points_color_list, cloud_points_label_list = [], [], []
            sub_cloud_points_list, sub_cloud_points_label_list, sub_cloud_points_color_list = [], [], []
            sub_cloud_tree_list = []
            cloud_rooms_list = []
            for cloud_idx, cloud_name in enumerate(cloud_names):
                # Pass if the cloud has already been computed
                # Get rooms of the current cloud
                cloud_folder = os.path.join(data_root,cloud_name)
                room_folders = [os.path.join(cloud_folder, room) for room in os.listdir(cloud_folder) if
                                os.path.isdir(os.path.join(cloud_folder, room))]
                # Initiate containers
                cloud_points = np.empty((0, 3), dtype=np.float32)
                cloud_colors = np.empty((0, 3), dtype=np.float32)
                cloud_classes = np.empty((0, 1), dtype=np.int32)
                cloud_room_split = [0]
                # Loop over rooms
                for i, room_folder in enumerate(room_folders):
                    print(
                        'Cloud %s - Room %d/%d : %s' % (
                            cloud_name, i + 1, len(room_folders), room_folder.split('\\')[-1]))
                    room_npoints = []
                    for object_name in os.listdir(os.path.join(room_folder, 'Annotations')):
                        if object_name[-4:] == '.txt':
                            # Text file containing point of the object
                            object_file = os.path.join(room_folder, 'Annotations', object_name)
                            # Object class and ID
                            tmp = object_name[:-4].split('_')[0]
                            if tmp in self.name_to_label:
                                object_class = self.name_to_label[tmp]
                            elif tmp in ['stairs']:
                                object_class = self.name_to_label['clutter']
                            else:
                                raise ValueError(
                                    'Unknown object name: ' + str(tmp))
                            # Read object points and colors
                            try:
                                object_data = np.loadtxt(object_file)
                            except:
                                logging.info(f"error in reading file {object_file}")
                            # Stack all data
                            room_npoints.append(len(object_data))
                            cloud_points = np.vstack(
                                (cloud_points, object_data[:, 0:3].astype(np.float32)))
                            cloud_colors = np.vstack(
                                (cloud_colors, object_data[:, 3:6].astype(np.uint8)))
                            object_classes = np.full(
                                (object_data.shape[0], 1), object_class, dtype=np.int32)
                            cloud_classes = np.vstack(
                                (cloud_classes, object_classes))
                    cloud_room_split.append(cloud_room_split[-1]+sum(room_npoints))
                cloud_points_list.append(cloud_points)
                cloud_points_color_list.append(cloud_colors)
                cloud_points_label_list.append(cloud_classes)
                cloud_rooms_list.append(cloud_room_split)
                
                sub_cloud_file = os.path.join(
                    processed_dir, cloud_name + f'_{voxel_size:.3f}_sub.pkl')
                if os.path.exists(sub_cloud_file):
                    with open(sub_cloud_file, 'rb') as f:
                        sub_points, sub_colors, sub_labels, search_tree = pickle.load(
                            f)
                else:
                    if voxel_size > 0:
                        sub_points, sub_colors, sub_labels = crop_pc(
                            cloud_points, cloud_colors, cloud_classes, voxel_size=voxel_size)
                        sub_labels = np.squeeze(sub_labels)
                    else:
                        sub_points = cloud_points
                        sub_colors = cloud_colors
                        sub_labels = cloud_classes

                    # Get chosen neighborhoods
                    search_tree = KDTree(sub_points, leaf_size=50)

                    with open(sub_cloud_file, 'wb') as f:
                        pickle.dump((sub_points, sub_colors,
                                    sub_labels, search_tree), f)

                sub_cloud_points_list.append(sub_points)
                sub_cloud_points_color_list.append(sub_colors)
                sub_cloud_points_label_list.append(sub_labels)
                sub_cloud_tree_list.append(search_tree)

            # original points
            self.clouds_points = cloud_points_list
            self.clouds_points_colors = cloud_points_color_list
            self.clouds_points_labels = cloud_points_label_list
            self.clouds_rooms = cloud_rooms_list
            
            # grid subsampled points
            self.sub_clouds_points = sub_cloud_points_list
            self.sub_clouds_points_colors = sub_cloud_points_color_list
            self.sub_clouds_points_labels = sub_cloud_points_label_list
            self.sub_cloud_trees = sub_cloud_tree_list

            with open(filename, 'wb') as f:
                pickle.dump((self.clouds_points, self.clouds_points_colors, self.clouds_points_labels, self.clouds_rooms, 
                             self.sub_clouds_points, self.sub_clouds_points_colors, self.sub_clouds_points_labels,
                             self.sub_cloud_trees), f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                (self.clouds_points, self.clouds_points_colors, self.clouds_points_labels, self.clouds_rooms, 
                 self.sub_clouds_points, self.sub_clouds_points_colors, self.sub_clouds_points_labels,
                 self.sub_cloud_trees) = pickle.load(f)
                print(f"{filename} loaded successfully")

        # prepare iteration indices
        filename = os.path.join(processed_dir,
                                f'{split}_{voxel_size:.3f}_{self.num_epochs}_{self.num_steps}_iterinds.pkl')
        if not os.path.exists(filename):
            potentials = []
            min_potentials = []  # self.sub_cloud_trees, the tree for the grid_subsampled area
            for cloud_i, tree in enumerate(self.sub_cloud_trees):
                print(f"{split}/{cloud_i} has {tree.data.shape[0]} points")
                cur_potential = np.random.rand(tree.data.shape[0]) * 1e-3
                potentials.append(cur_potential)
                min_potentials.append(float(np.min(cur_potential)))
            self.cloud_inds = []
            self.point_inds = []
            self.noise = []
            for ep in tqdm(range(self.num_epochs)):
                for st in range(self.num_steps):
                    # cloud_ind is the index for area in each split. for test, only 1 cloud.
                    cloud_ind = int(np.argmin(min_potentials))
                    point_ind = np.argmin(potentials[cloud_ind])
                    self.cloud_inds.append(cloud_ind)
                    self.point_inds.append(point_ind)
                    points = np.array(
                        self.sub_cloud_trees[cloud_ind].data, copy=False)  # sub_points
                    center_point = points[point_ind, :].reshape(1, -1)
                    noise = np.random.normal(
                        scale=self.in_radius / 10, size=center_point.shape)
                    self.noise.append(noise)
                    pick_point = center_point + noise.astype(center_point.dtype)
                    # Indices of points in input region
                    query_inds = self.sub_cloud_trees[cloud_ind].query_radius(pick_point,
                                                                              r=self.in_radius,
                                                                              return_distance=True,
                                                                              sort_results=True)[0][0]
                    cur_num_points = query_inds.shape[0]
                    if self.num_points < cur_num_points:
                        query_inds = query_inds[:self.num_points]
                    # Update potentials (Tuckey weights)
                    dists = np.sum(
                        np.square((points[query_inds] - pick_point).astype(np.float32)), axis=1)
                    tukeys = np.square(1 - dists / np.square(self.in_radius))
                    tukeys[dists > np.square(self.in_radius)] = 0
                    potentials[cloud_ind][query_inds] += tukeys
                    min_potentials[cloud_ind] = float(
                        np.min(potentials[cloud_ind]))
            with open(filename, 'wb') as f:
                pickle.dump((self.cloud_inds, self.point_inds, self.noise), f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                self.cloud_inds, self.point_inds, self.noise = pickle.load(f)
                print(f"{filename} loaded successfully")

        # prepare validation projection inds
        filename = os.path.join(
            processed_dir, f'{split}_{voxel_size:.3f}_proj.pkl')
        if not os.path.exists(filename):
            proj_ind_list = []
            for points, search_tree in zip(self.clouds_points, self.sub_cloud_trees):
                # points: the original points.  700K points
                # subcloud_trees: the tree of the subsampeld points. 500K points
                # find the nearest point in the subsampled points for each point in the original points.
                proj_inds = np.squeeze(search_tree.query(
                    points, return_distance=False))
                proj_inds = proj_inds.astype(np.int32)
                proj_ind_list.append(proj_inds)
            self.projections = proj_ind_list
            with open(filename, 'wb') as f:
                pickle.dump(self.projections, f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                self.projections = pickle.load(f)
                print(f"{filename} loaded successfully")

    def __getitem__(self, idx):
        """
        Returns:
            pts: (N, 3), a point cloud.
            mask: (N, ), 0/1 mask to distinguish padding points.
            features: (input_features_dim, N), input points features.
            pts_labels: (N), point label.
            current_cloud_index: (1), cloud index.
            input_inds: (N), the index of input points in point cloud.
        """
        cloud_ind = self.cloud_inds[idx + self.epoch * self.num_steps]
        point_ind = self.point_inds[idx + self.epoch * self.num_steps]
        noise = self.noise[idx + self.epoch * self.num_steps]
        points = np.array(
            self.sub_cloud_trees[cloud_ind].data, copy=False)  # subpoints
        center_point = points[point_ind, :].reshape(1, -1)
        pick_point = center_point + noise.astype(center_point.dtype)
        # Indices of points in input region
        # the point index in the subsampled point tree.
        query_inds = self.sub_cloud_trees[cloud_ind].query_radius(pick_point,
                                                                  r=self.in_radius,
                                                                  return_distance=True,
                                                                  sort_results=True)[0][0]
        # Number collected
        cur_num_points = query_inds.shape[0]
        if self.num_points < cur_num_points:
            shuffle_choice = np.random.permutation(np.arange(self.num_points))
            input_inds = query_inds[:self.num_points][shuffle_choice]
            mask = torch.ones(self.num_points).type(torch.int32)
        else:
            shuffle_choice = np.random.permutation(np.arange(cur_num_points))
            query_inds = query_inds[shuffle_choice]
            padding_choice = np.random.choice(
                cur_num_points, self.num_points - cur_num_points)
            input_inds = np.hstack([query_inds, query_inds[padding_choice]])
            mask = torch.zeros(self.num_points).type(torch.int32)
            mask[:cur_num_points] = 1
        # points: the whole point of an area (voxel downsampled). 
        # input_inds: the point index of the current area.
        original_points = points[input_inds]
        pts = (original_points - pick_point).astype(np.float32)
        colors = self.sub_clouds_points_colors[cloud_ind][input_inds].astype(
            np.float32)
        labels = self.sub_clouds_points_labels[cloud_ind][input_inds].astype(
            np.int64)
        current_cloud_index = np.array(cloud_ind).astype(np.int64)

        data = {'pos': pts,
                'x': colors,
                'y': labels,
                'mask': mask,
                'cloud_index': current_cloud_index,
                'input_inds': input_inds,
                }
        """ vis
        from openpoints.dataset import vis_multi_points, vis_points
        vis_multi_points([points, pts], [self.sub_clouds_points_colors[cloud_ind]/255., colors/255.])
        
        import copy
        old_data = copy.deepcopy(data)
        if self.transform is not None:
            data = self.transform(data)
        vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3].numpy()], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3].numpy()])
        """   
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' not in data.keys():
            data['heights'] =  torch.from_numpy(original_points[:, self.gravity_dim:self.gravity_dim+1].astype(np.float32))
        return data

    def __len__(self):
        return self.num_steps
