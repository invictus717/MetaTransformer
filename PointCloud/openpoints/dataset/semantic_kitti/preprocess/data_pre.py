import pickle, yaml, os, sys
import numpy as np
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
# from helper_tool import DataProcessing as DP


class DataProcessing:
    @staticmethod
    def load_pc_semantic3d(filename):
        pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float16)
        pc = pc_pd.values
        return pc

    @staticmethod
    def load_label_semantic3d(filename):
        label_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
        cloud_labels = label_pd.values
        return cloud_labels

    @staticmethod
    def load_pc_kitti(pc_path):
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]  # get xyz
        return points

    @staticmethod
    def load_label_kitti(label_path, remap_lut):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)

    @staticmethod
    def get_file_list(dataset_path, test_scan_num):
        seq_list = np.sort(os.listdir(dataset_path))

        train_file_list = []
        test_file_list = []
        val_file_list = []
        for seq_id in seq_list:
            seq_path = join(dataset_path, seq_id)
            pc_path = join(seq_path, 'velodyne')
            if seq_id == '08':
                val_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
                if seq_id == test_scan_num:
                    test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif int(seq_id) >= 11 and seq_id == test_scan_num:
                test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                train_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        train_file_list = np.concatenate(train_file_list, axis=0)
        val_file_list = np.concatenate(val_file_list, axis=0)
        if test_scan_num != 'None':
            test_file_list = np.concatenate(test_file_list, axis=0)
        else:
            test_file_list = None
        return train_file_list, val_file_list, test_file_list

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def get_class_weights(dataset_name):
        # pre-calculate the number of points in each category
        num_per_class = []
        if dataset_name is 'S3DIS':
            num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                      650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
        elif dataset_name is 'Semantic3D':
            num_per_class = np.array([5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353],
                                     dtype=np.int32)
        elif dataset_name is 'SemanticKITTI':
            num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                      240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                      9833174, 129609852, 4506626, 1168181])
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)


data_config = os.path.join(BASE_DIR, 'semantic-kitti.yaml')
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map"]
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

grid_size = 0.06
dataset_path = '/data/WQ/DataSet/semantic-kitti/dataset/sequences'
output_path = '/data/WQ/DataSet/semantic-kitti/dataset/sequences' + '_' + str(grid_size)
seq_list = np.sort(os.listdir(dataset_path))

for seq_id in seq_list:
    print('sequence' + seq_id + ' start')
    seq_path = join(dataset_path, seq_id)
    seq_path_out = join(output_path, seq_id)
    pc_path = join(seq_path, 'velodyne')
    pc_path_out = join(seq_path_out, 'velodyne')
    KDTree_path_out = join(seq_path_out, 'KDTree')
    os.makedirs(seq_path_out) if not exists(seq_path_out) else None
    os.makedirs(pc_path_out) if not exists(pc_path_out) else None
    os.makedirs(KDTree_path_out) if not exists(KDTree_path_out) else None

    if int(seq_id) < 11:
        label_path = join(seq_path, 'y')
        label_path_out = join(seq_path_out, 'y')
        os.makedirs(label_path_out) if not exists(label_path_out) else None
        scan_list = np.sort(os.listdir(pc_path))
        for scan_id in scan_list:
            print(scan_id)
            points = DP.load_pc_kitti(join(pc_path, scan_id))
            labels = DP.load_label_kitti(join(label_path, str(scan_id[:-4]) + '.label'), remap_lut)
            sub_points, sub_labels = DP.grid_sub_sampling(points, labels=labels, grid_size=grid_size)
            search_tree = KDTree(sub_points)
            KDTree_save = join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
            np.save(join(pc_path_out, scan_id)[:-4], sub_points)
            np.save(join(label_path_out, scan_id)[:-4], sub_labels)
            with open(KDTree_save, 'wb') as f:
                pickle.dump(search_tree, f)
            if seq_id == '08':
                proj_path = join(seq_path_out, 'proj')
                os.makedirs(proj_path) if not exists(proj_path) else None
                proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
                proj_inds = proj_inds.astype(np.int32)
                proj_save = join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
                with open(proj_save, 'wb') as f:
                    pickle.dump([proj_inds], f)
    else:
        proj_path = join(seq_path_out, 'proj')
        os.makedirs(proj_path) if not exists(proj_path) else None
        scan_list = np.sort(os.listdir(pc_path))
        for scan_id in scan_list:
            print(scan_id)
            points = DP.load_pc_kitti(join(pc_path, scan_id))
            sub_points = DP.grid_sub_sampling(points, grid_size=0.06)
            search_tree = KDTree(sub_points)
            proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            KDTree_save = join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
            proj_save = join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
            np.save(join(pc_path_out, scan_id)[:-4], sub_points)
            with open(KDTree_save, 'wb') as f:
                pickle.dump(search_tree, f)
            with open(proj_save, 'wb') as f:
                pickle.dump([proj_inds], f)
