import logging
import os
import pickle
import random
import shutil
import subprocess
import SharedArray

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.autograd import Variable,Function

from ..utils.spconv_utils import spconv

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    elif isinstance(x, (int, float)):
        # modified
        d = np.array([x],dtype=np.float32)
        return torch.from_numpy(d).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_pad_params(desired_size, cur_size):
    """
    Get padding parameters for np.pad function
    Args:
        desired_size: int, Desired padded output size
        cur_size: int, Current size. Should always be less than or equal to cur_size
    Returns:
        pad_params: tuple(int), Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)

    return pad_params


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    # os.environ['MASTER_PORT'] = str(tcp_port)
    # os.environ['MASTER_ADDR'] = 'localhost'
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        # init_method='tcp://127.0.0.1:%d' % tcp_port,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret


def generate_voxel2pinds(sparse_tensor):
    device = sparse_tensor.indices.device
    batch_size = sparse_tensor.batch_size
    spatial_shape = sparse_tensor.spatial_shape
    indices = sparse_tensor.indices.long()
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor


def sa_create(name, var):
    """
        Args:
            name: identify the shared memory, file:// prefix to indicate file while shm:// to indicate to be a POSIX shared memory object
            var: only use the var.shape and var.dtype to create SA object
            see more: https://pypi.org/project/SharedArray/
    """
    x = SharedArray.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def add_prefix_to_dict(dict, prefix):
    for key in list(dict.keys()):
        dict[prefix + key] = dict.pop(key)
    return dict


class DataReader(object):
    def __init__(self, dataloader, sampler):
        self.dataloader = dataloader
        self.sampler = sampler

    def construct_iter(self):
        self.dataloader_iter = iter(self.dataloader)

    def set_cur_epoch(self, cur_epoch):
        self.cur_epoch = cur_epoch

    def read_data(self):
        try:
            return self.dataloader_iter.next()
        except:
            if self.sampler is not None:
                self.sampler.set_epoch(self.cur_epoch)
            self.construct_iter()
            return self.dataloader_iter.next()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def calculate_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

class GRLayer(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.alpha = weight

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output, None


def grad_reverse(x, weight):
    return GRLayer.apply(x, weight)

def split_two_spare_tensor(split_tag_s1, split_tag_s2, sparse_tensor):
    """
        Function: split the sparse_tensor into two sparse_tensor, accodring to the given batch_size
        Args:
            split_tag_s1: array (batch_len)
            split_tag_s2: array (batch_len)
            sparse_tensor: 
        Returns:

    """
    voxel_features = sparse_tensor.features
    voxel_coords = sparse_tensor.indices
    
    # split the voxel_coords of the dataset-merged voxel_coords
    tar_coor_s1 = []
    tar_coor_s2 = []
    bs_s1 = 0
    bs_s2 = 0
    for i in split_tag_s1:
        voxel_coords_s1 = voxel_coords[i==voxel_coords[:,0]]
        voxel_coords_s1[:,0] = bs_s1
        bs_s1 += 1
        tar_coor_s1.append(voxel_coords_s1)
    tar_s1 = torch.cat(tar_coor_s1, axis=0)
    for j in split_tag_s2:
        voxel_coords_s2 = voxel_coords[j==voxel_coords[:,0]]
        voxel_coords_s2[:,0] = bs_s2
        bs_s2 += 1
        tar_coor_s2.append(voxel_coords_s2)
    tar_s2 = torch.cat(tar_coor_s2, axis=0)
    
    # split the voxel_tensor of the dataset-merged voxel_coords
    tar_list_s1 = []
    tar_list_s2 = []
    for i in split_tag_s1:
        voxel_s1 = voxel_features[i==voxel_coords[:,0]]
        tar_list_s1.append(voxel_s1.reshape(-1, voxel_s1.shape[-1]))
    for j in split_tag_s2:
        voxel_s2 = voxel_features[j==voxel_coords[:,0]]
        tar_list_s2.append(voxel_s2.reshape(-1, voxel_s2.shape[-1]))
    voxel_features_s1 = torch.cat(tar_list_s1, axis=0)
    voxel_features_s2 = torch.cat(tar_list_s2, axis=0)
    
    # convert the dense_tensor of voxel into the sparse representations
    input_sp_tensor_s1 = spconv.SparseConvTensor(
        features=voxel_features_s1,
        indices=tar_s1.int(),
        spatial_shape=sparse_tensor.spatial_shape,
        batch_size=len(split_tag_s1)
    )
    input_sp_tensor_s2 = spconv.SparseConvTensor(
        features=voxel_features_s2,
        indices=tar_s2.int(),
        spatial_shape=sparse_tensor.spatial_shape,
        batch_size=len(split_tag_s2)
    )

    return input_sp_tensor_s1, input_sp_tensor_s2

# For split the batch_dict for two head

def split_two_batch_dict(split_tag_s1, split_tag_s2, batch_dict):

    tar_dicts_s1 = {}
    tar_dicts_s2 = {}
    
    for key, val in batch_dict.items():
        if key in ['db_flag', 'frame_id', 'use_lead_xyz']:
            tar_list_s1 = []
            tar_list_s2 = []
            for i in split_tag_s1:
                tar_list_s1.append(val[i])
            for j in split_tag_s2:
                tar_list_s2.append(val[j])
            tar_dicts_s1[key] = tar_list_s1
            tar_dicts_s2[key] = tar_list_s2
        elif key in ['points', 'voxel_coords']:
            tar_list_s1 = []
            tar_list_s2 = []
            bs_s1 = 0
            bs_s2 = 0
            for i in split_tag_s1:
                idx_bs_s1 = [np.where(i==val[:,0])]
                point_s1 = val[tuple(idx_bs_s1)]
                point_s1[0,:,0] = bs_s1
                bs_s1 = bs_s1 + 1
                tar_list_s1.append(point_s1.reshape(-1, point_s1.shape[2]))
            for j in split_tag_s2:
                idx_bs_s2 = [np.where(j==val[:,0])]
                point_s2 = val[tuple(idx_bs_s2)]
                point_s2[0,:,0] = bs_s2
                bs_s2 += 1
                tar_list_s2.append(point_s2.reshape(-1, point_s2.shape[2]))
            tar_dicts_s1[key] = np.concatenate(tar_list_s1, axis=0)
            tar_dicts_s2[key] = np.concatenate(tar_list_s2, axis=0)
        elif key in ['gt_boxes']:
            tar_dicts_s1[key] = val[split_tag_s1, :, :]
            tar_dicts_s2[key] = val[split_tag_s2, :, :]
        elif key in ['voxels']:
            tar_list_s1 = []
            tar_list_s2 = []
            for i in split_tag_s1:
                idx_bs_s1 = [np.where(i==batch_dict['voxel_coords'][:,0])]
                voxel_s1 = val[tuple(idx_bs_s1)]
                tar_list_s1.append(voxel_s1.reshape(-1, voxel_s1.shape[-2], voxel_s1.shape[-1]))
            for j in split_tag_s2:
                idx_bs_s2 = [np.where(j==batch_dict['voxel_coords'][:,0])]
                voxel_s2 = val[tuple(idx_bs_s2)]
                tar_list_s2.append(voxel_s2.reshape(-1, voxel_s2.shape[-2], voxel_s2.shape[-1]))
            tar_dicts_s1[key] = np.concatenate(tar_list_s1, axis=0)
            tar_dicts_s2[key] = np.concatenate(tar_list_s2, axis=0)
        elif key in ['voxel_num_points']:
            tar_list_s1 = []
            tar_list_s2 = []
            for i in split_tag_s1:
                idx_bs_s1 = [np.where(i==batch_dict['voxel_coords'][:,0])]
                voxel_s1 = val[tuple(idx_bs_s1)]
                tar_list_s1.append(voxel_s1.reshape(-1))
            for j in split_tag_s2:
                idx_bs_s2 = [np.where(j==batch_dict['voxel_coords'][:,0])]
                voxel_s2 = val[tuple(idx_bs_s2)]
                tar_list_s2.append(voxel_s2.reshape(-1))
            tar_dicts_s1[key] = np.concatenate(tar_list_s1, axis=0)
            tar_dicts_s2[key] = np.concatenate(tar_list_s2, axis=0)
        elif key in [ 'metadata' ]:
            # Due to that the kitti do not have the 'metadata' key, and give the 'metadata' key to nusc branch
            if "kitti" in batch_dict['db_flag']:
                tar_dicts_s2[key] = val
            else:
                tar_list_s1 = []
                tar_list_s2 = []
                for i in split_tag_s1:
                    tar_list_s1.append(val[i])
                for j in split_tag_s2:
                    tar_list_s2.append(val[j])
                tar_dicts_s1[key] = tar_list_s1
                tar_dicts_s2[key] = tar_list_s2
        elif key in ['image_shape']:
            # Due to that the waymo and nusc do not have the 'image_shape' key, 
            # and assume that kitti feeds into the Branch ONE
            if "kitti" in batch_dict['db_flag']:
                tar_list_s1 = []
                for i in split_tag_s1:
                    tar_list_s1.append(val[i])
                tar_dicts_s1[key] = tar_list_s1
        elif key in ['batch_size']:
            tar_dicts_s1[key] = len(split_tag_s1)
            tar_dicts_s2[key] = len(split_tag_s2)
        else:
            continue
        
    return tar_dicts_s1, tar_dicts_s2


def split_two_batch_dict_gpu(split_tag_s1, split_tag_s2, batch_dict):

    tar_dicts_s1 = {}
    tar_dicts_s2 = {}
    
    for key, val in batch_dict.items():
        if key in ['db_flag', 'frame_id', 'use_lead_xyz']:
            tar_list_s1 = []
            tar_list_s2 = []
            for i in split_tag_s1:
                tar_list_s1.append(val[i])
            for j in split_tag_s2:
                tar_list_s2.append(val[j])
            tar_dicts_s1[key] = tar_list_s1
            tar_dicts_s2[key] = tar_list_s2
        elif key in ['points', 'voxel_coords', 'point_coords']:
            tar_list_s1 = []
            tar_list_s2 = []
            bs_s1 = 0
            bs_s2 = 0
            for i in split_tag_s1:
                point_s1 = val[i==val[:,0]]
                point_s1[:,0] = bs_s1
                bs_s1 += 1
                tar_list_s1.append(point_s1)
            for j in split_tag_s2:
                point_s2 = val[j==val[:,0]]
                point_s2[:,0] = bs_s2
                bs_s2 += 1
                tar_list_s2.append(point_s2)
            tar_dicts_s1[key] = torch.cat(tar_list_s1, axis=0)
            tar_dicts_s2[key] = torch.cat(tar_list_s2, axis=0)
        elif key in ['gt_boxes']:
            tar_dicts_s1[key] = val[split_tag_s1, :, :]
            tar_dicts_s2[key] = val[split_tag_s2, :, :]
        elif key in ['voxel_features']:
            tar_list_s1 = []
            tar_list_s2 = []
            for i in split_tag_s1:
                voxel_s1 = val[i==batch_dict['voxel_coords'][:,0]]
                tar_list_s1.append(voxel_s1.reshape(-1, voxel_s1.shape[-1]))
            for j in split_tag_s2:
                voxel_s2 = val[j==batch_dict['voxel_coords'][:,0]]
                tar_list_s2.append(voxel_s2.reshape(-1, voxel_s2.shape[-1]))
            tar_dicts_s1[key] = torch.cat(tar_list_s1, axis=0)
            tar_dicts_s2[key] = torch.cat(tar_list_s2, axis=0)
        elif key in ['point_features', 'point_features_before_fusion']:
            tar_list_s1 = []
            tar_list_s2 = []
            for i in split_tag_s1:
                point_s1 = val[i==batch_dict['point_coords'][:,0]]
                tar_list_s1.append(point_s1.reshape(-1, point_s1.shape[-1]))
            for j in split_tag_s2:
                point_s2 = val[j==batch_dict['point_coords'][:,0]]
                tar_list_s2.append(point_s2.reshape(-1, point_s2.shape[-1]))
            tar_dicts_s1[key] = torch.cat(tar_list_s1, axis=0)
            tar_dicts_s2[key] = torch.cat(tar_list_s2, axis=0)
        elif key in ['spatial_features', 'spatial_features_2d']:
            tar_dicts_s1[key] = val[split_tag_s1, :, :, :]
            tar_dicts_s2[key] = val[split_tag_s2, :, :, :]
        elif key in [ 'metadata' ]:
            # Due to that the kitti and once do not have the 'metadata' key, 
            # and only give the 'metadata' key to nusc branch
            if "kitti" or "once" in batch_dict['db_flag']:
                tar_dicts_s1[key] = val
            else:
                tar_list_s1 = []
                tar_list_s2 = []
                for i in split_tag_s1:
                    tar_list_s1.append(val[i])
                for j in split_tag_s2:
                    tar_list_s2.append(val[j])
                tar_dicts_s1[key] = tar_list_s1
                tar_dicts_s2[key] = tar_list_s2
        elif key in ['image_shape']:
            # Due to that the waymo and nusc do not have the 'image_shape' key, 
            if "kitti" in batch_dict['db_flag']:
                # assume that kitti feeds into the Branch ONE
                if batch_dict['db_flag'][0] == 'kitti':
                    tar_list_s1 = []
                    for i in split_tag_s1:
                        tar_list_s1.append(val)
                    tar_dicts_s1[key] = torch.cat(tar_list_s1, axis=0)
                # assume that kitti feeds into the Branch TWO
                else:
                    tar_list_s2 = []
                    for i in split_tag_s2:
                        tar_list_s2.append(val)
                    tar_dicts_s2[key] = torch.cat(tar_list_s2, axis=0)

        elif key in ['multi_scale_3d_strides']:
            # Since different datasets for the 'multi_scale_3d_strides' key have the same value,
            # we directly copy this value into tar_dicts_s1, tar_dicts_s2
            tar_dicts_s1[key] = val
            tar_dicts_s2[key] = val
        
        elif key in ['multi_scale_3d_features']:
            # We need to transfer the sparse tensor into the dense tensor
            sp_3d_s1 = {}
            sp_3d_s2 = {}
            for src_name in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:
                input_sp_tensor_s1, input_sp_tensor_s2 = split_two_spare_tensor(split_tag_s1, split_tag_s2, val[src_name])
                sp_3d_s1[src_name] = input_sp_tensor_s1
                sp_3d_s2[src_name] = input_sp_tensor_s2
            
            tar_dicts_s1[key] = sp_3d_s1
            tar_dicts_s2[key] = sp_3d_s2
        
        elif key in ['batch_size']:
            tar_dicts_s1[key] = len(split_tag_s1)
            tar_dicts_s2[key] = len(split_tag_s2)
        elif key in ['spatial_features_stride']:
            tar_dicts_s1[key] = val
            tar_dicts_s2[key] = val
        else:
            continue
        
    return tar_dicts_s1, tar_dicts_s2


def split_batch_dict(source_one_name, batch_dict):
    split_tag_s1 = []
    split_tag_s2 = []
    for k in range(batch_dict['batch_size']):
        if source_one_name in batch_dict['db_flag'][k]:
            split_tag_s1.append(k)
        else:
            split_tag_s2.append(k)
        
    return split_tag_s1, split_tag_s2

def merge_two_batch_dict(batch_dict_1, batch_dict_2):
    """
    To support a custom dataset, implement this function to merge two batch_dict (and labels)
    from different datasets

    Args:
        batch_dict_1:
        batch_dict_2:

    Returns:
        batch_merge_dict:
    """
   
    batch_merge_dict = {}
    batch_merge_dict['batch_size'] = batch_dict_1['batch_size'] + batch_dict_2['batch_size']

    for key, val in batch_dict_1.items():
        if key in ['batch_size']:
            continue
        elif key in ['db_flag', 'frame_id', 'use_lead_xyz']:
            tar_list_merge = []
            tar_list_merge = [val, batch_dict_2[key]]
            batch_merge_dict[key] = np.concatenate(tar_list_merge, axis=0)
        elif key in ['voxels', 'voxel_num_points']:
            tar_list_merge = []
            tar_list_merge = [val, batch_dict_2[key]]
            batch_merge_dict[key] = np.concatenate(tar_list_merge, axis=0)
        elif key in ['points', 'voxel_coords']:
            tar_list_merge = []
            batch_bias = batch_dict_1['batch_size']
            val_2 = batch_dict_2[key]
            val_2[:,0] = val_2[:,0] + batch_bias
            tar_list_merge = [val, val_2]
            batch_merge_dict[key] = np.concatenate(tar_list_merge, axis=0)
        elif key in ['gt_boxes']:
            max_gt_1 = max([len(x) for x in val])
            max_gt_2 = max([len(x) for x in batch_dict_2[key]])
            if max_gt_1 > max_gt_2:
                val_2 = batch_dict_2['gt_boxes']
                batch_gt_boxes3d = np.zeros((batch_dict_2['batch_size'], max_gt_1, val_2[0].shape[-1]), dtype=np.float32)
                #filling the gt_boxes of the batch_dict_2
                for k in range(batch_dict_2['batch_size']):
                    batch_gt_boxes3d[k, :val_2[k].__len__(), :] = val_2[k]
                
                tar_list_merge = []
                tar_list_merge = [val, batch_gt_boxes3d]
                batch_merge_dict[key] = np.concatenate(tar_list_merge, axis=0)
            else:
                val_2 = batch_dict_2['gt_boxes']
                batch_gt_boxes3d = np.zeros((batch_dict_1['batch_size'], max_gt_2, val[0].shape[-1]), dtype=np.float32)
                #filling the gt_boxes of the batch_dict_1
                for k in range(batch_dict_1['batch_size']):
                    batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]

                tar_list_merge = []
                tar_list_merge = [batch_gt_boxes3d, val_2]
                batch_merge_dict[key] = np.concatenate(tar_list_merge, axis=0)

        elif key in ['metadata', 'image_shape', 'road_plane', 'calib']:
            # Due to that the kitti do not have the 'metadata' key, and give the 'metadata' key to nusc branch
            if key in batch_dict_2.keys():
                # both dataset have the 'metadata key'
                tar_list_merge = []
                tar_list_merge = [val, batch_dict_2[key]]
                batch_merge_dict[key] = np.concatenate(tar_list_merge, axis=0)
            else:
                batch_merge_dict[key] = val
    
    if 'metadata' in batch_dict_2.keys() and 'metadata' not in batch_dict_1.keys() :
        batch_merge_dict['metadata'] = batch_dict_2['metadata']

    if 'image_shape' in batch_dict_2.keys() and 'image_shape' not in batch_dict_1.keys() :
        batch_merge_dict['image_shape'] = batch_dict_2['image_shape']

    if 'road_plane' in batch_dict_2.keys() and 'road_plane' not in batch_dict_1.keys() :
        batch_merge_dict['road_plane'] = batch_dict_2['road_plane']
    
    if 'calib' in batch_dict_2.keys() and 'calib' not in batch_dict_1.keys() :
        batch_merge_dict['calib'] = batch_dict_2['calib']

    return batch_merge_dict


def merge_two_batch_dict_gpu(split_tag_s1, split_tag_s2, batch_dict):
    """
    To support a custom dataset, implement this function to merge two batch_dict (and labels)
    from different datasets

    Args:
        split_tag_s1:
        split_tag_s2:
        batch_dict:

    Returns:
        batch_merge_dict:
    """
    raise NotImplementedError