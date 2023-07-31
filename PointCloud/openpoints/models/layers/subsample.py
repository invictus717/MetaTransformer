# subsample layer for 3d processing.
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.autograd import Function
import math
from openpoints.cpp.pointnet2_batch import pointnet2_cuda


class BaseSampler(ABC):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def __init__(self, ratio=None, num_to_sample=None, subsampling_param=None):
        if num_to_sample is not None:
            if (ratio is not None) or (subsampling_param is not None):
                raise ValueError(
                    "Can only specify ratio or num_to_sample or subsampling_param, not several !")
            self._num_to_sample = num_to_sample

        elif ratio is not None:
            self._ratio = ratio

        elif subsampling_param is not None:
            self._subsampling_param = subsampling_param

        else:
            raise Exception(
                'At least ["ratio, num_to_sample, subsampling_param"] should be defined')

    def __call__(self, xyz):
        return self.sample(xyz)

    def _get_num_to_sample(self, npoints) -> int:
        if hasattr(self, "_num_to_sample"):
            return self._num_to_sample
        else:
            return math.floor(npoints * self._ratio)

    def _get_ratio_to_sample(self, batch_size) -> float:
        if hasattr(self, "_ratio"):
            return self._ratio
        else:
            return self._num_to_sample / float(batch_size)

    @abstractmethod
    def sample(self, xyz, feature=None, batch=None):
        pass


class RandomSample(BaseSampler):
    """Random Sample for dense data
        Arguments:
            xyz -- [B, N, 3]
    """

    def sample(self, xyz, **kwargs):
        if len(xyz.shape) != 3:
            raise ValueError(" Expects the xyz tensor to be of dimension 3")
        B, N, _ = xyz.shape
        idx = torch.randint(
            0, N, (B, self._get_num_to_sample(N)), device=xyz.device)
        sampled_xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        # sampled_feature = torch.gather(feature, 2, idx.unsqueeze(1).repeat(1, C, 1))
        return sampled_xyz, idx


def random_sample(xyz, npoint):
    B, N, _ = xyz.shape
    idx = torch.randint(0, N, (B, npoint), device=xyz.device)
    return idx


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set (idx)
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        # output = torch.cuda.IntTensor(B, npoint, device=xyz.device)
        # temp = torch.cuda.FloatTensor(B, N, device=xyz.device).fill_(1e10)
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2_cuda.furthest_point_sampling_wrapper(
            B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint, device=features.device)

        pointnet2_cuda.gather_points_wrapper(
            B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.zeros(
            [B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.gather_points_grad_wrapper(
            B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply
# mark: torch gather is even faster. sampled_xyz = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))


def fps(data, number):
    '''
        data B N C
        number int
    '''
    fps_idx = furthest_point_sample(data[:, :, :3].contiguous(), number)
    fps_data = torch.gather(
        data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    return fps_data


if __name__ == '__main__':
    import time

    B, C, N = 2, 3, 10000
    K = 16
    device = 'cuda'
    points = torch.randn([B, N, 3], device=device, dtype=torch.float)
    print(points.shape, '\n', points)

    nsample = 4096
    idx = furthest_point_sample(points, nsample)

    st = time.time()
    for _ in range(100):
        query1 = torch.gather(
            points, 1, idx.long().unsqueeze(-1).expand(-1, -1, 3))
    print(time.time() - st)
    print(query1.shape)

    st = time.time()
    for _ in range(100):
        query2 = gather_operation(points.transpose(
            1, 2).contiguous(), idx).transpose(1, 2).contiguous()
    print(time.time() - st)
    print(query2.shape)

    print(torch.allclose(query1, query2))
