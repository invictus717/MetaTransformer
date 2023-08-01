from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def calc_square_dist(self, a, b, norm=True):
        """
        Calculating square distance between a and b
        a: [bs, n, c]
        b: [bs, m, c]
        """
        n = a.shape[1]
        m = b.shape[1]
        num_channel = a.shape[-1]
        a_square = a.unsqueeze(dim=2)  # [bs, n, 1, c]
        b_square = b.unsqueeze(dim=1)  # [bs, 1, m, c]
        a_square = torch.sum(a_square * a_square, dim=-1)  # [bs, n, 1]
        b_square = torch.sum(b_square * b_square, dim=-1)  # [bs, 1, m]
        a_square = a_square.repeat((1, 1, m))  # [bs, n, m]
        b_square = b_square.repeat((1, n, 1))  # [bs, n, m]

        coor = torch.matmul(a, b.transpose(1, 2))  # [bs, n, m]

        if norm:
            dist = a_square + b_square - 2.0 * coor  # [bs, npoint, ndataset]
            # dist = torch.sqrt(dist)
        else:
            dist = a_square + b_square - 2 * coor
            # dist = torch.sqrt(dist)
        return dist

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                pointnet2_utils.farthest_point_sample(xyz, self.npoint)
            ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method


class PointnetSAModuleMSG_WithSampling(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with specific downsampling and multiscale grouping """

    def __init__(self, *,
                 npoint_list: List[int],
                 sample_range_list: List[int],
                 sample_type_list: List[int],
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],                 
                 use_xyz: bool = True,
                 dilated_group=False,
                 pool_method='max_pool',
                 aggregation_mlp: List[int],
                 confidence_mlp: List[int],
                 num_class):
        """
        :param npoint_list: list of int, number of samples for every sampling type
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling type
        :param sample_type_list: list of str, list of used sampling type, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_group: whether to use dilated group
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        :param num_class: int, class for process
        """
        super().__init__()
        self.sample_type_list = sample_type_list
        self.sample_range_list = sample_range_list
        self.dilated_group = dilated_group

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        out_channels = 0
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if self.dilated_group:
                if i == 0:
                    min_radius = 0.
                else:
                    min_radius = radii[i-1]
                self.groupers.append(
                    pointnet2_utils.QueryDilatedAndGroup(
                        radius, min_radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
            else:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroup(
                        radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1],
                              kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method

        if (aggregation_mlp is not None) and (len(aggregation_mlp) != 0) and (len(self.mlps) > 0):
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_layer = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_layer = None

        if (confidence_mlp is not None) and (len(confidence_mlp) != 0):
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, num_class, kernel_size=1, bias=True),
            )
            self.confidence_layers = nn.Sequential(*shared_mlp)
        else:
            self.confidence_layers = None


    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, cls_features: torch.Tensor = None, new_xyz=None, ctr_xyz=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param cls_features: (B, N, num_class) tensor of the descriptors of the the confidence (classification) features 
        :param new_xyz: (B, M, 3) tensor of the xyz coordinates of the sampled points
        "param ctr_xyz: tensor of the xyz coordinates of the centers 
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
            cls_features: (B, npoint, num_class) tensor of confidence (classification) features
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous() 
        sampled_idx_list = []
        if ctr_xyz is None:
            last_sample_end_index = 0
            
            for i in range(len(self.sample_type_list)):
                sample_type = self.sample_type_list[i]
                sample_range = self.sample_range_list[i]
                npoint = self.npoint_list[i]

                if npoint <= 0:
                    continue
                if sample_range == -1: #全部
                    xyz_tmp = xyz[:, last_sample_end_index:, :]
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:, :].contiguous()  
                    cls_features_tmp = cls_features[:, last_sample_end_index:, :] if cls_features is not None else None 
                else:
                    xyz_tmp = xyz[:, last_sample_end_index:sample_range, :].contiguous()
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:sample_range, :]
                    cls_features_tmp = cls_features[:, last_sample_end_index:sample_range, :] if cls_features is not None else None 
                    last_sample_end_index += sample_range

                if xyz_tmp.shape[1] <= npoint: # No downsampling
                    sample_idx = torch.arange(xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32) * torch.ones(xyz_tmp.shape[0], xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32)

                elif ('cls' in sample_type) or ('ctr' in sample_type):
                    cls_features_max, class_pred = cls_features_tmp.max(dim=-1)
                    score_pred = torch.sigmoid(cls_features_max) # B,N
                    score_picked, sample_idx = torch.topk(score_pred, npoint, dim=-1)           
                    sample_idx = sample_idx.int()

                elif 'D-FPS' in sample_type or 'DFS' in sample_type:
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)

                elif 'F-FPS' in sample_type or 'FFS' in sample_type:
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)

                elif sample_type == 'FS':
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx_1 = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                    sample_idx_2 = pointnet2_utils.furthest_point_sample(xyz_tmp, npoint)
                    sample_idx = torch.cat([sample_idx_1, sample_idx_2], dim=-1)  # [bs, npoint * 2]
                elif 'Rand' in sample_type:
                    sample_idx = torch.randperm(xyz_tmp.shape[1],device=xyz_tmp.device)[None, :npoint].int().repeat(xyz_tmp.shape[0], 1)
                elif sample_type == 'ds_FPS' or sample_type == 'ds-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        radii = per_xyz.norm(dim=-1) -5 
                        storted_radii, indince = radii.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)
                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                elif sample_type == 'ry_FPS' or sample_type == 'ry-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        ry = torch.atan(per_xyz[:,0]/per_xyz[:,1])
                        storted_ry, indince = ry.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)

                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                sampled_idx_list.append(sample_idx)

            sampled_idx_list = torch.cat(sampled_idx_list, dim=-1) 
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sampled_idx_list).transpose(1, 2).contiguous()

        else:
            new_xyz = ctr_xyz

        if len(self.groupers) > 0:
            for i in range(len(self.groupers)):
                new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
                if self.pool_method == 'max_pool':
                    new_features = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                elif self.pool_method == 'avg_pool':
                    new_features = F.avg_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                else:
                    raise NotImplementedError

                new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
                new_features_list.append(new_features)

            new_features = torch.cat(new_features_list, dim=1)

            if self.aggregation_layer is not None:
                new_features = self.aggregation_layer(new_features)
        else:
            new_features = pointnet2_utils.gather_operation(features, sampled_idx_list).contiguous()

        if self.confidence_layers is not None:
            cls_features = self.confidence_layers(new_features).transpose(1, 2)
            
        else:
            cls_features = None

        return new_xyz, new_features, cls_features

class Vote_layer(nn.Module):
    """ Light voting module with limitation"""
    def __init__(self, mlp_list, pre_channel, max_translate_range):
        super().__init__()
        self.mlp_list = mlp_list
        if len(mlp_list) > 0:
            for i in range(len(mlp_list)):
                shared_mlps = []

                shared_mlps.extend([
                    nn.Conv1d(pre_channel, mlp_list[i], kernel_size=1, bias=False),
                    nn.BatchNorm1d(mlp_list[i]),
                    nn.ReLU()
                ])
                pre_channel = mlp_list[i]
            self.mlp_modules = nn.Sequential(*shared_mlps)
        else:
            self.mlp_modules = None

        self.ctr_reg = nn.Conv1d(pre_channel, 3, kernel_size=1)
        self.max_offset_limit = torch.tensor(max_translate_range).float() if max_translate_range is not None else None
       

    def forward(self, xyz, features):
        xyz_select = xyz
        features_select = features

        if self.mlp_modules is not None: 
            new_features = self.mlp_modules(features_select) #([4, 256, 256]) ->([4, 128, 256])            
        else:
            new_features = new_features
        
        ctr_offsets = self.ctr_reg(new_features) #[4, 128, 256]) -> ([4, 3, 256])

        ctr_offsets = ctr_offsets.transpose(1, 2)#([4, 256, 3])
        feat_offets = ctr_offsets[..., 3:]
        new_features = feat_offets
        ctr_offsets = ctr_offsets[..., :3]
        
        if self.max_offset_limit is not None:
            max_offset_limit = self.max_offset_limit.view(1, 1, 3)            
            max_offset_limit = self.max_offset_limit.repeat((xyz_select.shape[0], xyz_select.shape[1], 1)).to(xyz_select.device) #([4, 256, 3])
      
            limited_ctr_offsets = torch.where(ctr_offsets > max_offset_limit, max_offset_limit, ctr_offsets)
            min_offset_limit = -1 * max_offset_limit
            limited_ctr_offsets = torch.where(limited_ctr_offsets < min_offset_limit, min_offset_limit, limited_ctr_offsets)
            vote_xyz = xyz_select + limited_ctr_offsets
        else:
            vote_xyz = xyz_select + ctr_offsets

        return vote_xyz, new_features, xyz_select, ctr_offsets


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
            pool_method=pool_method
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()

        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    pass