"""PointNet++ variants Implementation.
1. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
    by Charles R. Qi, Li (Eric) Yi, Hao Su, Leonidas J. Guibas from Stanford University.
2. ASSANet: An Anisotropical Separable Set Abstraction forEfficient Point Cloud Representation Learning
    by Guocheng Qian, etal. @ NeurIPS 2021 Spotlight
Reference:
https://github.com/sshaoshuai/Pointnet2.PyTorch
"""
from typing import List, Optional

import torch
import torch.nn as nn
import logging
from ..layers import furthest_point_sample, random_sample,  LocalAggregation, three_interpolation, create_convblock1d # grid_subsampling,
from ..build import MODELS


class PointNetSAModuleMSG(nn.Module):
    """Original PointNet set abstraction layer with multi-scale grouping in parallel fashion
        PointNet++ Set Abstraction Module:
        1. For each module, downsample the point cloud ( support set) once as query set
        2. For each downsampled point cloud, query neighbors from the support set multiple times
        3. In each neighbor querying, perform local aggregations
    """

    def __init__(self,
                 stride: int,
                 radii: List[float],
                 nsamples: List[int],
                 channel_list: List[List[int]],
                 aggr_args: dict,
                 group_args: dict,
                 conv_args: dict,
                 norm_args: dict,
                 act_args: dict,
                 sampler='fps',
                 use_res=False,
                 query_as_support=False,
                 voxel_size=0.1,
                 **kwargs
                 ):
        super().__init__()
        self.stride = stride
        self.blocks = len(channel_list)
        self.query_as_support=query_as_support

        # build the sampling layer:
        if 'fps' in sampler.lower() or 'furthest' in sampler.lower():
            self.sample_fn = furthest_point_sample
        elif 'random' in sampler.lower():
            self.sample_fn = random_sample

        # holder for the grouper and convs (MLPs, \etc)
        self.local_aggregations = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            channels = channel_list[i]
            if i > 0 and query_as_support:
                channels[0] = channel_list[i-1][-1]

            group_args.radius = radius
            group_args.nsample = nsample
            # build the convs
            self.local_aggregations.append(
                LocalAggregation(channels, aggr_args, conv_args, norm_args, act_args,
                                 group_args, use_res))

    def forward(self,
                support_xyz: torch.Tensor,
                support_features: torch.Tensor = None,
                query_xyz=None):
        """
        :param support_xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param support_features: (B, C, N) tensor of the descriptors of the the features
        :param query_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, \sum_k(mlps[k][-1], npoint)) tensor of the new_features descriptors
        """
        new_features_list = []
        if query_xyz is None and self.stride > 1:
            idx = self.sample_fn(
                support_xyz, support_xyz.shape[1] // self.stride).long()
            query_xyz = torch.gather(
                support_xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        else:
            query_xyz = support_xyz
            idx = None

        for i in range(self.blocks):
            new_features = self.local_aggregations[i](
                query_xyz, support_xyz, support_features, query_idx=idx)
            new_features_list.append(new_features)

            if self.query_as_support:
                support_xyz = query_xyz
                support_features = new_features
                idx = None
        return query_xyz, torch.cat(new_features_list, dim=1)  # concatenate


class PointNetFPModule(nn.Module):
    r"""Feature Propagation module in PointNet++.
    Propagates the features of one set to another"""

    def __init__(self, mlp: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 ):
        """
        :param mlp: list of channel sizes
        """
        super().__init__()
        # Local Aggregations or Not
        convs = []
        for i in range(len(mlp) - 1):
            convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                            norm_args=norm_args, act_args=act_args,
                                            ))
        self.convs = nn.Sequential(*convs)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features. To upsample!!!
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            interpolated_feats = three_interpolation(
                unknown, known, known_feats)
        else:
            interpolated_feats = known_feats.expand(
                *known_feats.size()[0:2], unknown.size(1))
        if unknow_feats is not None:
            new_features = torch.cat(
                [unknow_feats, interpolated_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats
        new_features = self.convs(new_features)
        return new_features


@MODELS.register_module()
class PointNet2Encoder(nn.Module):
    """Encoder for PointNet++ and ASSANet
    Args:
        in_channels (int): input feature size
        radius (List[float]orfloat): radius to use at each stage or initial raidus
        num_samples (List[int]orint): neighbood size to use at each block or initial neighbohood size
        aggr_args (dict): dict of configurations for local aggregation
        group_args (dict): dict of configurations for neighborhood query
        conv_args (dict): dict of configurations for convolution layers
        norm_args (dict): dict of configurations for normalization layers
        act_args (dict): dict of configurations for activation layers
        blocks (Optional[List], optional): number of bloks per stage. Defaults to None.
        mlps (_type_, optional): channel size per block. Defaults to None.
        width (Optional[int], optional): initial channel size. Defaults to None.
        strides (List[int], optional): stride for each stage. Defaults to [4, 4, 4, 4].
        layers (int, optional): number of MLP layers in each SA block. Defaults to 3.
        width_scaling (int, optional): scale ratio of channel size after downsampling. Defaults to 2.
        radius_scaling (int, optional): scale ratio of radius after each stage. Defaults to 2.
        block_radius_scaling (int, optional): scale ratio of radius after each block. Defaults to 1.
        nsample_scaling (int, optional): scale ratio of radius after each stage. Defaults to 1.
        sampler (str, optional): the method for point cloud downsampling. Defaults to 'fps'.
        use_res (bool, optional): whether use residual connections in SA block. Defaults to False.  Set to True in ASSANet
        stem_conv (bool, optional): whether using stem MLP. Defaults to False.
        stem_aggr (bool, optional): whether use an additional local aggregation before downsampling. Defaults to False. Set to True in ASSANet
        double_last_channel (bool, optional): whether double the channel sizes of the last layer inside each block. Defaults to False. Set to False in ASSANet
        query_as_support (bool, optional): whether to use query set as support set. Defaults to False. Set to True in ASSANet
    """
    def __init__(self,
                 in_channels: int,
                 radius: List[float] or float,
                 num_samples: List[int] or int,
                 aggr_args: dict,
                 group_args: dict,
                 conv_args: dict,
                 norm_args: dict,
                 act_args: dict,
                 blocks: Optional[List] = None,
                 mlps=None,
                 width: Optional[int] = None,
                 strides: List[int] = [4, 4, 4, 4],
                 layers: int = 3,
                 width_scaling: int = 2,
                 radius_scaling: int = 2,
                 block_radius_scaling: int = 1,
                 nsample_scaling: int = 1,
                 sampler: str = 'fps',
                 use_res=False,
                 stem_conv=False,
                 stem_aggr=False,
                 double_last_channel=True,
                 query_as_support=False,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(
                f"kwargs: {kwargs} are not used in {__class__.__name__}")
        stages = len(strides)
        self.strides = strides
        self.blocks = blocks if mlps is None else [len(mlp) for mlp in mlps]
        radius = self._to_full_list(radius,
                                    blocks=self.blocks,
                                    param_scaling=radius_scaling,
                                    block_param_scaling=block_radius_scaling)
        num_samples = self._to_full_list(num_samples,
                                         blocks=self.blocks,
                                         param_scaling=nsample_scaling)
        self.radius = radius
        self.num_samples = num_samples
        logging.info(f'radius is modified to {radius}')
        logging.info(f'num_samples is modified to {num_samples}')

        # patchify stem
        self.stem_conv = stem_conv
        self.stem_aggr = stem_aggr
        if stem_conv:
            width = width if width is not None else mlps[0][0][0]
            self.conv1 = create_convblock1d(
                in_channels, width, norm_args=None, act_args=None)
            if stem_aggr:
                channels = [width] * (layers + 1)
                group_args.radius = radius[0][0]
                group_args.nsample = num_samples[0][0]
                self.stem = LocalAggregation(channels, aggr_args, conv_args, norm_args, act_args,
                                             group_args, use_res)
            in_channels = width

        if mlps is None:
            assert width is not None
            assert layers is not None
            assert strides is not None
            mlps = []
            for i in range(stages):
                if not double_last_channel:
                    mlps.append([[width] * layers] * (self.blocks[i]))
                    width = width * width_scaling if strides[i] > 1 else width
                else:
                    mlps_temp = [width] * (layers - 1)
                    width = width * width_scaling if strides[i] > 1 else width
                    mlps_temp += [width]
                    mlps.append([mlps_temp] + [[width] * layers]
                                * (self.blocks[i] - 1))

            logging.info(f'channels is modified to {mlps}')
        self.mlps = mlps

        self.SA_modules = nn.ModuleList()
        skip_channel_list = [in_channels]
        for k in range(stages):
            # sample times = # stages
            # obtain the in_channels and output channels from the configuration
            channel_list = mlps[k].copy()
            channel_out = 0
            for idx in range(channel_list.__len__()):
                channel_list[idx] = [in_channels] + channel_list[idx]
                channel_out += channel_list[idx][-1]  # concatenate
            # for each sample, may query points multiple times, the query radii and nsamples may be different
            self.SA_modules.append(
                PointNetSAModuleMSG(
                    stride=strides[k],
                    radii=radius[k],
                    nsamples=num_samples[k],
                    channel_list=channel_list,
                    aggr_args=aggr_args,
                    group_args=group_args,
                    conv_args=conv_args,
                    norm_args=norm_args,
                    act_args=act_args,
                    sampler=sampler,
                    use_res=use_res,
                    query_as_support=query_as_support
                )
            )
            skip_channel_list.append(channel_out)
            in_channels = channel_out
        self.out_channels = channel_out
        self.channel_list = skip_channel_list

    def _to_full_list(self, param, blocks, param_scaling=1, block_param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != blocks[i]:
                    value += [value[-1]] * (blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar, then create a list
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * blocks[i])
                else:
                    param_list.append(
                        [param] + [param * block_param_scaling] * (blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def forward_cls_feat(self, xyz, features=None):
        if hasattr(xyz, 'keys'):
            xyz, features = xyz['pos'], xyz['x']
        if features is None:
            features = xyz.clone().transpose(1, 2).contiguous()
        if self.stem_conv:
            features = self.conv1(features)
        if self.stem_aggr:
            features = self.stem(xyz, xyz, features)
        for i in range(len(self.SA_modules)):
            # query_xyz is None. query for neighbors
            xyz, features = self.SA_modules[i](xyz, features)
        return features.squeeze(-1)

    def forward_seg_feat(self, xyz, features=None):
        if hasattr(xyz, 'keys'):
            xyz, features = xyz['pos'], xyz['x']
        if features is None:
            features = xyz.clone().transpose(1, 2).contiguous()
        xyz = xyz.contiguous()
        if self.stem_conv:
            features = self.conv1(features)
        if self.stem_aggr:
            features = self.stem(xyz, xyz, features)
            # _, features = self.stem_sa(xyz, features, xyz)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            # query_xyz is None. query for neighbors
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        return l_xyz, l_features

    def forward(self, xyz, features=None):
        if hasattr(xyz, 'keys'):
            xyz, features = xyz['pos'], xyz['x']
        return self.forward_seg_feat(xyz, features)


@MODELS.register_module()
class PointNet2Decoder(nn.Module):
    """Decoder for PointNet++
    """
    def __init__(self,
                 encoder_channel_list: List[int],
                 mlps=None,
                 fp_mlps=None,
                 decoder_layers=1,
                 **kwargs
                 ):
        super().__init__()
        skip_channel_list = encoder_channel_list
        self.FP_modules = nn.ModuleList()
        if fp_mlps is None:
            fp_mlps = [[mlps[0][0][0]] * (decoder_layers + 1)]
            fp_mlps += [[c] * (decoder_layers + 1)
                        for c in skip_channel_list[1:-1]]
        for k in range(fp_mlps.__len__()):
            pre_channel = fp_mlps[k + 1][-1] if k + 1 < len(fp_mlps)\
                else skip_channel_list[-1]
            self.FP_modules.append(
                PointNetFPModule(
                    [pre_channel + skip_channel_list[k]] + fp_mlps[k]
                )
            )
        self.out_channels = fp_mlps[0][-1]

    def forward(self, l_xyz, l_features):
        for i in range(-1, -(len(self.FP_modules) + 1), -1):  # 768 features
            l_features[i - 1] = self.FP_modules[i](  # 288 -> 128
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )  # (B, C, N)
        return l_features[0]


@MODELS.register_module()
class PointNet2PartDecoder(nn.Module):
    """PointNet++ MSG.
    """
    def __init__(self,
                 in_channels: int,
                 radius: List[int] or int,
                 num_samples: List[int],
                 group_args: dict,
                 conv_args: dict,
                 norm_args: dict,
                 act_args: dict,
                 mlps=None,
                 blocks: Optional[List] = None,
                 width: Optional[int] = None,
                 strides=[4, 4, 4, 4],
                 layers=3,
                 fp_mlps=None,
                 decoder_layers=1,
                 decocder_aggr_args=None,
                 width_scaling=2,
                 radius_scaling=2,
                 nsample_scaling=1,
                 use_res=False,
                 stem_conv=False,
                 double_last_channel=False,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(
                f"kwargs: {kwargs} are not used in {__class__.__name__}")
        stages = len(strides)
        self.strides = strides

        self.blocks = blocks if mlps is None else [len(mlp) for mlp in mlps]
        radius = self._to_full_list(
            radius, self.blocks, param_scaling=radius_scaling)
        num_samples = self._to_full_list(
            num_samples, self.blocks, param_scaling=nsample_scaling)
        self.radius = radius
        self.num_samples = num_samples

        if stem_conv:
            in_channels = width

        if mlps is None:
            assert width is not None
            assert layers is not None
            assert strides is not None
            mlps = []
            for i in range(stages):
                if not double_last_channel:
                    # only add the output channels, not in_channels.
                    width = width * width_scaling if strides[i] > 1 else width
                    mlps.append([[width] * layers] * (self.blocks[i]))
                else:
                    mlps_temp = [width] * (layers - 1)
                    width = width * 2 if strides[i] > 1 else width
                    mlps_temp += [width]
                    mlps.append([mlps_temp] + [[width] * layers]
                                * (self.blocks[i] - 1))

            logging.info(f'channels is modified to {mlps}')
        self.mlps = mlps

        skip_channel_list = [in_channels]
        for k in range(stages):
            # sample times = # stages
            # obtain the in_channels and output channels from the configuration
            channel_list = mlps[k].copy()
            channel_out = 0
            for idx in range(channel_list.__len__()):
                channel_list[idx] = [in_channels] + channel_list[idx]
                channel_out += channel_list[idx][-1]  # concatenate
            skip_channel_list.append(channel_out)
            in_channels = channel_out

        self.FP_modules = nn.ModuleList()
        if fp_mlps is None:
            fp_mlps = [[mlps[0][0][0]] * (decoder_layers + 1)]
            fp_mlps += [[c] * (decoder_layers + 1)
                        for c in skip_channel_list[1:-1]]

        skip_channel_list[0] += 16
        for k in range(fp_mlps.__len__()):
            pre_channel = fp_mlps[k + 1][-1] if k + \
                1 < len(fp_mlps) else skip_channel_list[-1]
            self.FP_modules.append(
                PointNetFPModule(
                    [pre_channel + skip_channel_list[k]] + fp_mlps[k],
                )
            )
        self.out_channels = fp_mlps[0][-1]

    def _to_full_list(self, param, blocks, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != blocks[i]:
                    value += [value[-1]] * (blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar, then create a list
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def forward(self, l_xyz, l_features, cls_label):
        for i in range(-1, -(len(self.FP_modules)), -1):  # 768 features
            l_features[i - 1] = self.FP_modules[i](  # 288 -> 128
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )
        B, N = l_xyz[0].shape[0:2]
        cls_one_hot = torch.zeros((B, 16), device=l_xyz[0].device)
        cls_one_hot = cls_one_hot.scatter_(
            1, cls_label, 1).unsqueeze(-1).repeat(1, 1, N)
        l_features[0] = self.FP_modules[0](
            l_xyz[0], l_xyz[1], torch.cat([cls_one_hot, l_features[0]], 1),
            l_features[1]
        )
        return l_features[0]
