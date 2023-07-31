"""Official implementation of PointNext
PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies
https://arxiv.org/abs/2206.04670
Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Abed Al Kader Hammoud, Mohamed Elhoseiny, Bernard Ghanem
"""
from typing import List, Type
import logging
import torch
import torch.nn as nn
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, furthest_point_sample, random_sample, three_interpolation, get_aggregation_feautres


def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
    return pool


class LocalAggregation(nn.Module):
    """Local aggregation layer for a set 
    Set abstraction layer abstracts features from a larger set to a smaller set
    Local aggregation layer aggregates features from the same set
    """

    def __init__(self,
                 channels: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 conv_args=None,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        channels[0] = CHANNEL_MAP[feature_type](channels[0])
        convs = []
        for i in range(len(channels) - 1):  # #layers in each blocks
            convs.append(create_convblock2d(channels[i], channels[i + 1],
                                            norm_args=norm_args,
                                            act_args=None if i == (
                                                    len(channels) - 2) and not last_act else act_args,
                                            **conv_args)
                         )
        self.convs = nn.Sequential(*convs)
        self.grouper = create_grouper(group_args)
        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)
        self.feature_type = feature_type

    def forward(self, pf) -> torch.Tensor:
        # p: position, f: feature
        p, f = pf
        # neighborhood_features
        dp, fj = self.grouper(p, p, f)
        fj = get_aggregation_feautres(p, dp, f, fj, self.feature_type)
        f = self.pool(self.convs(fj))
        """ DEBUG neighbor numbers. 
        if f.shape[-1] != 1:
            query_xyz, support_xyz = p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(
                f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
        DEBUG end """
        return f


class SetAbstraction(nn.Module):
    """The modified set abstraction module in PointNet++ with residual connection support
    """

    def __init__(self,
                 in_channels, out_channels,
                 layers=1,
                 stride=1,
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 sampler='fps',
                 feature_type='dp_fj',
                 use_res=False,
                 is_head=False,
                 **kwargs, 
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * \
                   (layers - 1) + [out_channels]
        channels[0] = in_channels if is_head else CHANNEL_MAP[feature_type](channels[0])

        if self.use_res:
            self.skipconv = create_convblock1d(
                in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
                -1] else nn.Identity()
            self.act = create_act(act_args)

        # actually, one can use local aggregation layer to replace the following
        create_conv = create_convblock1d if is_head else create_convblock2d
        convs = []
        for i in range(len(channels) - 1):
            convs.append(create_conv(channels[i], channels[i + 1],
                                     norm_args=norm_args if not is_head else None,
                                     act_args=None if i == len(channels) - 2
                                                      and (self.use_res or is_head) else act_args,
                                     **conv_args)
                         )
        self.convs = nn.Sequential(*convs)
        if not is_head:
            if self.all_aggr:
                group_args.nsample = None
                group_args.radius = None
            self.grouper = create_grouper(group_args)
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
            if sampler.lower() == 'fps':
                self.sample_fn = furthest_point_sample
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample

    def forward(self, pf):
        p, f = pf
        if self.is_head:
            f = self.convs(f)  # (n, c)
        else:
            if not self.all_aggr:
                idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            else:
                new_p = p
            """ DEBUG neighbor numbers. 
            query_xyz, support_xyz = new_p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
            DEBUG end """
            if self.use_res or 'df' in self.feature_type:
                fi = torch.gather(
                    f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
                if self.use_res:
                    identity = self.skipconv(fi)
            else:
                fi = None
            dp, fj = self.grouper(new_p, p, f)
            fj = get_aggregation_feautres(new_p, dp, fi, fj, feature_type=self.feature_type)
            f = self.pool(self.convs(fj))
            if self.use_res:
                f = self.act(f + identity)
            p = new_p
        return p, f


class FeaturePropogation(nn.Module):
    """The Feature Propogation module in PointNet++
    """

    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(
                nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=act_args
                                                ))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None):
        # pfb1 is with the same size of upsampled points
        if pf2 is None:
            _, f = pf1  # (B, N, 3), (B, C, N)
            f_global = self.pool(f)
            f = torch.cat(
                (f, self.linear2(f_global).unsqueeze(-1).expand(-1, -1, f.shape[-1])), dim=1)
            f = self.linear1(f)
        else:
            p1, f1 = pf1
            p2, f2 = pf2
            if f1 is not None:
                f = self.convs(
                    torch.cat((f1, three_interpolation(p1, p2, f2)), dim=1))
            else:
                f = self.convs(three_interpolation(p1, p2, f2))
        return f


class InvResMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,
                 less_act=False,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation([in_channels, in_channels],
                                      norm_args=norm_args, act_args=act_args if num_posconvs > 0 else None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        # point wise after depth wise conv (without last layer)
        for i in range(len(channels) - 1):
            pwconv.append(create_convblock1d(channels[i], channels[i + 1],
                                             norm_args=norm_args,
                                             act_args=act_args if
                                             (i != len(channels) - 2) and not less_act else None,
                                             **conv_args)
                          )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.convs([p, f])
        f = self.pwconv(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f]


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = in_channels * expansion
        self.convs = LocalAggregation([in_channels, in_channels, mid_channels, in_channels],
                                      norm_args=norm_args, act_args=None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        self.act = create_act(act_args)

    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.convs([p, f])
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f]


@MODELS.register_module()
class PointNextEncoder(nn.Module):
    r"""The Encoder for PointNext 
    `"PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies".
    <https://arxiv.org/abs/2206.04670>`_.
    .. note::
        For an example of using :obj:`PointNextEncoder`, see
        `examples/segmentation/main.py <https://github.com/guochengqian/PointNeXt/blob/master/cfgs/s3dis/README.md>`_.
    Args:
        in_channels (int, optional): input channels . Defaults to 4.
        width (int, optional): width of network, the output mlp of the stem MLP. Defaults to 32.
        blocks (List[int], optional): # of blocks per stage (including the SA block). Defaults to [1, 4, 7, 4, 4].
        strides (List[int], optional): the downsampling ratio of each stage. Defaults to [4, 4, 4, 4].
        block (strorType[InvResMLP], optional): the block to use for depth scaling. Defaults to 'InvResMLP'.
        nsample (intorList[int], optional): the number of neighbors to query for each block. Defaults to 32.
        radius (floatorList[float], optional): the initial radius. Defaults to 0.1.
        aggr_args (_type_, optional): the args for local aggregataion. Defaults to {'feature_type': 'dp_fj', "reduction": 'max'}.
        group_args (_type_, optional): the args for grouping. Defaults to {'NAME': 'ballquery'}.
        norm_args (_type_, optional): the args for normalization layer. Defaults to {'norm': 'bn'}.
        act_args (_type_, optional): the args for activation layer. Defaults to {'act': 'relu'}.
        expansion (int, optional): the expansion ratio of the InvResMLP block. Defaults to 4.
        sa_layers (int, optional): the number of MLP layers to use in the SA block. Defaults to 1.
        sa_use_res (bool, optional): wheter to use residual connection in SA block. Set to True only for PointNeXt-S. 
    """

    def __init__(self,
                 in_channels: int = 4,
                 width: int = 32,
                 blocks: List[int] = [1, 4, 7, 4, 4],
                 strides: List[int] = [4, 4, 4, 4],
                 block: str or Type[InvResMLP] = 'InvResMLP',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 0.1,
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args: dict = {'NAME': 'ballquery'},
                 sa_layers: int = 1,
                 sa_use_res: bool = False,
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.aggr_args = aggr_args
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.conv_args = kwargs.get('conv_args', None)
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get('use_res', True)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

        # double width after downsampling.
        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1
            ))
        self.encoder = nn.Sequential(*encoder)
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        layers.append(SetAbstraction(self.in_channels, channels,
                                     self.sa_layers if not is_head else 1, stride,
                                     group_args=group_args,
                                     sampler=self.sampler,
                                     norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                     is_head=is_head, use_res=self.sa_use_res, **self.aggr_args 
                                     ))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(0, len(self.encoder)):
            p0, f0 = self.encoder[i]([p0, f0])
        return f0.squeeze(-1)

    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        for i in range(0, len(self.encoder)):
            _p, _f = self.encoder[i]([p[-1], f[-1]])
            p.append(_p)
            f.append(_f)
        return p, f

    def forward(self, p0, f0=None):
        return self.forward_seg_feat(p0, f0)

@MODELS.register_module()
class PointNextLanguage(nn.Module):
    r"""The Encoder for PointNext 
    `"PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies".
    <https://arxiv.org/abs/2206.04670>`_.
    .. note::
        For an example of using :obj:`PointNextEncoder`, see
        `examples/segmentation/main.py <https://github.com/guochengqian/PointNeXt/blob/master/cfgs/s3dis/README.md>`_.
    Args:
        in_channels (int, optional): input channels . Defaults to 4.
        width (int, optional): width of network, the output mlp of the stem MLP. Defaults to 32.
        blocks (List[int], optional): # of blocks per stage (including the SA block). Defaults to [1, 4, 7, 4, 4].
        strides (List[int], optional): the downsampling ratio of each stage. Defaults to [4, 4, 4, 4].
        block (strorType[InvResMLP], optional): the block to use for depth scaling. Defaults to 'InvResMLP'.
        nsample (intorList[int], optional): the number of neighbors to query for each block. Defaults to 32.
        radius (floatorList[float], optional): the initial radius. Defaults to 0.1.
        aggr_args (_type_, optional): the args for local aggregataion. Defaults to {'feature_type': 'dp_fj', "reduction": 'max'}.
        group_args (_type_, optional): the args for grouping. Defaults to {'NAME': 'ballquery'}.
        norm_args (_type_, optional): the args for normalization layer. Defaults to {'norm': 'bn'}.
        act_args (_type_, optional): the args for activation layer. Defaults to {'act': 'relu'}.
        expansion (int, optional): the expansion ratio of the InvResMLP block. Defaults to 4.
        sa_layers (int, optional): the number of MLP layers to use in the SA block. Defaults to 1.
        sa_use_res (bool, optional): wheter to use residual connection in SA block. Set to True only for PointNeXt-S. 
    """

    def __init__(self,
                 in_channels: int = 4,
                 width: int = 32,
                 blocks: List[int] = [1, 4, 7, 4, 4],
                 strides: List[int] = [4, 4, 4, 4],
                 block: str or Type[InvResMLP] = 'InvResMLP',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 0.1,
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 embed_dim=384, depth=12,
    num_heads=6, mlp_ratio=4., qkv_bias=False,
    num_groups=256, group_size=32,
    subsample='fps',  # random, FPS
                 group='ballquery', 
    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 state_args = {'pretrained': True , 'frozen': True, 'encode':'embedding'}, 
                 embed_args={'NAME': 'PointPatchEmbed', 
                             'num_groups': 256,
                             'group_size': 32,
                             'subsample': 'fps', 
                             'group': 'knn', 
                             'feature_type': 'fj',
                             'norm_args': {'norm': 'in2d'},
                             }, 
                 act_args={'act': 'gelu'},
                 norm_args={'norm': 'ln', 'eps': 1.0e-6}, 
                 group_args: dict = {'NAME': 'ballquery'},
                 sa_layers: int = 1,
                 sa_use_res: bool = False,
                 distill=False, 
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.aggr_args = aggr_args
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.conv_args = kwargs.get('conv_args', None)
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get('use_res', True)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

        # double width after downsampling.
        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1
            ))
        self.encoder = nn.Sequential(*encoder)
        self.out_channels = channels[-1]
        self.channel_list = channels
        # ---------------------------------------------- Language Relevant
        from ..layers import create_norm, create_linearblock
        from ..build import MODELS, build_model_from_cfg
        from ..layers.attention import Block
        self.num_features = self.embed_dim = embed_dim 
        self.pos_embed = nn.Sequential(
            create_linearblock(3, 128, norm_args=None, act_args=act_args),
            nn.Linear(128, self.embed_dim)
        )
        self.patch_embed = build_model_from_cfg(embed_args)
        if self.patch_embed.out_channels != self.embed_dim: 
            self.proj = nn.Linear(self.patch_embed.out_channels, self.embed_dim)
        else:
            self.proj = nn.Identity() 
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        # self.weight = nn.Linear(257,768)
        self.conv = nn.Conv1d(257,512,kernel_size=1)
        self.norm = create_norm(norm_args, self.embed_dim)
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.weight = nn.Linear(1024,512)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_args=norm_args, act_args=act_args
            )
            for i in range(depth)])
        if distill:
            self.dist_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.dist_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.n_tokens = 2
        else:
            self.dist_token = None
            self.n_tokens = 1
        # define language models
        from transformers import T5Config, T5Model
        self.state = state_args
        if self.state.pretrained:
            self.t5 = T5Model.from_pretrained('t5-base') # Load pretrained weights
        else:
            self.t5_config = T5Config()
            self.t5_config.d_model = 768
            self.t5 =  T5Model(self.t5_config)  # Random initialize
        if self.state.frozen:
            self.t5.eval()
        else:
            self.t5.train()

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        layers.append(SetAbstraction(self.in_channels, channels,
                                     self.sa_layers if not is_head else 1, stride,
                                     group_args=group_args,
                                     sampler=self.sampler,
                                     norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                     is_head=is_head, use_res=self.sa_use_res, **self.aggr_args 
                                     ))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        # x = f0
        # p_list, x_list = self.patch_embed(p0, f0)
        # center_p, x = p_list[-1], self.proj(x_list[-1].transpose(1, 2))

        # pos_embed = self.pos_embed(center_p)
        # pos_embed = [self.cls_pos.expand(x.shape[0], -1, -1), pos_embed]
        # tokens = [self.cls_token.expand(x.shape[0], -1, -1), x]
        # if self.dist_token is not None:
        #     pos_embed.insert(1, self.dist_pos.expand(x.shape[0], -1, -1)) 
        #     tokens.insert(1, self.dist_token.expand(x.shape[0], -1, -1)) 
        # pos_embed = torch.cat(pos_embed, dim=1)
        # x = torch.cat(tokens, dim=1)
        # hidden_state=self.t5.forward(inputs_embeds=pos_embed,decoder_inputs_embeds=pos_embed)[0]
        # for block in self.blocks:
        #     x = block(x + hidden_state)
        # x = self.norm(x)
        # x = self.pool(self.conv(x))
        # p = 3.0
        # x = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
        for i in range(0, len(self.encoder)):
            p0, f0 = self.encoder[i]([p0, f0])
        # out = self.weight(torch.concat([f0.squeeze(-1),x],dim=1))
        f0 = f0.expand(32,512,768)

        out = self.t5.forward(inputs_embeds=f0,decoder_inputs_embeds=f0)[0]
        
        return out

    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        for i in range(0, len(self.encoder)):
            _p, _f = self.encoder[i]([p[-1], f[-1]])
            p.append(_p)
            f.append(_f)
        return p, f

    def forward(self, p0, f0=None):
        return self.forward_seg_feat(p0, f0)

@MODELS.register_module()
class PointNextDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_stages: int = 4, 
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        if len(skip_channels) < decoder_stages:
            skip_channels.insert(0, kwargs.get('in_channels', 3))
        # the output channel after interpolation
        fp_channels = encoder_channel_list[:decoder_stages]

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f):
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]
        return f[-len(self.decoder) - 1]


@MODELS.register_module()
class PointNextPartDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_blocks: List[int] = [1, 1, 1, 1],
                 decoder_strides: List[int] = [4, 4, 4, 4],
                 act_args: str = 'relu',
                 cls_map='pointnet2',
                 num_classes: int = 16,
                 cls2partembed=None,
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        fp_channels = encoder_channel_list[:-1]
        
        # the following is for decoder blocks
        self.conv_args = kwargs.get('conv_args', None)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)
        block = kwargs.get('block', 'InvResMLP')
        if isinstance(block, str):
            block = eval(block)
        self.blocks = decoder_blocks
        self.strides = decoder_strides
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.expansion = kwargs.get('expansion', 4)
        radius = kwargs.get('radius', 0.1)
        nsample = kwargs.get('nsample', 16)
        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        self.cls_map = cls_map
        self.num_classes = num_classes
        self.use_res = kwargs.get('use_res', True)
        group_args = kwargs.get('group_args', {'NAME': 'ballquery'})
        self.aggr_args = kwargs.get('aggr_args', 
                                    {'feature_type': 'dp_fj', "reduction": 'max'}
                                    )  
        if self.cls_map == 'curvenet':
            # global features
            self.global_conv2 = nn.Sequential(
                create_convblock1d(fp_channels[-1] * 2, 128,
                                   norm_args=None,
                                   act_args=act_args))
            self.global_conv1 = nn.Sequential(
                create_convblock1d(fp_channels[-2] * 2, 64,
                                   norm_args=None,
                                   act_args=act_args))
            skip_channels[0] += 64 + 128 + 16  # shape categories labels
        elif self.cls_map == 'pointnet2':
            self.convc = nn.Sequential(create_convblock1d(16, 64,
                                                          norm_args=None,
                                                          act_args=act_args))
            skip_channels[0] += 64  # shape categories labels

        elif self.cls_map == 'pointnext':
            self.global_conv2 = nn.Sequential(
                create_convblock1d(fp_channels[-1] * 2, 128,
                                   norm_args=None,
                                   act_args=act_args))
            self.global_conv1 = nn.Sequential(
                create_convblock1d(fp_channels[-2] * 2, 64,
                                   norm_args=None,
                                   act_args=act_args))
            skip_channels[0] += 64 + 128 + 50  # shape categories labels
            self.cls2partembed = cls2partembed
        elif self.cls_map == 'pointnext1':
            self.convc = nn.Sequential(create_convblock1d(50, 64,
                                                          norm_args=None,
                                                          act_args=act_args))
            skip_channels[0] += 64  # shape categories labels
            self.cls2partembed = cls2partembed

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i], group_args=group_args, block=block, blocks=self.blocks[i])

        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels, group_args=None, block=None, blocks=1):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp, act_args=self.act_args))
        self.in_channels = fp_channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def forward(self, p, f, cls_label):
        B, N = p[0].shape[0:2]
        if self.cls_map == 'curvenet':
            emb1 = self.global_conv1(f[-2])
            emb1 = emb1.max(dim=-1, keepdim=True)[0]  # bs, 64, 1
            emb2 = self.global_conv2(f[-1])
            emb2 = emb2.max(dim=-1, keepdim=True)[0]  # bs, 128, 1
            cls_one_hot = torch.zeros((B, self.num_classes), device=p[0].device)
            cls_one_hot = cls_one_hot.scatter_(1, cls_label, 1).unsqueeze(-1)
            cls_one_hot = torch.cat((emb1, emb2, cls_one_hot), dim=1)
            cls_one_hot = cls_one_hot.expand(-1, -1, N)
        elif self.cls_map == 'pointnet2':
            cls_one_hot = torch.zeros((B, self.num_classes), device=p[0].device)
            cls_one_hot = cls_one_hot.scatter_(1, cls_label, 1).unsqueeze(-1).repeat(1, 1, N)
            cls_one_hot = self.convc(cls_one_hot)
        elif self.cls_map == 'pointnext':
            emb1 = self.global_conv1(f[-2])
            emb1 = emb1.max(dim=-1, keepdim=True)[0]  # bs, 64, 1
            emb2 = self.global_conv2(f[-1])
            emb2 = emb2.max(dim=-1, keepdim=True)[0]  # bs, 128, 1
            self.cls2partembed = self.cls2partembed.to(p[0].device)
            cls_one_hot = self.cls2partembed[cls_label.squeeze()].unsqueeze(-1)
            cls_one_hot = torch.cat((emb1, emb2, cls_one_hot), dim=1)
            cls_one_hot = cls_one_hot.expand(-1, -1, N)
        elif self.cls_map == 'pointnext1':
            self.cls2partembed = self.cls2partembed.to(p[0].device)
            cls_one_hot = self.cls2partembed[cls_label.squeeze()].unsqueeze(-1).expand(-1, -1, N)
            cls_one_hot = self.convc(cls_one_hot)

        for i in range(-1, -len(self.decoder), -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i-1], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]

        # TODO: study where to add this ? 
        f[-len(self.decoder) - 1] = self.decoder[0][1:](
            [p[1], self.decoder[0][0]([p[1], torch.cat([cls_one_hot, f[1]], 1)], [p[2], f[2]])])[1]

        return f[-len(self.decoder) - 1]
