"""PointNext for inputs with variable sizes.

This file is still under development. DO NOT USE. 
"""
from re import X
from typing import List
import logging
import torch
import torch.nn as nn

from torch_geometric.nn import radius as ballquery, radius_graph, knn, knn_graph, fps, knn_interpolate
from torch_scatter import scatter

from ..build import MODELS
from ..layers import create_linearblock, create_linearblock, create_convblock2d, create_act, CHANNEL_MAP, \
    random_sample, three_interpolation


# TODO: PyG for large-scale graph is super slow!!!! especially the KNN cuda. 
def create_grouper(group_args, support_same_as_query=True):
    method = group_args.get('NAME', 'ballquery').lower()
    radius = group_args.get('radius', 0.1)
    nsample = group_args.get('nsample', 16)

    logging.info(group_args)
    if method in ['ballquery', 'ball', 'query']:
        if support_same_as_query:
            # target: center. [center, neighbor]
            grouper = lambda x, b: radius_graph(x, radius, b, max_num_neighbors=nsample, loop=True,
                                                flow='target_to_source')
        else:
            # support, query,
            grouper = lambda x, y, b_x, b_y: ballquery(x, y, radius, b_x, b_y, max_num_neighbors=nsample)
    elif method in ['knn', 'knn_graph']:
        if support_same_as_query:
            grouper = lambda x, b: knn_graph(x, nsample, b, loop=True)
        else:
            grouper = lambda x, y, b_x, b_y: knn(x, y, nsample, b_x, b_y)
    return grouper


class LocalAggregation(nn.Module):
    def __init__(self,
                 channels: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 conv_args=None,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(
                f"kwargs: {kwargs} are not used in {__class__.__name__}")
        channels[0] = CHANNEL_MAP[feature_type](channels[0])
        convs = []
        for i in range(len(channels) - 1):  # #layers in each blocks
            convs.append(create_linearblock(channels[i], channels[i + 1],
                                            norm_args=norm_args,
                                            act_args=None if i == (
                                                    len(channels) - 2) and not last_act else act_args,
                                            **conv_args)
                         )
        self.convs = nn.Sequential(*convs)
        self.grouper = create_grouper(group_args)

        reduction = 'mean' if reduction.lower() == 'avg' else reduction.lower()
        assert reduction in ['sum', 'max', 'mean']
        self.pool = lambda x, index: scatter(x, index, dim=0, reduce=reduction.lower())

    def forward(self, pxb) -> torch.Tensor:
        p, x, b = pxb
        # p: position, f: feature, b: batch
        edge_index = self.grouper(p, b)
        dp = torch.index_select(p, 0, edge_index[1]) - torch.index_select(p, 0, edge_index[0])
        xj = torch.index_select(x, 0, edge_index[1])
        # neighborhood_features
        x = torch.cat((dp, xj), dim=1)
        x = self.convs(x)
        x = self.pool(x, edge_index[0])

        # # """ DEBUG neighbor numbers. """
        # # (we should think about rescaling them at first? )
        # # different samples have very different numb of neighbors within the same radius.
        # if f.shape[-1] != 1:
        #     query_xyz, support_xyz = p, p
        #     # query_xyz, support_xyz = p[10:11], p[10:11]
        #     radius = self.grouper.radius
        #     dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
        #     points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
        #     logging.info(
        #         f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
        # # """ DEBUG end """
        return x


class SetAbstraction(nn.Module):
    """The set abstraction module in PointNet++, which is commonly named as SetAbstraction
    """

    def __init__(self,
                 in_channels, out_channels,
                 layers=2,
                 stride=1,
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 sampler='fps',
                 use_res=True,
                 is_head=False,
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1  # current blocks aggregates all spatial information.
        self.use_res = use_res and not self.all_aggr and not self.is_head

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * (layers - 1) + [out_channels]
        channels[0] = in_channels + 3 * (not is_head)

        if self.use_res:
            self.skipconv = create_linearblock(
                in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
                -1] else nn.Identity()
        create_conv = create_linearblock
        convs = []
        for i in range(len(channels) - 1):
            convs.append(create_conv(channels[i], channels[i + 1],
                                     norm_args=norm_args,
                                     act_args=None if i == len(channels) - 2
                                                      and self.use_res else act_args,
                                     **conv_args)
                         )
        self.convs = nn.Sequential(*convs)
        self.act = create_act(act_args)
        if not is_head:
            if self.all_aggr:
                group_args.nsample = None
                group_args.radius = None
            self.grouper = create_grouper(group_args, support_same_as_query=False)
            self.pool = lambda x, index: scatter(x, index, dim=0, reduce='max')
            if sampler.lower() == 'fps':
                self.sample_fn = fps
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample 

    def forward(self, pxb):
        p, x, b = pxb
        if self.is_head:
            x = self.convs(x)  # (n, c)
        else:
            if not self.all_aggr:
                idx = self.sample_fn(p, b, ratio=(1. / self.stride))
                new_p = p[idx]
                new_b = b[idx]
            else:
                new_p = p
                new_b = b
            # # """ DEBUG neighbor numbers. """
            # query_xyz, support_xyz = new_p, p
            # radius = self.grouper.radius
            # dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            # points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            # logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
            # # """ DEBUG end """
            if self.use_res:
                identity = x[idx]
                identity = self.skipconv(identity)
            edge_index = self.grouper(p, new_p, b, new_b)  # TODO: this part is extremly slow!
            dp = torch.index_select(p, 0, edge_index[1]) - torch.index_select(p, 0, edge_index[0])
            xj = torch.index_select(x, 0, edge_index[1])
            x = self.pool(self.convs(torch.cat((dp, xj), dim=1)), edge_index[0])
            if self.use_res:
                x = self.act(x + identity)
            p = new_p
            b = new_b
        return p, x, b


class FeaturePropogation(nn.Module):
    """The FeaturePropogation module in PointNet++, which is also commonly named as FeaturePropogation
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
                linear1.append(create_linearblock(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_linearblock(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=act_args
                                                ))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pxb1, pxb2=None):
        # pxb1 is with the same size of upsampled points
        if pxb2 is None:
            _, x, b = pxb1  # (B, N, 3), (B, C, N)
            x_global = self.pool(x)
            x = torch.cat(
                (x, self.linear2(x_global).unsqueeze(-1).expand(-1, -1, x.shape[-1])), dim=1)
            x = self.linear1(x)
        else:
            p1, x1, b1 = pxb1
            p2, x2, b2 = pxb2
            x = self.convs(
                torch.cat((x1, knn_interpolate(x2, p2, p1, b2, b1, k=3)), dim=1))
        return x


class Block(nn.Module):
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
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = in_channels * expansion
        self.convs = LocalAggregation([in_channels, in_channels],
                                      norm_args=norm_args, act_args=act_args if num_posconvs > 0 else None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        # point wise after depth wise conv (without last layer)
        for i in range(len(channels) - 1):
            pwconv.append(create_linearblock(channels[i], channels[i + 1],
                                             norm_args=norm_args,
                                             act_args=act_args if
                                             (i != len(channels) - 2) else None,
                                             **conv_args)
                          )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pxb):
        p, x, b = pxb
        identity = x
        x = self.convs(pxb)
        x = self.pwconv(x)
        if x.shape[-1] == identity.shape[-1] and self.use_res:
            x += identity
        x = self.act(x)
        return [p, x, b]


@MODELS.register_module()
class PyGPointNextEncoder(nn.Module):
    r"""The Encoder for PointNext 
    `"DeepGCNs: Can GCNs Go as Deep as CNNs?"
    <https://arxiv.org/abs/1904.03751>`_.
    .. note::
        For an example of using :obj:`PointNextEncoder`, see
        `examples/segmentation/main.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.
    Args:
    """

    def __init__(self,
                 block,
                 blocks,  # depth
                 in_channels=6,
                 width=32,
                 strides=[4, 4, 4, 4],
                 nsample=[16, 16, 16, 16],
                 radius=0.1,
                 radius_scaling=2,
                 nsample_scaling=1,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 norm_args={'norm': 'bn'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 mid_res=False,
                 use_res=True,
                 expansion=1,
                 sa_layers=2,
                 num_posconvs=2,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(
                f"kwargs: {kwargs} are not used in {__class__.__name__}")

        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.c = in_channels
        self.in_channels = in_channels
        self.mid_res = mid_res
        self.use_res = use_res
        self.aggr_args = aggr_args
        self.norm_args = norm_args
        self.act_args = act_args
        self.conv_args = conv_args
        self.expansion = expansion
        self.sa_layers = sa_layers
        self.num_posconvs = num_posconvs
        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

        # width *2 after downsampling.
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
        else:  # radius is a scalar, then create a list
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
                                     norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                     is_head=is_head
                                     ))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, mid_res=self.mid_res, expansion=self.expansion,
                                use_res=self.use_res, num_posconvs=self.num_posconvs
                                ))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0, batch0):
        for i in range(0, len(self.encoder)):
            p0, f0, batch0 = self.encoder[i]([p0, f0, batch0])
        return f0.squeeze(-1)

    def forward_seg_feat(self, p0, f0=None, batch0=None):
        if hasattr(p0, 'keys'):
            p0, f0, batch0 = p0['pos'], p0['x'], p0['batch']
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f, batch = [p0], [f0], [batch0]
        for i in range(0, len(self.encoder)):
            _p, _f, _batch = self.encoder[i]([p[-1], f[-1], batch[-1]])
            p.append(_p)
            f.append(_f)
            batch.append(_batch)
        return p, f, batch

    def forward(self, p0, f0, batch0):
        self.forward_seg_feat(p0, f0, batch0)


@MODELS.register_module()
class PyGPointNextDecoder(nn.Module):
    """PointNextSeg for point cloud segmentation with inputs of variable sizes
    """

    def __init__(self,
                 block,
                 decoder_blocks=[1, 1, 1, 1],  # depth
                 decoder_layers=2,
                 in_channels=6,
                 width=32,
                 strides=[1, 4, 4, 4, 4],
                 nsample=[8, 16, 16, 16, 16],
                 radius=0.1,
                 radius_scaling=2,
                 nsample_scaling=1,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 norm_args={'norm': 'bn'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 mid_res=False,
                 expansion=1,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(
                f"kwargs: {kwargs} are not used in {__class__.__name__}")
        if isinstance(block, str):
            block = eval(block)
        self.blocks = decoder_blocks
        self.decoder_layers = decoder_layers
        self.strides = strides[:-1]
        self.mid_res = mid_res
        self.aggr_args = aggr_args
        self.norm_args = norm_args
        self.act_args = act_args
        self.conv_args = conv_args
        self.c = in_channels
        self.in_channels = in_channels
        self.expansion = expansion

        # self.radii = self._to_full_list(radius, radius_scaling)
        # self.nsample = self._to_full_list(nsample, nsample_scaling)
        # logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

        # width *2 after downsampling.
        channels = []
        initial_width = width
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)

        self.in_channels = channels[-1]
        skip_channels = [in_channels] + channels[:-1]
        fp_channels = [initial_width] + channels[:-1]
        decoder = [[] for _ in range(len(decoder_blocks))]
        for i in range(-1, -len(decoder_blocks) - 1, -1):
            # group_args.radius = self.radii[i]
            # group_args.nsample = self.nsample[i]
            decoder[i] = self._make_dec(skip_channels[i], fp_channels[i], block, decoder_blocks[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[0]

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
        else:  # radius is a scalar, then create a list
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_dec(self, skip_channels, fp_channels, block, blocks, group_args=None, is_head=False):
        """_summary_

        Args:
            skip_channels (int): channels for the incomming upsampled features
            fp_channels (_type_): channels for the output upsampled features
            block (_type_): _description_
            blocks (_type_): _description_
            group_args (_type_, optional): _description_. Defaults to None.
            is_head (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        layers = []
        if is_head:
            mlp = [skip_channels] + [fp_channels] * self.decoder_layers
        else:
            mlp = [skip_channels + self.in_channels] + [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp, not is_head))
        self.in_channels = fp_channels

        # radii = group_args.radius
        # nsample = group_args.nsample
        # for i in range(1, blocks):
        #     group_args.radius = radii[i]
        #     group_args.nsample = nsample[i]
        #     layers.append(block(self.in_channels, self.in_channels,
        #                         aggr_args=self.aggr_args,
        #                         norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
        #                         conv_args=self.conv_args, mid_res=self.mid_res))
        return nn.Sequential(*layers)

    def forward(self, p, f, b):
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1], b[i - 1]], [p[i], f[i], b[i]])])[1]
        return f[-len(self.decoder) - 1]
