#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from openpoints.models.layers.graph_conv import GraphConv
from openpoints.models.layers import create_convblock1d
from openpoints.models.layers.group import KNNGroup, QueryAndGroup
import logging
from ..build import MODELS


@MODELS.register_module()
class BallDGCNN(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels=64,
                 embed_dim=1024,
                 n_blocks=5,
                 conv='edge',
                 k=20,
                 group='ballquery', 
                 radius=0.1,
                 norm_args={'norm': 'bn'},
                 act_args={'act': 'leakyrelu', 'negative_slope': 0.2},
                 conv_args={'order': 'conv-act-norm'},
                 **kwargs
                 ):
        """

        Args:
            in_channels (int, optional): Dimension of input. Defaults to 3.
            channels (int, optional): number of channels of deep features. Defaults to 64.
            embed_dim (int, optional): Dimension of embeddings. Defaults to 1024.
            n_blocks (int, optional): number of basic blocks in the backbone. Defaults to 14.
            conv (str, optional): graph conv layer {edge, mr}. Defaults to 'edge'.
            block (str, optional): graph backbone block type {res, plain, dense}. Defaults to 'res'.
            k (int, optional): neighbor num. Defaults to 20 for 1024 points, and 40 for 2048 points.
            epsilon (float, optional): stochastic epsilon for gcn. Defaults to 0.2.
            use_stochastic (bool, optional): stochastic for gcn. Defaults to True.
            use_dilation (bool, optional): dilated gcn. Defaults to True.
            dropout (float, optional): dropout rate. Defaults to 0.5.
            norm_args (dict, optional): batch or instance normalization {bn, in}. Defaults to {'norm': 'bn'}.
            act_args (dict, optional): activation layer {relu, prelu, leakyrelu}. Defaults to {'act': 'relu'}.
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

        self.n_blocks = n_blocks

        if 'ball' in group or 'query' in group:
            self.grouper = QueryAndGroup(nsample=k, radius=radius,
                                         return_only_idx=True
                                         )
        elif 'knn' in group.lower():
            self.grouper = KNNGroup(k, return_only_idx=True)

        self.head = GraphConv(in_channels, channels, conv,
                              norm_args=norm_args, act_args=act_args,
                              **conv_args)
        out_channels = [channels]
        in_channels = channels
        backbone = []
        for i in range(self.n_blocks - 2):
            backbone.append(GraphConv(in_channels, channels, conv,
                                      act_args=act_args, norm_args=norm_args, **conv_args)
                            )
            out_channels.append(channels)
            in_channels = channels
            channels *= 2
        self.backbone = nn.Sequential(*backbone)
        fusion_dims = int(sum(out_channels))
        self.fusion_block = create_convblock1d(fusion_dims, embed_dim,
                                               act_args=act_args, norm_args=norm_args, **conv_args,
                                               bias=False)
        self.maxpool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        self.avgpool = lambda x: torch.mean(x, dim=-1, keepdim=False)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d)):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, pts, features=None):
        if features is None:
            features = pts.transpose(1, 2).contiguous().unsqueeze(-1)
        edge_index = self.grouper(pts, pts)
        feats = [self.head(features, edge_index)]
        for i in range(self.n_blocks - 2):
            feats.append(self.backbone[i](feats[-1], edge_index))
        feats = torch.cat(feats, dim=1).squeeze(-1)
        fusion = self.fusion_block(feats)
        return fusion

    def forward_final_feat(self, pts, features=None):
        return self.forward(pts, features)

    def forward_cls_feat(self, pts, features=None):
        fusion = self.forward_final_feat(pts, features)
        return torch.cat((self.maxpool(fusion), self.avgpool(fusion)), dim=1)


if __name__ == '__main__':
    device = torch.device('cuda')

    feats = torch.rand((2, 3, 1024), dtype=torch.float).to(device)
    points = torch.rand((2, 1024, 3), dtype=torch.float).to(device)
    num_neighbors = 20

    print('Input size {}'.format(feats.size()))
    net = DGCNN().to(device)
    print(net)
    out = net(points, feats)

    print('Output size {}'.format(out.size()))
