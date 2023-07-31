#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from openpoints.models.layers import create_convblock2d, create_grouper, furthest_point_sample, random_sample
import logging
from ..build import MODELS


@MODELS.register_module()
class GroupPointNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels=64,
                 n_blocks=5,
                 sample_fn='furthest_point_sample',  # random, FPS
                 sample_ratio=0.25,
                 group_args={'group': 'knn',
                             'radius': 0.1,
                             'nsample': 20
                             },
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

        self.sample_ratio = sample_ratio
        self.sample_fn = eval(sample_fn)
        self.grouper = create_grouper(group_args)

        in_channels *= 2
        backbone = []
        for i in range(self.n_blocks - 2):
            backbone.append(create_convblock2d(in_channels, channels,
                                               act_args=act_args, norm_args=norm_args, **conv_args,
                                               bias=False))
            in_channels = channels
        self.backbone = nn.Sequential(*backbone)
        self.maxpool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        self.avgpool = lambda x: torch.mean(x, dim=-1, keepdim=False)
        self.out_channels = channels*2
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

    def forward(self, p, f=None):   # position, features
        if f is None:
            f = p.transpose(1, 2).contiguous().unsqueeze(-1)

        # sample points
        idx = self.sample_fn(p, int(p.shape[1] * self.sample_ratio)).long()
        p1 = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

        dp, gf = self.grouper(p1, p, f)  # relative position, grouped features (neighborhood features)
        gf = torch.cat((dp, gf), dim=1)
        f = self.backbone(gf)
        return self.maxpool(f)

    def forward_final_feat(self, pts, features=None):
        return self.forward(pts, features)

    def forward_cls_feat(self, pts, features=None):
        f = self.forward_final_feat(pts, features)
        return torch.cat((self.maxpool(f), self.avgpool(f)), dim=1)

    def ssl_forward(self, p, f):   # position, features
        f = self.backbone(f)
        return self.maxpool(f)
