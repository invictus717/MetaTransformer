#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq
from openpoints.models.layers.graph_conv import DynConv, GraphConv, ResDynBlock, DenseDynBlock, DilatedKNN
from openpoints.models.layers import create_convblock1d
import logging
from ..build import MODELS


@MODELS.register_module()
class DeepGCN(nn.Module):
    def __init__(self,
                 in_channels=3,
                 channels=64,
                 emb_dims=1024,
                 n_blocks=14,
                 conv='edge',
                 block='res',
                 k=16,
                 epsilon=0.2,
                 use_stochastic=True,
                 use_dilation=True,
                 norm_args={'norm': 'bn'},
                 act_args={'act': 'relu'},
                 conv_args={'order': 'conv-norm-act'},
                 is_seg=False, 
                 **kwargs
                 ):
        """
        Args:
            in_channels (int, optional): Dimension of input. Defaults to 3.
            channels (int, optional): number of channels of deep features. Defaults to 64.
            emb_dims (int, optional): Dimension of embeddings. Defaults to 1024.
            n_blocks (int, optional): number of basic blocks in the backbone. Defaults to 14.
            conv (str, optional): graph conv layer {edge, mr}. Defaults to 'edge'.
            block (str, optional): graph backbone block type {res, plain, dense}. Defaults to 'res'.
            k (int, optional): neighbor num. Defaults to 16.
            epsilon (float, optional): stochastic epsilon for gcn. Defaults to 0.2.
            use_stochastic (bool, optional): stochastic for gcn. Defaults to True.
            use_dilation (bool, optional): dilated gcn. Defaults to True.
            dropout (float, optional): dropout rate. Defaults to 0.5.
            norm_args (dict, optional): batch or instance normalization {bn, in}. Defaults to {'norm': 'bn'}.
            act_args (dict, optional): activation layer {relu, prelu, leakyrelu}. Defaults to {'act': 'relu'}.
        """
        super(DeepGCN, self).__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

        c_growth = channels
        self.n_blocks = n_blocks

        self.knn = DilatedKNN(k, 1, use_stochastic, epsilon)
        self.head = GraphConv(in_channels, channels, conv, bias=False,
                              norm_args=norm_args, act_args=act_args,
                              **conv_args)

        if block.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock(channels + c_growth * i, c_growth, conv,
                                                k, 1 + i, use_stochastic, epsilon,
                                                act_args=act_args, norm_args=norm_args, **conv_args)
                                  for i in range(self.n_blocks - 1)])
            fusion_dims = int(
                (channels + channels + c_growth * (self.n_blocks - 1)) * self.n_blocks // 2)

        elif block.lower() == 'res':
            if use_dilation:
                self.backbone = Seq(*[ResDynBlock(channels, conv,
                                                  k, 1 + i, use_stochastic, epsilon,
                                                  act_args=act_args, norm_args=norm_args, **conv_args)
                                      for i in range(self.n_blocks - 1)])
            else:
                self.backbone = Seq(*[ResDynBlock(channels, conv,
                                                  k, 1, use_stochastic, epsilon,
                                                  act_args=act_args, norm_args=norm_args, **conv_args)
                                      for i in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        else:
            # Plain GCN. No dilation, no stochastic, no residual connections
            stochastic = False
            self.backbone = Seq(*[DynConv(channels, channels, conv,
                                          k, 1, stochastic, epsilon,
                                          act_args=act_args, norm_args=norm_args, **conv_args)
                                  for i in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))

        self.fusion_block = create_convblock1d(fusion_dims, emb_dims,
                                               act_args={'act': 'leakyrelu', 'negative_slope': 0.2},
                                               norm_args=norm_args, **conv_args,
                                               bias=False)
        self.model_init()
        self.maxpool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        self.avgpool = lambda x: torch.mean(x, dim=-1, keepdim=False)
        self.out_channels = emb_dims if is_seg else emb_dims * 2
        
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
                
    def forward_seg_feat(self, pts, features=None):
        fusion = self.forward(pts, features)
        return pts, fusion
    
    def forward_cls_feat(self, pts, features=None):
        fusion = self.forward(pts, features)
        return torch.cat((self.maxpool(fusion), self.avgpool(fusion)), dim=1)
    
    def forward(self, pts, features=None):
        if hasattr(pts, 'keys'):
            pts, features = pts['pos'], pts['x']
        if features is None:
            features = pts.transpose(1, 2).contiguous()
        features = features.unsqueeze(-1)
        feats = [self.head(features, self.knn(pts))]
        for i in range(self.n_blocks - 1):
            feats.append(self.backbone[i](feats[-1]))
        feats = torch.cat(feats, dim=1).squeeze(-1)
        fusion = self.fusion_block(feats)
        return fusion


if __name__ == '__main__':
    device = torch.device('cuda')

    feats = torch.rand((2, 3, 1024), dtype=torch.float).to(device)
    points = torch.rand((2, 1024, 3), dtype=torch.float).to(device)
    num_neighbors = 20

    print('Input size {}'.format(feats.size()))
    net = DeepGCN().to(device)
    print(net)
    out = net(points, feats)

    print('Output size {}'.format(out.size()))
