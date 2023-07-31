""" Maksed PointViT in PyTorch
Copyright 2022@PointNeXt team
"""
import logging
import torch
import torch.nn as nn
from ..build import MODELS
from ..layers.attention import Block
from ..layers import create_norm, create_linearblock
from ..layers.graph_conv import GraphConv, DilatedKNN


@MODELS.register_module()
class MaskedTransformerDecoder(nn.Module):
    """ MaskedTransformerDecoder
    """

    def __init__(self,
                 embed_dim,  # the out features dim of the encoder
                 group_size=32,  # the number of points inside each group
                 decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=16,
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 act_args={'act': 'gelu'},
                 add_pos_each_block=True, 
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

        # ------------------------------------------------------------------
        # MAE decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embed_dim))
        self.decoder_cls_pos = nn.Parameter(torch.randn(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, decoder_embed_dim)
        )
        self.add_pos_each_block = add_pos_each_block
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim, num_heads=decoder_num_heads,
                norm_args=norm_args, act_args=act_args
            )
            for _ in range(decoder_depth)])
        self.decoder_norm = create_norm(norm_args, decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, group_size * 3, bias=True)  # decoder to patch
        # ------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.decoder_cls_pos, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, center_xyz, features, ids_restore):
        # embed tokens
        features = self.decoder_embed(features)
        B, L, C = features.shape  # batch size, length, channels
        
        # in the decoder part. we know the positional encoding of groups
        decoder_pos_embed = torch.cat(
            (self.decoder_cls_pos.expand(B, -1, -1), self.decoder_pos_embed(center_xyz)), dim=1)

        # append mask tokens to sequence
        # use mask tokens to fill the masked features. this is why missing part should be 1 not 0.
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] + 1 - L, 1)    # +1, since features contains additional cls token. 
        x_ = torch.cat([features[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))  # unshuffle
        features = torch.cat([features[:, :1, :], x_], dim=1)  # append cls token

        if self.add_pos_each_block:
            for block in self.decoder_blocks:
                features = block(features + decoder_pos_embed)
        else:
            features = self.pos_drop(features + decoder_pos_embed)
            for block in self.decoder_blocks:
                features = block(features)

        features = self.decoder_norm(features)
        # predictor projection
        features = self.decoder_pred(features)
        # remove cls token
        features = features[:, 1:, :]
        return features


@MODELS.register_module()
class FoldingNet(nn.Module):
    """ FoldingNet.
    Used in many methods, e.g. FoldingNet, PCN, OcCo, Point-BERT
    learning point reconstruction only from global feature
    """
    def __init__(self, in_channels, emb_dims=1024,
                 num_fine=1024,
                 grid_size=2,
                 **kwargs):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

        self.grid_size = grid_size
        self.num_coarse = num_fine // grid_size**2
        self.num_fine = num_fine

        self.folding1 = nn.Sequential(
            nn.Linear(in_channels, emb_dims),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dims, emb_dims),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dims, self.num_coarse * 3))

        self.folding2 = nn.Sequential(
            nn.Linear(emb_dims+2+3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3))

        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.register_buffer('folding_seed', torch.cat([a, b], dim=0).reshape(1, 2, self.grid_size ** 2).transpose(1, 2))
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

    def forward(self, xyz, x, **kwargs):
        B = x.shape[0]
        coarse = self.folding1(x)
        coarse = coarse.view(-1, self.num_coarse, 3)
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1).reshape([-1, self.num_fine, 3])

        seed = self.folding_seed.unsqueeze(1).expand(B, self.num_coarse, -1, -1).reshape(B, self.num_fine, -1)

        x = x.unsqueeze(1).expand(-1, self.num_fine, -1)
        feat = torch.cat([x, seed, point_feat], dim=-1)

        center = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1).reshape([-1, self.num_fine, 3])

        fine = self.folding2(feat) + center

        return coarse, fine


@MODELS.register_module()
class NodeShuffle(nn.Module):
    """ NodeShuffle
        proposed in PU-GCN
    """

    def __init__(self,
                 in_channels,
                 up_ratio=16,
                 emb_dims=1024,
                 k=16,
                 norm_args={'norm': 'bn'},
                 act_args={'act': 'relu'},
                 **kwargs):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

        self.up_ratio = up_ratio
        conv = 'edge'
        self.knn = DilatedKNN(k, 1)
        self.convs = nn.Sequential(
            GraphConv(in_channels, emb_dims, conv, norm_args=norm_args, act_args=act_args),
            GraphConv(emb_dims, emb_dims, conv, norm_args=norm_args, act_args=act_args)
        )
        self.proj = create_linearblock(emb_dims, 3 * up_ratio)
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

    def forward(self, xyz, feature, **kwargs):
        # learn displacement
        B, C, N = feature.shape
        feature = feature.unsqueeze(-1)
        edge_index = self.knn(xyz)
        for conv in self.convs:
            feature = conv(feature, edge_index)
        new_xyz = self.proj(feature.squeeze(-1).transpose(1, 2)).view(B, N, -1, 3) + xyz.unsqueeze(2).repeat(-1, -1, self.up_ratio, -1)
        return new_xyz.view(B, -1, 3)
