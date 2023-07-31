""" Vision Transformer (ViT) for Point Cloud Understanding in PyTorch
Hacked together by / Copyright 2020, Ross Wightman
Modified to 3D application by / Copyright 2022@PointNeXt team
"""
import logging
import torch
import torch.nn as nn
from ..layers import GroupEmbed, KMeansEmbed, TransformerEncoder, create_norm, create_linearblock


class ViTGraph(nn.Module):
    """ Vision Transformer for Graph (where data size varies)
    """
    def __init__(self,
                 in_chans=6, num_classes=40,
                 encoder_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_args={'NAME': 'groupembed', 
                             'num_groups': 256,
                             'group_size': 32, 
                             'embed_dim': 256, 
                             'subsample': 'fps', 
                             'group': 'knn', 
                             'feature_type': 'fj'}, 
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 act_args={'act': 'gelu'},
                 posembed_norm_args=None, 
                 **kwargs
                 ):
        """
        Args:
            num_group (int, tuple): number of patches (groups in 3d)
            group_size (int, tuple): the size (# points) of each group
            in_chans (int): number of input channels. Default: 6. (xyz + rgb)
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

        self.num_classes = num_classes
        self.num_features = self.encoder_dim = encoder_dim  # num_features for consistency with other models
        self.num_tokens = 1
        
        self.embed_layer = embed_args.NAME.lower()
        if self.embed_layer == 'groupembed':
            self.group_embed = GroupEmbed(in_chans=in_chans, **embed_args)
        elif self.embed_layer == 'kmeans':
            self.group_embed = KMeansEmbed(**embed_args)
        
        self.proj_layer = nn.Linear(embed_args.embed_dim, self.encoder_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.encoder_dim))
        self.pos_embed = nn.Sequential(
            create_linearblock(3, 128, norm_args=posembed_norm_args, act_args=act_args), 
            nn.Linear(128, self.encoder_dim)
        )
        # self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = TransformerEncoder(
            embed_dim=self.encoder_dim,
            depth=depth,
            drop_path_rate=dpr,
            attn_drop_rate=attn_drop_rate,
            num_heads=num_heads,
            act_args=act_args, norm_args=norm_args
        )
        self.norm = create_norm(norm_args, self.encoder_dim) or nn.Identity() # Norm layer is extremly important here!
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.cls_pos, std=.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self, xyz, features=None):
        center_xyz, features, _, _ = self.group_embed(xyz, features)
        features = self.proj_layer(features)

        pos_embed = self.pos_embed(center_xyz)
        pos_embed = torch.cat((self.cls_pos.expand(features.shape[0], -1, -1), pos_embed), dim=1)
        
        cls_token = self.cls_token.expand(features.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        features = torch.cat((cls_token, features), dim=1)
        
        # features = self.pos_drop(features + pos_embed)
        features = self.blocks(features, pos_embed)
        features = self.norm(features)
        return center_xyz, features

    # return hierarchical features
    def forward_features(self, xyz, features=None, num_points=None):
        center_xyz, features, _, _ = self.group_embed(xyz, features)
        features = self.proj_layer(features)

        pos_embed = self.pos_embed(center_xyz)
        pos_embed = torch.cat((self.cls_pos.expand(features.shape[0], -1, -1), pos_embed), dim=1)

        cls_token = self.cls_token.expand(features.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        features = torch.cat((cls_token, features), dim=1)

        # features = self.pos_drop(features + pos_embed)
        out_features = self.blocks.forward_features(features, pos_embed, num_points)
        out_features[-1] = self.norm(out_features[-1])
        
        return center_xyz, out_features
