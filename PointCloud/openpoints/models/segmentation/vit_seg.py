'''
Copyright 2022@Pixel2Point
File Description: Vision Transformer for Point Cloud Segmentation
'''
from base64 import encode
import torch
import torch.nn as nn
from .head_seg import SceneSegHeadPointNet
from ..backbone import PointViT, PointNet2Decoder, PointNetFPModule
from ..layers import furthest_point_sample
from ..build import MODELS
import logging


# class PointNet2Decoder(nn.Module):
#     """PointNet++ MSG.
#     """

#     def __init__(self,
#                  conv_args: dict,
#                  norm_args: dict,
#                  act_args: dict,
#                  fp_mlps=None,
#                  **kwargs
#                  ):
#         super().__init__()
#         if kwargs:
#             logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

#         self.FP_modules = nn.ModuleList()
#         for k in range(fp_mlps.__len__()):
#             pre_channel = fp_mlps[k + 1][-1] if k + 1 < len(fp_mlps) else skip_channel_list[-1]
#             self.FP_modules.append(
#                 PointNetFPModule(
#                     [pre_channel + skip_channel_list[k]] + fp_mlps[k],
#                     conv_args, norm_args, act_args
#                 )
#             )
#         self.num_point_features = fp_mlps[0][-1]

#     def forward(self, l_xyz, l_features):
#         for i in range(-1, -(len(self.FP_modules) + 1), -1):  # 768 features
#             l_features[i - 1] = self.FP_modules[i](  # 288 -> 128
#                 l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
#             )  # (B, C, N)
#         return l_features[0]



@MODELS.register_module()
class PointVitSeg(nn.Module):
    def __init__(self,
                 in_channels=6, num_classes=40,
                 encoder_dim=768,  depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_args={'NAME': 'groupembed',
                             'num_groups': 256,
                             'group_size': 32,
                             'embed_dim': 256,
                             'subsample': 'fps',
                             'group': 'knn',
                             'feature_type': 'fj'},
                 conv_args={},
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 act_args={'act': 'gelu'},
                 posembed_norm_args=None,
                 num_points=None, 
                 fp_mlps=None,
                 **kwargs
                 ):
        """ViT for point cloud segmentation

        Args:
            cfg (dict): configuration
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.encoder = PointViT(
            in_channels, num_classes,
            encoder_dim, depth,
            num_heads, mlp_ratio, qkv_bias,
            drop_rate, attn_drop_rate, drop_path_rate,
            embed_args, norm_args, act_args, posembed_norm_args
        )
        skip_channel_list = [in_channels]*(len(num_points)+1) + [encoder_dim]
        # self.decoder = PointNetFPModule(fp_mlps, conv_args, norm_args, act_args)
        self.decoder = PointNet2Decoder(conv_args, norm_args, act_args,
                                        skip_channel_list, fp_mlps)
        self.head = SceneSegHeadPointNet(num_classes=num_classes, in_channels=fp_mlps[0][0])
        self.num_points = num_points 
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyz, features):
        center_xyz, l_feature = self.encoder(xyz, features)
        
        # to B, C, N
        l_feature = l_feature[:, 1:, :].transpose(1, 2).contiguous()
        
        # generate l_xyz
        l_xyz, l_features = [xyz], [features]
        for npoints in self.num_points[:-1]:
            idx = furthest_point_sample(xyz, npoints).long()
            l_xyz.append(torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3)))
            l_features.append(torch.gather(features, -1, idx.unsqueeze(1).expand(-1, features.shape[1], -1))
)
        l_xyz.append(center_xyz)
        l_features.append(l_feature)
        
        """Debug
        for i in l_xyz:
            print(i.shape)
            
        for i in l_features:
            print(i.shape)
        """    
        up_features = self.decoder(l_xyz, l_features)
        # up_features = self.decoder(xyz, l_xyz, features, l_features)
        return self.head(up_features)
