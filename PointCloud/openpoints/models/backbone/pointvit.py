""" Vision Transformer (ViT) for Point Cloud Understanding in PyTorch
Hacked together by / Copyright 2020, Ross Wightman
Modified to 3D application by / Copyright 2022@Pix4Point team
"""
import logging
from typing import List
import torch
import torch.nn as nn
from ..layers import create_norm, create_linearblock, create_convblock1d, three_interpolation, \
    furthest_point_sample, random_sample
from ..layers.attention import Block
from .pointnext import FeaturePropogation
from ..build import MODELS, build_model_from_cfg


@MODELS.register_module()
class PointViT(nn.Module):
    """ Point Vision Transformer ++: with early convolutions
    """
    def __init__(self,
                 in_channels=3,
                 embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_args={'NAME': 'PointPatchEmbed', 
                             'num_groups': 256,
                             'group_size': 32,
                             'subsample': 'fps', 
                             'group': 'knn', 
                             'feature_type': 'fj',
                             'norm_args': {'norm': 'in2d'},
                             }, 
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 act_args={'act': 'gelu'},
                 add_pos_each_block=True,
                 global_feat='cls,max',
                 distill=False, 
                 **kwargs
                 ):
        """
        Args:
            in_channels (int): number of input channels. Default: 6. (p + rgb)
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
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        embed_args.in_channels = in_channels
        embed_args.embed_dim = embed_dim
        self.patch_embed = build_model_from_cfg(embed_args)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embed = nn.Sequential(
            create_linearblock(3, 128, norm_args=None, act_args=act_args),
            nn.Linear(128, self.embed_dim)
        )
        if self.patch_embed.out_channels != self.embed_dim: 
            self.proj = nn.Linear(self.patch_embed.out_channels, self.embed_dim)
        else:
            self.proj = nn.Identity() 
        self.add_pos_each_block = add_pos_each_block
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_args=norm_args, act_args=act_args
            )
            for i in range(depth)])
        self.norm = create_norm(norm_args, self.embed_dim)  # Norm layer is extremely important here!
        self.global_feat = global_feat.split(',')
        self.out_channels = len(self.global_feat)*embed_dim
        self.distill_channels = embed_dim
        self.channel_list = self.patch_embed.channel_list
        self.channel_list[-1] = embed_dim

        # distill
        if distill:
            self.dist_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.dist_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.n_tokens = 2
        else:
            self.dist_token = None
            self.n_tokens = 1
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.cls_pos, std=.02)
        if self.dist_token is not None:
            torch.nn.init.normal_(self.dist_token, std=.02)
            torch.nn.init.normal_(self.dist_pos, std=.02)
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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', 'dist_token', 'dist_token'}

    def get_num_layers(self):
        return self.depth

    def forward(self, p, x=None):
        if hasattr(p, 'keys'): 
            p, x = p['pos'], p['x'] if 'x' in p.keys() else None
        if x is None:
            x = p.clone().transpose(1, 2).contiguous()
        p_list, x_list = self.patch_embed(p, x)
        center_p, x = p_list[-1], self.proj(x_list[-1].transpose(1, 2))
        pos_embed = self.pos_embed(center_p)

        pos_embed = [self.cls_pos.expand(x.shape[0], -1, -1), pos_embed]
        tokens = [self.cls_token.expand(x.shape[0], -1, -1), x]
        if self.dist_token is not None:
            pos_embed.insert(1, self.dist_pos.expand(x.shape[0], -1, -1)) 
            tokens.insert(1, self.dist_token.expand(x.shape[0], -1, -1)) 
        pos_embed = torch.cat(pos_embed, dim=1)
        x = torch.cat(tokens, dim=1)
        
        if self.add_pos_each_block:
            for block in self.blocks:
                x = block(x + pos_embed)
        else:
            x = self.pos_drop(x + pos_embed)
            for block in self.blocks:
                x = block(x)
        x = self.norm(x)
        return p_list, x_list, x

    def forward_cls_feat(self, p, x=None):  # p: p, x: features
        _, _, x = self.forward(p, x)
        token_features = x[:, self.n_tokens:, :]
        cls_feats = []
        for token_type in self.global_feat:
            if 'cls' in token_type:
                cls_feats.append(x[:, 0, :])
            elif 'max' in token_type:
                cls_feats.append(torch.max(token_features, dim=1, keepdim=False)[0])
            elif token_type in ['avg', 'mean']:
                cls_feats.append(torch.mean(token_features, dim=1, keepdim=False))
        global_features = torch.cat(cls_feats, dim=1)
        
        if self.dist_token is not None and self.training:
            return global_features, x[:, 1, :]
        else: 
            return global_features

    def forward_seg_feat(self, p, x=None):  # p: p, x: features
        p_list, x_list, x = self.forward(p, x)
        x_list[-1] = x.transpose(1, 2)
        return p_list, x_list


@MODELS.register_module()
class PointViTDecoder(nn.Module):
    """ Decoder of Point Vision Transformer for segmentation.
    """
    def __init__(self,
                 encoder_channel_list: List[int], 
                 decoder_layers: int = 2, 
                 n_decoder_stages: int = 2, # TODO: ablate this 
                 scale: int = 4,
                 channel_scaling: int = 1,  
                 sampler: str = 'fps',
                 global_feat=None, 
                 progressive_input=False,
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        if global_feat is not None:
            self.global_feat = global_feat.split(',')
            num_global_feat = len(self.global_feat)
        else:
            self.global_feat = None
            num_global_feat = 0 
 
        self.in_channels = encoder_channel_list[-1]
        self.scale = scale
        self.n_decoder_stages = n_decoder_stages
        
        if progressive_input:
            skip_dim = [self.in_channels//2**i for i in range(n_decoder_stages-1, 0, -1)]
        else:
            skip_dim = [0 for i in range(n_decoder_stages-1)]
        skip_channels = [encoder_channel_list[0]] + skip_dim
        
        fp_channels = [self.in_channels*channel_scaling]
        for _ in range(n_decoder_stages-1):
           fp_channels.insert(0, fp_channels[0] * channel_scaling) 
        
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages-1, -1):
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages] * (num_global_feat + 1)

        if sampler.lower() == 'fps':
            self.sample_fn = furthest_point_sample
        elif sampler.lower() == 'random':
            self.sample_fn = random_sample
            
    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
                [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f):
        """
        Args:
            p (List(Tensor)): List of tensor for p, length 2, input p and center p
            f (List(Tensor)): List of tensor for feature maps, input features and out features
        """
        if len(p) != (self.n_decoder_stages + 1):
            for i in range(self.n_decoder_stages - 1): 
                pos = p[i] 
                idx = self.sample_fn(pos, pos.shape[1] // self.scale).long()
                new_p = torch.gather(pos, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
                p.insert(1, new_p)
                f.insert(1, None)
        cls_token = f[-1][:, :, 0:1]
        f[-1] = f[-1][:, :, 1:].contiguous()
        
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]
        f_out = f[-len(self.decoder) - 1] 
        
        if self.global_feat is not None:
            global_feats = []
            for token_type in self.global_feat:
                if 'cls' in token_type:
                    global_feats.append(cls_token)
                elif 'max' in token_type:
                    global_feats.append(torch.max(f_out, dim=2, keepdim=True)[0])
                elif token_type in ['avg', 'mean']:
                    global_feats.append(torch.mean(f_out, dim=2, keepdim=True))
            global_feats = torch.cat(global_feats, dim=1).expand(-1, -1, f_out.shape[-1])
            f_out = torch.cat((global_feats, f_out), dim=1)
        return f_out 


@MODELS.register_module()
class PointViTPartDecoder(nn.Module):
    """ Decoder of Point Vision Transformer for segmentation.
    """
    def __init__(self,
                 encoder_channel_list: List[int], 
                 decoder_layers: int = 2, 
                 n_decoder_stages: int = 2,
                 scale: int = 4,
                 channel_scaling: int = 1,  
                 sampler: str = 'fps',
                 global_feat=None, 
                 progressive_input=False,
                 cls_map='pointnet2',
                 num_classes: int = 16,
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        if global_feat is not None:
            self.global_feat = global_feat.split(',')
            num_global_feat = len(self.global_feat)
        else:
            self.global_feat = None
            num_global_feat = 0 
 
        self.in_channels = encoder_channel_list[-1]
        self.scale = scale
        self.n_decoder_stages = n_decoder_stages
        
        if progressive_input:
            skip_dim = [self.in_channels//2**i for i in range(n_decoder_stages-1, 0, -1)]
        else:
            skip_dim = [0 for i in range(n_decoder_stages-1)]
        skip_channels = [encoder_channel_list[0]] + skip_dim
        
        fp_channels = [self.in_channels*channel_scaling]
        for _ in range(n_decoder_stages-1):
           fp_channels.insert(0, fp_channels[0] * channel_scaling) 

        self.cls_map = cls_map
        self.num_classes = num_classes
        act_args = kwargs.get('act_args', {'act': 'relu'}) 
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
            
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages-1, -1):
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages] * (num_global_feat + 1)

        if sampler.lower() == 'fps':
            self.sample_fn = furthest_point_sample
        elif sampler.lower() == 'random':
            self.sample_fn = random_sample
            
    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
                [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f, cls_label):
        """
        Args:
            p (List(Tensor)): List of tensor for p, length 2, input p and center p
            f (List(Tensor)): List of tensor for feature maps, input features and out features
        """
        if len(p) != (self.n_decoder_stages + 1):
            for i in range(self.n_decoder_stages - 1): 
                pos = p[i] 
                idx = self.sample_fn(pos, pos.shape[1] // self.scale).long()
                new_p = torch.gather(pos, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
                p.insert(1, new_p)
                f.insert(1, None)
        cls_token = f[-1][:, :, 0:1]
        f[-1] = f[-1][:, :, 1:].contiguous()
        
        B, N = p[0].shape[0:2]
        if self.cls_map == 'pointnet2':
            cls_one_hot = torch.zeros((B, self.num_classes), device=p[0].device)
            cls_one_hot = cls_one_hot.scatter_(1, cls_label, 1).unsqueeze(-1).repeat(1, 1, N)
            cls_one_hot = self.convc(cls_one_hot)
             
        for i in range(-1, -len(self.decoder), -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]

        i = -len(self.decoder) 
        f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], torch.cat([cls_one_hot, f[i - 1]], 1)], [p[i], f[i]])])[1]

        f_out = f[-len(self.decoder) - 1] 
        
        if self.global_feat is not None:
            global_feats = []
            for token_type in self.global_feat:
                if 'cls' in token_type:
                    global_feats.append(cls_token)
                elif 'max' in token_type:
                    global_feats.append(torch.max(f_out, dim=2, keepdim=True)[0])
                elif token_type in ['avg', 'mean']:
                    global_feats.append(torch.mean(f_out, dim=2, keepdim=True))
            global_feats = torch.cat(global_feats, dim=1).expand(-1, -1, f_out.shape[-1])
            f_out = torch.cat((global_feats, f_out), dim=1)
        return f_out 