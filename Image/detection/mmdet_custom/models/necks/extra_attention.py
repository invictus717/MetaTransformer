import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.builder import NECKS
from timm.models.layers import trunc_normal_, DropPath
import math
import torch
import torch.utils.checkpoint as cp


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@NECKS.register_module()
class ExtraAttention(BaseModule):
    
    def __init__(self,
                 in_channels,
                 num_head,
                 with_ffn=True,
                 ffn_ratio=4.0,
                 norm_layer=nn.LayerNorm,
                 drop_path=0.,
                 init_values=None,
                 with_cp=False,
                 use_final_norm=True):
        super(ExtraAttention, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.norm1 = norm_layer(in_channels[-1])
        self.attn = Attention(dim=in_channels[-1], num_heads=num_head,
                              qkv_bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_final_norm = use_final_norm
        self.with_cp = with_cp
        
        if with_ffn:
            self.norm2 = norm_layer(in_channels[-1])
            hidden_features = int(in_channels[-1] * ffn_ratio)
            self.ffn = Mlp(in_features=in_channels[-1], hidden_features=hidden_features)
        else:
            self.ffn = None
        
        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((in_channels[-1])), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((in_channels[-1])), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
        
        if self.use_final_norm:
            self.final_norm = norm_layer(in_channels[-1])
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    @auto_fp16()
    def forward(self, inputs):
        
        def _inner_forward(feat):
            b, c, h, w = feat.shape
            feat = feat.flatten(2).transpose(1, 2)
            
            # add layer scale
            if self.gamma_1 is not None:  # self-attention
                feat = feat + self.gamma_1 * self.drop_path(self.attn(self.norm1(feat)))
            else:
                feat = feat + self.drop_path(self.attn(self.norm1(feat)))
            
            if self.ffn is not None:  # ffn
                if self.gamma_2 is not None:
                    feat = feat + self.gamma_2 * self.drop_path(self.ffn(self.norm2(feat)))
                else:
                    feat = feat + self.drop_path(self.ffn(self.norm2(feat)))
            if self.use_final_norm:
                feat = self.final_norm(feat)
            feat = feat.transpose(1, 2).reshape(b, c, h, w).contiguous()
            return feat
        
        """Forward function."""
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        assert len(inputs) == len(self.in_channels)
        
        feat = inputs[-1]
        
        if self.with_cp and feat.requires_grad:
            feat = cp.checkpoint(_inner_forward, feat)
        else:
            feat = _inner_forward(feat)
        
        inputs[-1] = feat  # replace original feature map
        
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        
        return inputs