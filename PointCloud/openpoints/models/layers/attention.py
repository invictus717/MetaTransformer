'''
File Description: attention layer for transformer. borrowed from TIMM
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Mlp, DropPath, trunc_normal_, lecun_normal_
from . import create_norm, create_act


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # this is different from what I understand before. 
        # the num_heads here actually works as groups for shared attentions. it partition the channels to different groups.
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple), shape [B, #Heads, N, C]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_args={'act': 'gelu'}, norm_args={'norm': 'ln'}):
        super().__init__()
        self.norm1 = create_norm(norm_args, dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = create_norm(norm_args, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # invertable bottleneck layer
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_args=act_args, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 act_args={'act': 'gelu'}, norm_args={'norm': 'ln'}
                 ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                norm_args=norm_args, act_args=act_args
            )
            for i in range(depth)])
        self.depth = depth
        # dilation = depth//3
        # self.out_depth = list(range(depth))[dilation-1::dilation]

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            # x = block(x)

            # Injecting the positional information at each block.
            #  (Dehghani et al., 2018) and (Lan et al.,2019) observe better performance
            #  by further injecting the position information at each block
            # Reference: Learning to Encode Position for Transformer with Continuous Dynamical Model.
            # http://proceedings.mlr.press/v119/liu20n/liu20n.pdf
            x = block(x + pos)
        return x

    def forward_features(self, x, pos, num_outs=None):
        dilation = self.depth // num_outs
        out_depth = list(range(self.depth))[(self.depth - (num_outs-1)*dilation -1) :: dilation]
        
        out = []
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in out_depth:
                out.append(x)
        return out
