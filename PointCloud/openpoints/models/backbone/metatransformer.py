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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_scatter import scatter
from timm.models.resnet import BasicBlock
import numpy as np
from timm.models import create_model
from torchvision import transforms

@MODELS.register_module()
class MetaTransformer(nn.Module):
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

        from timm.models.vision_transformer import Block
        ckpt = torch.load("Meta-Transformer_base_patch16_encoder.pth")
        self.blocks = nn.Sequential(*[
                    Block(
                        dim=768,
                        num_heads=12,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        norm_layer=nn.LayerNorm,
                        act_layer=nn.GELU
                    )
                    for i in range(12)])
        self.blocks.load_state_dict(ckpt,strict=True)
        for p in self.blocks.parameters():
            p.requires_grad = False
        trainables = [p for p in self.blocks.parameters() if p.requires_grad]
        print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in self.blocks.parameters()) / 1e6))
        print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
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


    
def rotate_point_clouds(pc, rotation_matrix, use_normals=False):
    '''
        Input: 
            pc  B N 3
            R   3 3
        Output:
            B N 3
    '''
    if not use_normals:
        new_pc = torch.einsum('bnc, dc -> bnd', pc, rotation_matrix.float().to(pc.device))
    else:
        new_pc = torch.einsum('bnc, dc -> bnd', pc[:, :, :3], rotation_matrix.float().to(pc.device))
        new_normal = torch.einsum('bnc, dc -> bnd', pc[:, :, 3:], rotation_matrix.float().to(pc.device))
        new_pc = torch.cat([new_pc, new_normal], dim=-1)
    return new_pc

def rotate_angle_vector(theta, v):
    '''
        theta: B 1
        v:  B 3
    '''
    cos_a = torch.cos(theta)
    sin_a = torch.sin(theta)
    x, y, z = v[:, 0:1], v[:, 1:2], v[:, 2:3]
    
    R = torch.stack([
        torch.cat([cos_a+(1-cos_a)*x*x, (1-cos_a)*x*y-sin_a*z, (1-cos_a)*x*z+sin_a*y], dim=-1) , # [b1 b1 b1]
        torch.cat([(1-cos_a)*y*x+sin_a*z, cos_a+(1-cos_a)*y*y, (1-cos_a)*y*z-sin_a*x], dim=-1) ,
        torch.cat([(1-cos_a)*z*x-sin_a*y, (1-cos_a)*z*y+sin_a*x, cos_a+(1-cos_a)*z*z], dim=-1) 
    ], dim = 1)

    return R

def rotate_theta_phi(angles):
    '''
        angles: B, 2
    '''
    assert len(angles.shape) == 2
    B = angles.size(0)
    theta, phi = angles[:, 0:1], angles[:, 1:2]

    v1 = torch.Tensor([[0, 0, 1]]).expand(B, -1) # B 3
    v2 = torch.cat([torch.sin(theta) , -torch.cos(theta), torch.zeros_like(theta)], dim=-1) # B 3

    R1_inv = rotate_angle_vector(-theta, v1)
    R2_inv = rotate_angle_vector(-phi, v2)
    R_inv = R1_inv @ R2_inv

    return R_inv


def rotate_point_clouds_batch(pc, rotation_matrix, use_normals=False):
    '''
        Input: 
            pc  B N 3
            R   B 3 3
        Output:
            B N 3
    '''
    if not use_normals:
        new_pc = torch.einsum('bnc, bdc -> bnd', pc, rotation_matrix.float().to(pc.device))
    else:
        new_pc = torch.einsum('bnc, bdc -> bnd', pc[:, :, :3], rotation_matrix.float().to(pc.device))
        new_normal = torch.einsum('bnc, bdc -> bnd', pc[:, :, 3:], rotation_matrix.float().to(pc.device))
        new_pc = torch.cat([new_pc, new_normal], dim=-1)
    return new_pc



def euler2mat(angle: torch.Tensor):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
    source
    https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """

    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zero = z.detach() * 0
    one = zero.detach() + 1
    zmat = torch.stack([cosz, -sinz, zero,
                        sinz, cosz, zero,
                        zero, zero, one], dim=_dim).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zero, siny,
                        zero, one, zero,
                        -siny, zero, cosy], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([one, zero, zero,
                        zero, cosx, -sinx,
                        zero, sinx, cosx], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    # print(rot_mat)
    return rot_mat

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., selfatt=True, kv_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        if selfatt:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, z=None):
        if z is None:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
        else:
            q = self.to_q(x)
            k, v = self.to_kv(z).chunk(2, dim=-1)
            qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., selfatt=True, kv_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                                       dropout=dropout, selfatt=selfatt, kv_dim=kv_dim)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, z=None):
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            x = ff(x) + x
        return x
    
class MultiViewFusionV2(nn.Module):
    def __init__(
        self,
        img_size,
        graph_dim,
        num_views,   
    ):
        super(MultiViewFusionV2, self).__init__()
        self.img_size = img_size
        self.graph_dim = graph_dim
        self.num_views = num_views
        self.kernel_size = 7
        self.sub_image_size = self.img_size // self.kernel_size
        self.conv1 = nn.Conv2d(in_channels=graph_dim, out_channels=128, kernel_size=self.kernel_size,
                               stride=self.kernel_size, bias=False)
        self.transformer = Transformer(128, depth=1, heads=8, dim_head=128, mlp_dim=256, selfatt=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=graph_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=graph_dim * 2, out_channels=graph_dim, kernel_size=1),
            nn.GELU()
        )

    def forward(self, feats: torch.Tensor):
        shortcut = feats
        feats = self.conv1(feats)
        B, C = feats.shape[:2]
        feats = rearrange(feats.reshape(B // self.num_views, self.num_views, C, self.sub_image_size ** 2),
                          'B N C S -> B (N S) C')
        feats = self.transformer(feats)
        feats = rearrange(feats, 'B (N S) C -> (B N) C S', N=self.num_views).reshape(B, C, self.sub_image_size,
                                                                                     self.sub_image_size)
        feats = F.interpolate(feats, size=self.img_size)
        feats = self.conv2(feats)
        feats = self.conv3(torch.cat((feats, shortcut), dim=1))
        return feats
    
    
class ProjEnc(nn.Module):
    def __init__(
        self,
        local_size,
        trans_dim,
        graph_dim,
        imgblock_dim,
        img_size,
        obj_size,
        num_views,
        atten_fusion,
        imagenet_default_mean,
        imagenet_default_std
    ):
        super().__init__()
        self.local_size = local_size
        self.trans_dim = trans_dim
        self.graph_dim = graph_dim
        self.imgblock_dim = imgblock_dim
        self.img_size = img_size
        self.obj_size = obj_size
        self.num_views = num_views
        self.atten_fusion = atten_fusion
        self.imagenet_mean = torch.Tensor(imagenet_default_mean)
        self.imagenet_std = torch.Tensor(imagenet_default_std)

        self.input_trans = nn.Conv1d(3, self.trans_dim, 1)
        self.graph_layer = nn.Sequential(nn.Conv2d(self.trans_dim * 2, self.graph_dim, kernel_size=1, bias=False),
                                         nn.GroupNorm(4, self.graph_dim),
                                         nn.LeakyReLU(negative_slope=0.2)
                                         )
        self.proj_layer = nn.Conv1d(self.graph_dim, self.graph_dim, kernel_size=1)

        self.img_block = nn.Sequential(
            BasicBlock(self.graph_dim, self.graph_dim),
            nn.Conv2d(self.graph_dim, self.graph_dim, kernel_size=1),
        )

        if self.atten_fusion:
            self.fusion = MultiViewFusionV2(
                img_size,
                graph_dim,
                num_views
            )

        self.img_layer = nn.Conv2d(self.graph_dim, 3, kernel_size=1)

        self.offset = torch.Tensor([[-1, -1], [-1, 0], [-1, 1],
                                    [0, -1], [0, 0], [0, 1],
                                    [1, -1], [1, 0], [1, 1]])

    @staticmethod
    def get_graph_feature(coor_q: torch.Tensor, x_q: torch.Tensor, coor_k: torch.Tensor, x_k: torch.Tensor, k: int):
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            idx = knn_point(k, coor_k.transpose(1, 2).contiguous(), coor_q.transpose(1, 2).contiguous())  # B G k
            idx = idx.transpose(1, 2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, original_pc: torch.Tensor, pc: torch.Tensor):
        B, N, _ = pc.shape

        pc_range = pc.max(dim=1)[0] - pc.min(dim=1)[0]  
        grid_size = pc_range[:, :2].max(dim=-1)[0] / (self.obj_size - 3)  
        idx_xy = torch.floor(
            (pc[:, :, :2] - pc.min(dim=1)[0][:, :2].unsqueeze(dim=1)) / grid_size.unsqueeze(dim=1).unsqueeze(
                dim=2))  
        idx_xy_dense = (idx_xy.unsqueeze(dim=2) + self.offset.unsqueeze(dim=0).unsqueeze(dim=0).to(pc.device)).view(
            idx_xy.size(0), N * 9, 2) + 1
        idx_xy_dense_center = torch.floor(
            (idx_xy_dense.max(dim=1)[0] + idx_xy_dense.min(dim=1)[0]) / 2).int()  
        offset_x = self.obj_size / 2 - idx_xy_dense_center[:, 0:1] - 1
        offset_y = self.obj_size / 2 - idx_xy_dense_center[:, 1:2] - 1
        idx_xy_dense_offset = idx_xy_dense + torch.cat([offset_x, offset_y], dim=1).unsqueeze(dim=1)

        original_pc = original_pc.transpose(1, 2).contiguous()  
        f = self.input_trans(original_pc)  #
        f = self.get_graph_feature(original_pc, f, original_pc, f, self.local_size)
        f = self.graph_layer(f)
        f = f.max(dim=-1, keepdim=False)[0]  

        f = self.proj_layer(f).transpose(1, 2).contiguous() 

        f_dense = f.unsqueeze(dim=2).expand(-1, -1, 9, -1).contiguous().view(f.size(0), N * 9, self.graph_dim)
        assert idx_xy_dense_offset.min() >= 0 and idx_xy_dense_offset.max() <= (self.obj_size - 1), str(
            idx_xy_dense_offset.min()) + '-' + str(idx_xy_dense_offset.max())
        new_idx_xy_dense = idx_xy_dense_offset[:, :, 0] * self.obj_size + idx_xy_dense_offset[:, :, 1]  
        out = scatter(f_dense, new_idx_xy_dense.long(), dim=1, reduce="sum")

        if out.size(1) < self.obj_size * self.obj_size:
            delta = self.obj_size * self.obj_size - out.size(1)
            zero_pad = torch.zeros(out.size(0), delta, out.size(2)).to(out.device)
            res = torch.cat([out, zero_pad], dim=1).reshape((out.size(0), self.obj_size, self.obj_size, out.size(2)))
        else:
            res = out.reshape((out.size(0), self.obj_size, self.obj_size, out.size(2)))
        if self.obj_size < self.img_size:
            pad_size = self.img_size - self.obj_size
            zero_pad_h = torch.zeros(out.size(0), int(pad_size // 2), self.obj_size, out.size(2)).to(out.device)
            zero_pad_w = torch.zeros(out.size(0), self.img_size, int(pad_size // 2), out.size(2)).to(out.device)
            res = torch.cat([zero_pad_h, res, zero_pad_h], dim=1)
            res = torch.cat([zero_pad_w, res, zero_pad_w], dim=2)

        res = res.permute(0, 3, 1, 2).contiguous()
   
        img_feat = self.img_block(res)
        if self.atten_fusion:
            img_feat = self.fusion(img_feat)

        img = self.img_layer(img_feat)  

        mean_vec = self.imagenet_mean.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).to(img.device)  
        std_vec = self.imagenet_std.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).to(img.device)   
        img = torch.sigmoid(img)
        img_norm = img.sub(mean_vec).div(std_vec)

        return img_norm
    
class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc

train_transforms = transforms.Compose(
    [
        PointcloudScaleAndTranslate(),
    ]
)

@MODELS.register_module()
class MetaTransformer_MultiView(nn.Module):
    def __init__(
        self,
        num_views,
        base_model_variant,
        proj_args,
        checkpoint_path,
        update_type,
        use_normals,
    ):
        super().__init__()
        self.use_normals = use_normals
        TRANS = -1.4
        self.views = nn.Parameter(
            torch.tensor(
                np.array([
                    [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
                    [[1 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
                    [[2 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
                    [[3 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
                    [[5 * np.pi / 4, -np.pi / 4, np.pi / 2], [0, 0, TRANS]],
                    [[5 * np.pi / 4, np.pi / 4, np.pi / 2], [0, 0, TRANS]],
                    [[7 * np.pi / 4, -np.pi / 4, np.pi / 2], [0, 0, TRANS]],
                    [[7 * np.pi / 4, np.pi / 4, np.pi / 2], [0, 0, TRANS]],
                    [[0, -np.pi / 2, np.pi / 2], [0, 0, TRANS]],
                    [[0, np.pi / 2, np.pi / 2], [0, 0, TRANS]]
                ]), dtype=torch.float32
            ), requires_grad=False)
        
        self.num_views = num_views
        self.update_type = update_type
        self.enc = ProjEnc(**proj_args)

        self.base_model_name = base_model_variant
        self.base_model = create_model('vit_base_patch16_224') 
        
        ckpt = torch.load("Meta-Transformer_base_patch16_encoder.pth")
        self.base_model.blocks.load_state_dict(ckpt)
    

    def _set_last_feat_hook(self):
        def layer_hook(module, inp, out):
            self.last_feats = out

        return self.base_model.layer4.register_forward_hook(layer_hook)

    def _fix_weight(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

        if self.update_type is not None:
            for name, param in self.base_model.named_parameters():
                if self.update_type in name:
                    param.requires_grad = True
            print('Learnable {} parameters!'.format(self.update_type))

    def forward(self, pc, original_pc):
        original_pc = torch.repeat_interleave(original_pc, self.num_views, dim=0)
        pc = self.point_transform(pc)
        img = self.enc(original_pc, pc) 

        out = self.base_model.forward_features(img)
        return out
    
    def forward_cls_feat(self, p, x=None):
        if hasattr(p, 'keys'):
            p, x = p['pos'], p['x'] if 'x' in p.keys() else None
        if x is None:
            x = p.clone().transpose(1, 2).contiguous()
            
        if self.training:
            angle = torch.stack([torch.rand(p.size(0)) * 1.9 + 0.04,                       # 0.04 ~ 1.94pi
                             (torch.rand(p.size(0)) * 0.2 - 0.4)], dim=-1) * math.pi   # -0.4 ~ -0.2 pi
            rotation_matrix = rotate_theta_phi(angle)
            input_pc = rotate_point_clouds_batch(p, rotation_matrix, use_normals=self.use_normals).contiguous()  
            input_pc = train_transforms(input_pc)  
            
        else:
            angle = torch.stack([torch.zeros(p.size(0)) + 0.04,                       # 0.04 ~ 1.94pi
                             (torch.zeros(p.size(0)) - 0.4)], dim=-1) * math.pi   # -0.4 ~ -0.2 pi
            rotation_matrix = rotate_theta_phi(angle)
            input_pc = rotate_point_clouds_batch(p, rotation_matrix, use_normals=self.use_normals).contiguous()
            
        return self.forward(input_pc, p)
    
    def forward_seg_feat(self, p, x=None):
        pass

    def point_transform(self, points: torch.Tensor):
        views = self.views[:self.num_views]
        angle = views[:, 0, :]
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        self.translation = views[:, 1, :]
        self.translation = self.translation.unsqueeze(1)

        b = points.shape[0]
        v = self.translation.shape[0]

        points = torch.repeat_interleave(points, v, dim=0)
        rot_mat = self.rot_mat.repeat(b, 1, 1)
        translation = self.translation.repeat(b, 1, 1)

        points = torch.matmul(points, rot_mat)
        points = points - translation
        return points
