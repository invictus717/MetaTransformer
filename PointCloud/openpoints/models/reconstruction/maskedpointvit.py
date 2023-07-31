""" Maksed PointViT in PyTorch
Copyright 2022@PointNeXt team
"""
import logging
import torch
import torch.nn as nn
from openpoints.models.layers import create_norm
from openpoints.models.layers.attention import Block
from openpoints.models.build import MODELS
from openpoints.cpp.chamfer_dist import ChamferDistanceL1
from ..backbone import PointViT


@MODELS.register_module()
class MaskedPointViT(nn.Module):
    """ Vision Transformer for 3D
    """

    def __init__(self,
                 in_channels=6, num_classes=40,
                 embed_dim=384,
                 depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=False,
                 decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=16,
                 embed_args={'NAME': 'PointPatchEmbed', 
                             'sample_ratio': 0.0625,
                             'group_size': 32,
                             'subsample': 'fps', 
                             'group': 'knn', 
                             'feature_type': 'dp'},
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 act_args={'act': 'gelu'},
                 posembed_norm_args=None, add_pos_each_block=True,
                 mask_ratio=0.75,
                 **kwargs
                 ):
        """
        Args:
            num_group (int, tuple): number of patches (groups in 3d)
            group_size (int, tuple): the size (# points) of each group
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.mask_ratio = mask_ratio
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.add_pos_each_block = add_pos_each_block
        
        # MAE encoder
        # ------------------------------------------------------------------
        self.encoder = PointViT(
            in_channels, num_classes,
            embed_dim, depth,
            num_heads, mlp_ratio, qkv_bias,
            embed_args=embed_args, norm_args=norm_args, act_args=act_args,
            posembed_norm_args=posembed_norm_args,
            add_pos_each_block=add_pos_each_block)

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
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                norm_args=norm_args, act_args=act_args
            )
            for _ in range(decoder_depth)])
        self.decoder_norm = create_norm(norm_args, decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_args.group_size * 3, bias=True)  # decoder to patch
        self.initialize_weights()
        self.build_loss_func()

    def initialize_weights(self):
        # initialization
        torch.nn.init.normal_(self.encoder.cls_token, std=.02)
        torch.nn.init.normal_(self.encoder.cls_pos, std=.02)
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

    @staticmethod
    def random_masking(x, pos_embed, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        pos_embed_masked = torch.gather(pos_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, pos_embed_masked, mask, ids_restore, ids_keep

    def build_loss_func(self, smoothing=False):
        self.criterion = ChamferDistanceL1()

    # Facebook.
    def forward_encoder(self, center_xyz, features, mask_ratio):
        pos_embed = self.encoder.pos_embed(center_xyz)
        # masking: length -> length * mask_ratio. Here, pos_embed has been shuffled!!!
        features, pos_embed, mask, ids_restore, idx_keep = self.random_masking(features, pos_embed, mask_ratio)

        """DEBUG. visualization code for masking. 
            from openpoints.dataset import vis_multi_points
            xyz_keep = torch.gather(center_xyz, 1, idx_keep.unsqueeze(-1).expand(-1, -1, 3))
            vis_multi_points((center_xyz[0].cpu().numpy(), xyz_keep[0].cpu().numpy()))
        """
        
        # append cls token after masking
        pos_embed = torch.cat((self.encoder.cls_pos.expand(features.shape[0], -1, -1), pos_embed), dim=1)
        cls_token = self.encoder.cls_token.expand(features.shape[0], -1, -1)
        features = torch.cat((cls_token, features), dim=1)

        if self.encoder.add_pos_each_block:
            for block in self.encoder.blocks:
                features = block(features + pos_embed)
        else:
            features = self.pos_drop(features + pos_embed)
            for block in self.encoder.blocks:
                features = block(features)

        features = self.encoder.norm(features)
        return features, mask, ids_restore, idx_keep

    def forward_decoder(self, features, center_xyz, ids_restore):
        # embed tokens
        features = self.decoder_embed(features)
        B, L, C = features.shape  # batch size, length, channels

        # in the decoder part. we know the positional encoding of groups
        decoder_pos_embed = torch.cat(
            (self.decoder_cls_pos.expand(B, -1, -1), self.decoder_pos_embed(center_xyz)), dim=1)

        # append mask tokens to sequence
        # use mask tokens to fill the masked features. this is why missing part should be 1 not 0.
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] + 1 - features.shape[1], 1)
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

    def forward_loss(self, xyz, grouped_xyz, pred, mask, idx_keep):
        """
        # chamfer distance. two options 
        1. chamfer distance on the merged point clouds. 
        2. chamfer distance per local patch
        
        
        xyz: the original points [B, N, 3] 
        grouped_xyz: the points after grouping. [B, 3, L, K]
        pred: [B, L, K*3]
        mask: [B, L], 0 is keep, 1 is remove,
        idx_keep: [B, L]
        """
        # option 2, per patch chamfer distance. 
        B, C, L, K = grouped_xyz.shape
        # consider (B, L) as batchs, chamfer distance per patch
        
        # idx_keep = idx_keep.unsqueeze(-1).expand(-1, -1, K)
        # pred = torch.gather(pred, 1, idx_keep)  # B, 
        # grouped_xyz = torch.gather(grouped_xyz, 2, idx_keep.unsqueeze(1).expand(-1, C, -1, -1))

        grouped_xyz = grouped_xyz.permute(0, 2, 3, 1).reshape(-1, K, C)    
        pred = pred.reshape(-1, K, C)  
        loss = self.criterion(pred, grouped_xyz)
        
        # mask = mask.unsqueeze(-1)
        # grouped_xyz = torch.mul(grouped_xyz.permute(0, 2, 3, 1), mask.unsqueeze(-1)).reshape(-1, K, C)    
        # pred = torch.mul(pred, mask).reshape(-1, K, C)  
        # loss = self.criterion(pred, grouped_xyz)/ mask.sum()
        return loss

    def forward(self, xyz, features=None):
        center_xyz, features, grouped_xyz, grouped_features = self.encoder.patch_embed(xyz, features)
        latent, mask, ids_restore, idx_keep = self.forward_encoder(center_xyz, features, self.mask_ratio)
        pred = self.forward_decoder(latent, center_xyz, ids_restore)  # [N, L, p*p*3]

        """visualize pred. TODO: add to Tensorboard 
            from openpoints.dataset import vis_multi_points
            B, C, L, K = grouped_xyz.shape
            input_group_pts = grouped_xyz.clone().permute(0, 2, 3, 1).reshape(B, -1, C).detach().cpu().numpy()
            # input_group_pts = grouped_xyz.clone().permute(0, 2, 3, 1).reshape(B, -1, C).detach().cpu().numpy()
            pred_group_pts = pred.clone().reshape(B, -1, C).detach().cpu().numpy()
            vis_multi_points((xyz[0].cpu().numpy(), center_xyz[0].detach().cpu().numpy(), input_group_pts[0], pred_group_pts[0]))
        """
        loss = self.forward_loss(xyz, grouped_xyz, pred, mask, idx_keep)
        return loss, pred

