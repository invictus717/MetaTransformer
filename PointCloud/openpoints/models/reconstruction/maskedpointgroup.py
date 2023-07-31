""" Maksed PointViT in PyTorch
Copyright 2022@PointNeXt team
"""
import logging
import torch
import torch.nn as nn
from openpoints.cpp.chamfer_dist import ChamferDistanceL1
from ..build import build_model_from_cfg, MODELS
from ..layers.subsample import furthest_point_sample, random_sample
from ..layers.group import KNNGroup, QueryAndGroup


@MODELS.register_module()
class MaskedPointGroup(nn.Module):
    """ Masked AutoEncoder for Point-based methods
    """

    def __init__(self,
                 encoder_args,
                 decoder_args,
                 mask_ratio,
                 subsample='fps',  # random, FPS
                 group='knn', 
                 group_size=32,
                 sample_ratio=0.25,
                 radius=0.1,
                 add_cls_token=False,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

        # ------------------------------------------------------------------
        # Grouping
        self.group_size = group_size
        self.sample_ratio = sample_ratio  # downsample 4x
        if subsample.lower() == 'fps':
            self.sample_fn = furthest_point_sample
        elif 'random' in subsample.lower():
            self.sample_fn = random_sample

        self.group = group.lower()
        if 'ball' in self.group or 'query' in self.group:
            self.grouper = QueryAndGroup(nsample=self.group_size,
                                         relative_p=False, normalize_p=False,
                                         radius=radius)
        elif 'knn' in self.group.lower():
            self.grouper = KNNGroup(self.group_size, relative_p=False)
        else:
            raise NotImplementedError(f'{self.group.lower()} is not implemented. Only support ballquery, knn')

        # ------------------------------------------------------------------
        # MAE encoder. e.g. DGCNN, DeepGCN, PointNet++
        self.encoder = build_model_from_cfg(encoder_args)
        self.add_cls_token = add_cls_token
        if self.add_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, decoder_args.embed_dim))
            torch.nn.init.normal_(self.cls_token, std=.02)
        # ------------------------------------------------------------------
        # MAE decoder. e.g. FoldingNet (works bad in random sampling), PU-GCN (NodeShuffle)
        self.use_global_feat = True if decoder_args.NAME.lower() in ['foldingnet', 'pointcompletion'] else False
        self.maxpool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        self.decoder = build_model_from_cfg(decoder_args)

        # ------------------------------------------------------------------
        # loss
        self.mask_ratio = mask_ratio
        self.build_loss_func()

    @staticmethod
    def group_random_masking(x, f=None, mask_ratio=0.9):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, 3], sequence
        f: [N, D, L], sequence
        """
        B, _, N, K = x.shape  # batch, dim, num_points, num_neighbors
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(1).unsqueeze(-1).expand(-1, 3, -1, K))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # f
        if f is not None:
            f_masked = torch.gather(f, dim=2,
                                           index=ids_keep.unsqueeze(1).unsqueeze(-1).expand(-1, f.shape[1], -1,
                                                                                            f.shape[-1]))
        else:
            f_masked = None
        return x_masked, f_masked, mask, ids_restore, ids_keep

    def build_loss_func(self, smoothing=False):
        self.criterion = ChamferDistanceL1()

    def forward_loss(self, dp, pred, mask=None):
        """
        # chamfer distance. two options
        1. chamfer distance on the merged point clouds.
        2. chamfer distance per local patch


        p: the original points [B, N, 3]
        dp: the points after grouping. [B, 3, L, K]
        pred: [B, L, K*3]
        mask: [B, L], 0 is keep, 1 is remove,
        idx_keep: [B, L]
        """
        # option 2, per patch chamfer distance.
        B, C, L, K = dp.shape
        # reshape dp and pred as [BL, K, C]
        dp = dp.permute(0, 2, 3, 1).reshape(-1, K, C)
        pred = pred.reshape(-1, K, C)
        loss = self.criterion(pred, dp)
        return loss

    def forward(self, p, f=None):
        # downsample, N -> L, e.g. 1024 -> 256
        if isinstance(p, dict):
            p, f = p['pos'], p.get('x', None)
        if f is None:
            f = p.transpose(1, 2).contiguous()
        B, N, _ = p.shape[:3]
        idx = self.sample_fn(p, int(N * self.sample_ratio)).long()
        center_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))  # e.g.  [B, L, 3]

        # query neighbors. dp: [B, 3, L, K], a typical K is 8
        dp, gf = self.grouper(center_p, p, f)
        dp_masked, gf_masked, mask, ids_restore, idx_keep = self.group_random_masking(
            dp, gf, self.mask_ratio)

        gf_masked = torch.cat((dp_masked, gf_masked), dim=1)
        latent = self.encoder.ssl_forward(dp_masked, gf_masked)  # latent: [B, C, MK]
        if self.use_global_feat:
            latent = self.maxpool(latent)
        else:
            latent = latent.transpose(1, 2)

        # concat token
        if self.add_cls_token:
            latent = torch.cat((self.cls_token.expand(B, -1, -1), latent), dim=1)
        pred = self.decoder(center_p, latent, ids_restore)  # [N, L, p*p*3]

        # pred is the reconsructed grouped p.
        """visualize pred. TODO: add to Tensorboard 
            from openpoints.dataset import vis_multi_points
            B, C, L, K = dp.shape
            input_group_pts = dp.permute(0, 2, 3, 1).reshape(B, -1, 3).detach().cpu().numpy()
            pred_group_pts = pred.clone().reshape(B, -1, 3).detach().cpu().numpy()
            # vis_multi_points((p[0].cpu().numpy(), center_p[0].detach().cpu().numpy(), dp_masked[0].detach().cpu().numpy(), pred_group_pts[0]))
            vis_multi_points((center_p[0].detach().cpu().numpy(), input_group_pts[0], dp_masked[0].detach().cpu().numpy(), pred_group_pts[0]))
        """
        loss = self.forward_loss(dp, pred, mask)
        return loss, pred
