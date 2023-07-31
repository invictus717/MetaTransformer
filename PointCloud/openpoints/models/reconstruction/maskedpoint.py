""" Maksed PointViT in PyTorch
Copyright 2022@PointNeXt team
"""
import logging
import torch
import torch.nn as nn
from openpoints.cpp.chamfer_dist import ChamferDistanceL1
from ..build import build_model_from_cfg, MODELS


@MODELS.register_module()
class MaskedPoint(nn.Module):
    """ Masked AutoEncoder for Point-based methods
    """
    def __init__(self,
                 backbone_args,
                 decoder_args,
                 mask_ratio,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")

        # ------------------------------------------------------------------
        # MAE encoder. e.g. DGCNN, DeepGCN, PointNet++
        self.encoder = build_model_from_cfg(backbone_args)

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
    def random_masking(x, features=None, mask_ratio=0.9):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, 3], sequence
        features: [N, D, L], sequence
        TODO: suppport other masking. Like OcCo, block masking as ablation
        """
        N, L, _ = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, 3))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # features
        if features is not None:
            features_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(1).expand(-1, features.shape[1], -1))
        else:
            features_masked = None
        return x_masked, features_masked, mask, ids_restore, ids_keep

    def build_loss_func(self, smoothing=False):
        self.criterion = ChamferDistanceL1()

    def forward_loss(self, xyz, pred, mask=None):
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
        if isinstance(pred, (tuple, list)):
            loss = 0.
            for pred_i in pred:
                loss += self.criterion(pred_i, xyz)
        else:
            loss = self.criterion(pred, xyz)
        # mask = mask.unsqueeze(-1)
        # grouped_xyz = torch.mul(grouped_xyz.permute(0, 2, 3, 1), mask.unsqueeze(-1)).reshape(-1, K, C)    
        # pred = torch.mul(pred, mask).reshape(-1, K, C)  
        # loss = self.criterion(pred, grouped_xyz)/ mask.sum()
        return loss

    def forward(self, xyz, features=None):
        xyz_masked, features_masked, mask, ids_restore, idx_keep = self.random_masking(xyz, features, self.mask_ratio)
        latent = self.encoder(xyz_masked, features_masked)
        if self.use_global_feat:
            latent = self.maxpool(latent)
        pred = self.decoder(xyz_masked, latent, ids_restore)  # [N, L, p*p*3]

        """visualize pred. TODO: add to Tensorboard 
            from openpoints.dataset import vis_multi_points
            vis_multi_points((xyz[0].cpu().detach().numpy(), xyz_masked[0].cpu().detach().numpy(), pred[-1][0].cpu().detach().numpy()))
        """
        loss = self.forward_loss(xyz, pred, mask)
        return loss, pred

