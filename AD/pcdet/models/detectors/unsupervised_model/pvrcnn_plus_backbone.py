import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..detector3d_template import Detector3DTemplate
from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils

from ....utils import common_utils

# copy from voxel set abstraction module
def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans

def sample_points_with_roi(rois, points, sample_radius_with_roi, num_max_points_of_part=200000):
    """
    Args:
        rois: (M, 7 + C)
        points: (N, 3)
        sample_radius_with_roi:
        num_max_points_of_part:

    Returns:
        sampled_points: (N_out, 3)
    """
    if points.shape[0] < num_max_points_of_part:
        distance = (points[:, None, :] - rois[None, :, 0:3]).norm(dim=-1)
        min_dis, min_dis_roi_idx = distance.min(dim=-1)
        roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
        point_mask = min_dis < roi_max_dim + sample_radius_with_roi
    else:
        start_idx = 0
        point_mask_list = []
        while start_idx < points.shape[0]:
            distance = (points[start_idx:start_idx + num_max_points_of_part, None, :] - rois[None, :, 0:3]).norm(dim=-1)
            min_dis, min_dis_roi_idx = distance.min(dim=-1)
            roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
            cur_point_mask = min_dis < roi_max_dim + sample_radius_with_roi
            point_mask_list.append(cur_point_mask)
            start_idx += num_max_points_of_part
        point_mask = torch.cat(point_mask_list, dim=0)

    sampled_points = points[:1] if point_mask.sum() == 0 else points[point_mask, :]

    return sampled_points, point_mask


# TODO add LOSS_CFG TO MODEL_CFG
class PVRCNN_PLUS_BACKBONE(Detector3DTemplate):
    def __init__(self, model_cfg, dataset, num_class=None):
        super().__init__(model_cfg=model_cfg, num_class=None, dataset=dataset)
        self.module_list = self.build_networks()
    
    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        # batch_dict = self.backbone_2d(batch_dict)
        return batch_dict


# TODO add POS_THRESH, NEG_THRESH
class HardestContrastiveLoss():
    def __init__(self, loss_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.point_feature_names = []
        self.pos_thresh = loss_cfg.POS_THRESH
        self.neg_thresh = loss_cfg.NEG_THRESH
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        SA_cfg = self.loss_cfg.SA_LAYER
        self.point_feature_names = []
        self.downsample_times_map = {}
        self.SA_layers = nn.ModuleList()
   
        for src_name in self.loss_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            self.point_feature_names.append(src_name)
            self.SA_layers.append(build_feature_aggregation_module(config=SA_cfg[src_name]))

    def pdist(self, point_features_1, point_features_2):
        D = torch.sum((point_features_1.unsqueeze(1) - point_features_2.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D + 1e-7)

    # TODO add self.pos_thresh and self.neg_thresh
    def get_hardest_contrastive_loss(self, batch_dict_1, batch_dict_2):
        batch_size = batch_dict_1['batch_size']
        batch_dict_1, batch_dict_2, keypoints_inds = self.get_point_features(batch_dict_1, batch_dict_2, tag='positive')
        batch_dict_1, batch_dict_2, (keypoints_inds_1, keypoints_inds_2) = self.get_point_features(batch_dict_1, batch_dict_2, tag='negative')
        pos_features_1, pos_features_2 = batch_dict_1['point_features_positive'], batch_dict_2['point_features_positive']
        neg_features_1, neg_features_2 = batch_dict_1['point_features_negative'], batch_dict_2['point_features_negative']
        
        batch_size = batch_dict_1['batch_size']
        pos_loss_all = None
        neg_loss_all = None
        for bs_idx in range(batch_size):
            mask_pos = batch_dict_1['points_coords_positive'][:, 0] == bs_idx
            cur_pos_features_1, cur_pos_features_2 = pos_features_1[mask_pos], pos_features_2[mask_pos]
            pos_loss = torch.relu((cur_pos_features_1 - cur_pos_features_2).pow(2).sum(1) - self.pos_thresh)
            mask_neg = batch_dict_1['points_coords_negative'][:, 0] == bs_idx
            cur_neg_features_1, cur_neg_features_2 = neg_features_1[mask_neg], neg_features_2[mask_neg]
            distance_1 = self.pdist(cur_pos_features_1, cur_neg_features_2)
            distance_2 = self.pdist(cur_pos_features_2, cur_neg_features_1)

            distance_1_min, distance_1_ind = distance_1.min(1)
            distance_2_min, distance_2_ind = distance_2.min(1)
            mask_1 = keypoints_inds[mask_pos] != keypoints_inds_2[distance_1_ind].to(keypoints_inds.device)
            mask_2 = keypoints_inds[mask_pos] != keypoints_inds_1[distance_2_ind].to(keypoints_inds.device)
            neg_loss_1 = torch.relu(self.neg_thresh - distance_1_min[mask_1]).pow(2)
            neg_loss_2 = torch.relu(self.neg_thresh - distance_2_min[mask_2]).pow(2)
            
            pos_loss = pos_loss.mean()
            neg_loss = (neg_loss_1.mean() + neg_loss_2.mean()) / 2
            if pos_loss_all is None and neg_loss_all is None:
                pos_loss_all = pos_loss
                neg_loss_all = neg_loss
            else:
                pos_loss_all += pos_loss
                neg_loss_all += neg_loss

        pos_loss_all = pos_loss_all / batch_size
        neg_loss_all = neg_loss_all / batch_size       
        return pos_loss_all, neg_loss_all


    def get_point_features(self, batch_dict_1, batch_dict_2, tag='positive'):
        if tag == 'positive':
            keypoints_1, keypoints_2, keypoints_inds = self.get_positive_sampled_points(batch_dict_1, batch_dict_2)
        else:
            keypoints_1, keypoints_2, keypoints_inds_1, keypoints_inds_2 = self.get_negative_sampled_points(batch_dict_1, batch_dict_2, method='random')
            keypoints_inds = (keypoints_inds_1, keypoints_inds_2)

        point_feature_list_1 = []
        point_feature_list_2 = []
        if 'bev' in self.loss_cfg.FEATURES_SOURCE:
            point_bev_features_1 = self.interpolate_from_bev_features(
                keypoints_1, batch_dict_1['spatial_features'], batch_dict_1['batch_size'],
                bev_stride=batch_dict_1['spatial_features_stride']
            )
            point_bev_features_2 = self.interpolate_from_bev_features(
                keypoints_2, batch_dict_2['spatial_features'], batch_dict_2['batch_size'],
                bev_stride=batch_dict_2['spatial_features_stride']
            )
            point_feature_list_1.append(point_bev_features_1)
            point_feature_list_2.append(point_bev_features_2)

        batch_size = batch_dict_1['batch_size']
        new_xyz_1 = keypoints_1[:, 1:4].contiguous()
        new_xyz_2 = keypoints_2[:, 1:4].contiguous()
        new_xyz_batch_cnt_1 = new_xyz_1.new_zeros(batch_size).int()
        new_xyz_batch_cnt_2 = new_xyz_2.new_zeros(batch_size).int()
        for k in range(batch_size):
            new_xyz_batch_cnt_1[k] = (keypoints_1[:, 0] == k).sum()
        for k in range(batch_size):
            new_xyz_batch_cnt_2[k] = (keypoints_2[:, 0] == k).sum()
        
        for k, src_name in enumerate(self.point_feature_names):
            cur_coords_1 = batch_dict_1['multi_scale_3d_features'][src_name].indices
            cur_features_1 = batch_dict_1['multi_scale_3d_features'][src_name].features.contiguous()

            # TODO: add self.down_sample_times_map, self.voxel_size, self.point_cloud_range to __init__
            xyz_1 = common_utils.get_voxel_centers(
                cur_coords_1[:, 1:4], downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range
            )

            cur_coords_2 = batch_dict_2['multi_scale_3d_features'][src_name].indices
            cur_features_2 = batch_dict_2['multi_scale_3d_features'][src_name].features.contiguous()
            xyz_2 = common_utils.get_voxel_centers(
                cur_coords_2[:, 1:4], downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range
            )

            pooled_features_1 = self.aggregate_keypoint_features_from_one_source(
                batch_size=batch_size, aggregate_func=self.SA_layers[k],
                xyz=xyz_1.contiguous(), xyz_features=cur_features_1, xyz_bs_idxs=cur_coords_1[:, 0],
                new_xyz=new_xyz_1, new_xyz_batch_cnt=new_xyz_batch_cnt_1,
                filter_neighbors_with_roi=self.loss_cfg.SA_LAYER[src_name].get('FILTER_NEIGHBOR_WITH_ROI', False),
                radius_of_neighbor=self.loss_cfg.SA_LAYER[src_name].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
                rois=batch_dict_1.get('rois', None),
                cover_feat_4=self.model_cfg.COVER_FEAT if self.loss_cfg.get('COVER_FEAT', None) else None
            )
            pooled_features_2 = self.aggregate_keypoint_features_from_one_source(
                batch_size=batch_size, aggregate_func=self.SA_layers[k],
                xyz=xyz_2.contiguous(), xyz_features=cur_features_2, xyz_bs_idxs=cur_coords_2[:, 0],
                new_xyz=new_xyz_2, new_xyz_batch_cnt=new_xyz_batch_cnt_2,
                filter_neighbors_with_roi=self.loss_cfg.SA_LAYER[src_name].get('FILTER_NEIGHBOR_WITH_ROI', False),
                radius_of_neighbor=self.loss_cfg.SA_LAYER[src_name].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
                rois=batch_dict_2.get('rois', None),
                cover_feat_4=self.model_cfg.COVER_FEAT if self.loss_cfg.get('COVER_FEAT', None) else None
            )

            point_feature_list_1.append(pooled_features_1)
            point_feature_list_2.append(pooled_features_2)
        
        point_features_1 = torch.cat(point_feature_list_1, dim=-1)
        point_features_2 = torch.cat(point_feature_list_2, dim=-1)

        save_name = 'point_features_' + tag
        save_name_coords = 'points_coords_' + tag
        batch_dict_1[save_name] = point_features_1.view(-1, point_features_1.shape[-1])
        batch_dict_2[save_name] = point_features_2.view(-1, point_features_2.shape[-1])
        batch_dict_1[save_name_coords] = keypoints_1
        batch_dict_2[save_name_coords] = keypoints_2

        return batch_dict_1, batch_dict_2, keypoints_inds

    @staticmethod
    def aggregate_keypoint_features_from_one_source(
            batch_size, aggregate_func, xyz, xyz_features, xyz_bs_idxs, new_xyz, new_xyz_batch_cnt,
            filter_neighbors_with_roi=False, radius_of_neighbor=None, num_max_points_of_part=200000, rois=None, cover_feat_4=False
    ):
        """

        Args:
            aggregate_func:
            xyz: (N, 3)
            xyz_features: (N, C)
            xyz_bs_idxs: (N)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size), [N1, N2, ...]

            filter_neighbors_with_roi: True/False
            radius_of_neighbor: float
            num_max_points_of_part: int
            rois: (batch_size, num_rois, 7 + C)
            cover_feat_4: if cover the xyz_features using the values in z-dimension
        Returns:

        """
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        if filter_neighbors_with_roi:
            point_features = torch.cat((xyz, xyz_features), dim=-1) if xyz_features is not None else xyz
            point_features_list = []
            for bs_idx in range(batch_size):
                bs_mask = (xyz_bs_idxs == bs_idx)
                _, valid_mask = sample_points_with_roi(
                    rois=rois[bs_idx], points=xyz[bs_mask],
                    sample_radius_with_roi=radius_of_neighbor, num_max_points_of_part=num_max_points_of_part,
                )
                point_features_list.append(point_features[bs_mask][valid_mask])
                xyz_batch_cnt[bs_idx] = valid_mask.sum()

            valid_point_features = torch.cat(point_features_list, dim=0)
            xyz = valid_point_features[:, 0:3]
            xyz_features = valid_point_features[:, 3:] if xyz_features is not None else None
        else:
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (xyz_bs_idxs == bs_idx).sum()

        #modify: for z-axes as the fourth dimension feature of point-cloud representations
        if xyz_features is None:
            if cover_feat_4:
                xyz_features=xyz[:, 2].view(-1, 1)
                pooled_points, pooled_features = aggregate_func(
                    xyz=xyz.contiguous(),
                    xyz_batch_cnt=xyz_batch_cnt,
                    new_xyz=new_xyz,
                    new_xyz_batch_cnt=new_xyz_batch_cnt,
                    features=xyz_features.contiguous(),
                )
            else:
                pooled_points, pooled_features = aggregate_func(
                    xyz=xyz.contiguous(),
                    xyz_batch_cnt=xyz_batch_cnt,
                    new_xyz=new_xyz,
                    new_xyz_batch_cnt=new_xyz_batch_cnt,
                    features=xyz_features,
                )
        else:
            pooled_points, pooled_features = aggregate_func(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=xyz_features.contiguous(),
            )
        return pooled_features

    # copy from voxel set abstraction module
    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        """
        Args:
            keypoints: (N1 + N2 + ..., 4)
            bev_features: (B, C, H, W)
            batch_size:
            bev_stride:

        Returns:
            point_bev_features: (N1 + N2 + ..., C)
        """
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]

        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            bs_mask = (keypoints[:, 0] == k)

            cur_x_idxs = x_idxs[bs_mask]
            cur_y_idxs = y_idxs[bs_mask]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (N1 + N2 + ..., C)
        return point_bev_features

    # copy from voxel set abstraction module
    def get_positive_sampled_points(self, batch_dict_1, batch_dict_2):
        """
        Args:
            batch_dict:

        Returns:
            keypoints: (N1 + N2 + ..., 4), where 4 indicates [bs_idx, x, y, z]
        """
        batch_size = batch_dict_1['batch_size']
        if self.loss_cfg.POINT_SOURCE == 'raw_points':
            src_points_1 = batch_dict_1['points'][:, 1:4]
            src_points_2 = batch_dict_2['points'][:, 1:4]
            batch_indices = batch_dict_1['points'][:, 0].long()
        elif self.loss_cfg.POINT_SOURCE == 'voxel_centers':
            src_points_1 = common_utils.get_voxel_centers(
                batch_dict_1['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            src_points_2 = common_utils.get_voxel_centers(
                batch_dict_2['voxel_coords'],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict_1['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list_1 = []
        keypoints_list_2 = []
        keypoints_inds_list = []

        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points_1 = src_points_1[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            sampled_points_2 = src_points_2[bs_mask].unsqueeze(dim=0)

            # using FPS to sample points
            cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(
                sampled_points_1[:, :, 0:3].contiguous(), self.loss_cfg.NUM_KEYPOINTS
            ).long()

            if sampled_points_1.shape[1] < self.loss_cfg.NUM_KEYPOINTS:
                times = int(self.loss_cfg.NUM_KEYPOINTS / sampled_points_1.shape[1]) + 1
                non_empty = cur_pt_idxs[0, :sampled_points_1.shape[1]]
                cur_pt_idxs[0] = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS]

            keypoints_1 = sampled_points_1[0][cur_pt_idxs[0]].unsqueeze(dim=0)
            keypoints_2 = sampled_points_2[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            keypoints_list_1.append(keypoints_1)
            keypoints_list_2.append(keypoints_2)
            keypoints_inds_list.append(cur_pt_idxs[0])

        keypoints_1 = torch.cat(keypoints_list_1, dim=0)  # (B, M, 3) or (N1 + N2 + ..., 4)
        keypoints_2 = torch.cat(keypoints_list_2, dim=0)
        keypoints_inds = torch.cat(keypoints_inds_list, dim=0)
        if len(keypoints_1.shape) == 3:
            batch_idx = torch.arange(batch_size, device=keypoints_1.device).view(-1, 1).repeat(1, keypoints_1.shape[1]).view(-1, 1)
            keypoints_1 = torch.cat((batch_idx.float(), keypoints_1.view(-1, 3)), dim=1)
            keypoints_2 = torch.cat((batch_idx.float(), keypoints_2.view(-1, 3)), dim=1)

        return keypoints_1, keypoints_2, keypoints_inds

    # TODO add NUM_NEGATIVE_KEYPOINTS to config
    def get_negative_sampled_points(self, batch_dict_1, batch_dict_2, method='random'):
        """
        Args:
            batch_dict:

        Returns:
            keypoints: (N1 + N2 + ..., 4), where 4 indicates [bs_idx, x, y, z]
        """
        batch_size = batch_dict_1['batch_size']
        if self.loss_cfg.POINT_SOURCE == 'raw_points':
            src_points_1 = batch_dict_1['points'][:, 1:4]
            src_points_2 = batch_dict_2['points'][:, 1:4]
            batch_indices = batch_dict_1['points'][:, 0].long()
        elif self.loss_cfg.POINT_SOURCE == 'voxel_centers':
            src_points_1 = common_utils.get_voxel_centers(
                batch_dict_1['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            src_points_2 = common_utils.get_voxel_centers(
                batch_dict_2['voxel_coords'],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict_1['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list_1 = []
        keypoints_list_2 = []

        keypoints_inds_list_1 = []
        keypoints_inds_list_2 = []

        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points_1 = src_points_1[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            sampled_points_2 = src_points_2[bs_mask].unsqueeze(dim=0)

            # using FPS to sample points
            if method == 'fps':
                cur_pt_idxs_1 = pointnet2_stack_utils.farthest_point_sample(
                    sampled_points_1[:, :, 0:3].contiguous(), self.loss_cfg.NUM_NEGATIVE_KEYPOINTS
                ).long()
                
                cur_pt_idxs_2 = pointnet2_stack_utils.farthest_point_sample(
                    sampled_points_2[:, :, 0:3].contiguous(), self.loss_cfg.NUM_NEGATIVE_KEYPOINTS
                ).long()
            elif method == 'random':
                num_points = sampled_points_1.shape[1]
                cur_pt_idxs_1 = torch.from_numpy(np.random.choice(num_points, self.loss_cfg.NUM_NEGATIVE_KEYPOINTS, replace=False)).long()
                cur_pt_idxs_2 = torch.from_numpy(np.random.choice(num_points, self.loss_cfg.NUM_NEGATIVE_KEYPOINTS, replace=False)).long()
                cur_pt_idxs_1 = cur_pt_idxs_1.view(1, -1)
                cur_pt_idxs_2 = cur_pt_idxs_2.view(1, -1)

            if sampled_points_1.shape[1] < self.loss_cfg.NUM_NEGATIVE_KEYPOINTS:
                times = int(self.loss_cfg.NUM_NEGATIVE_KEYPOINTS / sampled_points_1.shape[1]) + 1
                non_empty = cur_pt_idxs_1[0, :sampled_points_1.shape[1]]
                cur_pt_idxs_1[0] = non_empty.repeat(times)[:self.model_cfg.NUM_NEGATIVE_KEYPOINTS]
            
            if sampled_points_2.shape[1] < self.loss_cfg.NUM_NEGATIVE_KEYPOINTS:
                times = int(self.loss_cfg.NUM_NEGATIVE_KEYPOINTS / sampled_points_2.shape[1]) + 1
                non_empty = cur_pt_idxs_2[0, :sampled_points_2.shape[1]]
                cur_pt_idxs_2[0] = non_empty.repeat(times)[:self.model_cfg.NUM_NEGATIVE_KEYPOINTS]

            keypoints_1 = sampled_points_1[0][cur_pt_idxs_1[0]].unsqueeze(dim=0)
            keypoints_2 = sampled_points_2[0][cur_pt_idxs_2[0]].unsqueeze(dim=0)
            
            keypoints_inds_list_1.append(cur_pt_idxs_1[0])
            keypoints_inds_list_2.append(cur_pt_idxs_2[0])
            keypoints_list_1.append(keypoints_1)
            keypoints_list_2.append(keypoints_2)

        keypoints_1 = torch.cat(keypoints_list_1, dim=0)  # (B, M, 3) or (N1 + N2 + ..., 4)
        keypoints_2 = torch.cat(keypoints_list_2, dim=0)
        keypoints_inds_1 = torch.cat(keypoints_inds_list_1, dim=0)
        keypoints_inds_2 = torch.cat(keypoints_inds_list_2, dim=0)
        if len(keypoints_1.shape) == 3:
            batch_idx = torch.arange(batch_size, device=keypoints_1.device).view(-1, 1).repeat(1, keypoints_1.shape[1]).view(-1, 1)
            keypoints_1 = torch.cat((batch_idx.float(), keypoints_1.view(-1, 3)), dim=1)
            keypoints_2 = torch.cat((batch_idx.float(), keypoints_2.view(-1, 3)), dim=1)

        return keypoints_1, keypoints_2, keypoints_inds_1, keypoints_inds_2
        

def build_feature_aggregation_module(config):
    local_aggregation_name = config.get('NAME', 'StackSAModuleMSG')

    if local_aggregation_name == 'StackSAModuleMSG':
        cur_layer = StackPointFeature(
            radii=config.POOL_RADIUS, nsamples=config.NSAMPLE, pool_method='max_pool',
        )
    else:
        raise NotImplementedError

    return cur_layer


class StackPointFeature(nn.Module):

    def __init__(self, *, radii, nsamples, pool_method='max_pool'):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples)

        self.groupers = nn.ModuleList()

        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_stack_utils.QueryAndGroup(radius, nsample, use_xyz=False))
        self.pool_method = pool_method

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
            )  # (M1 + M2, C, nsample)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)

            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features