"""
This file is borrowed from 3DAL-Project: https://gitlab.com/pjlab-adg/3dal-toolchain-v2
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.ops.iou3d_nms import iou3d_nms_utils

from ...utils import loss_utils
from ..model_utils import centernet_utils


class CenterPointSingle(nn.Module):
    """
    Center point based segmentation head.
    Reference Paper: https://arxiv.org/pdf/2006.11275.pdf
    Center-based 3D Object Detection and Tracking
    """
    def __init__(self, num_class, class_names, input_channels, model_cfg,
                 grid_size, voxel_size, point_cloud_range,
                 tta=False, predict_boxes_when_training=True):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.tta = tta
        self.predict_boxes_when_training = predict_boxes_when_training
        self.tasks = {"hm": num_class, "ofs": 2, "height": 1, "dim": 3, "rot": 2}
        self.pred_velo = model_cfg.get("PRED_VELOCITY", False)
        if self.pred_velo:
            self.tasks.update({"velo": 2})

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

        self.shared_layers = nn.Sequential(
            nn.Conv2d(input_channels, model_cfg.SHARED_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(model_cfg.SHARED_CHANNELS, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.task_layers = nn.ModuleDict()
        for head, channel in self.tasks.items():
            current_layer = self.make_conv_layers(
                conv_cfg=model_cfg.TASK_CONV,
                input_channels=model_cfg.SHARED_CHANNELS,
                output_channels=channel
            )
            if head == 'hm':
                current_layer[-1].bias.data.fill_(-2.19)
            else:
                for m in current_layer.modules():
                    if isinstance(m, nn.Conv2d):
                        self.kaiming_init(m)
            self.task_layers[head] = current_layer

        self.min_overlap = model_cfg.MIN_OVERLAP
        self.min_radius = model_cfg.MIN_RADIUS
        self.stride = model_cfg.STRIDE

    @staticmethod
    def make_conv_layers(conv_cfg, input_channels, output_channels):
        conv_layers = []
        c_in = input_channels
        for k in range(0, conv_cfg.__len__()):
            conv_layers.extend([
                nn.Conv2d(c_in, conv_cfg[k], kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(conv_cfg[k], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ])
            c_in = conv_cfg[k]
        conv_layers.append(nn.Conv2d(c_in, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
        return nn.Sequential(*conv_layers)

    @staticmethod
    def _sigmoid(x):
        return torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)

    @staticmethod
    def kaiming_init(module, a=0, mode="fan_out", nonlinearity="relu",
                     bias=0, distribution="normal"):
        assert distribution in ["uniform", "normal"]
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def test_time_augment(data_dict, cls_preds, reg_preds):
        tta_ops = data_dict["tta_ops"]
        tta_num = len(tta_ops)
        bs = int(data_dict["batch_size"] // tta_num)
        _, H, W, C = cls_preds.shape

        # scatter the original and augmented preds
        cls_preds = cls_preds.reshape(bs, tta_num, H, W, -1)
        reg_preds = reg_preds.reshape(bs, tta_num, H, W, -1)
        
        # restore the augmented preds to original coordinates
        for i, tta_cfg in enumerate(tta_ops):
            name, param = tta_cfg.split("_")[1], tta_cfg.split("_")[2]
            if name == "original":
                continue
            elif name == "flip":
                if param == 'x':
                    cls_preds[:, i] = cls_preds[:, i].flip(dims=[1])
                    reg_preds[:, i] = reg_preds[:, i].flip(dims=[1])
                    # process flipped local offset
                    reg_preds[:, i, ..., 1] = 1 - reg_preds[:, i, ..., 1]
                    # process flipped heading
                    reg_preds[:, i, ..., 3] *= -1
                    
                elif param == 'y':
                    cls_preds[:, i] = cls_preds[:, i].flip(dims=[2])
                    reg_preds[:, i] = reg_preds[:, i].flip(dims=[2])
                    # process flipped local offset
                    reg_preds[:, i, ..., 0] = 1 - reg_preds[:, i, ..., 0]
                    # process flipped heading
                    reg_preds[:, i, ..., 2] *= -1
                elif param == 'xy':
                    cls_preds[:, i] = cls_preds[:, i].flip(dims=[1, 2])
                    reg_preds[:, i] = reg_preds[:, i].flip(dims=[1, 2])
                    # process flipped local offset
                    reg_preds[:, i, ..., 0:2] = 1 - reg_preds[:, i, ..., 0:2]
                    # process flipped heading
                    reg_preds[:, i, ..., 2:4] *= -1
            elif name == "rot":
                raise NotImplementedError
            elif name == "scale":
                raise NotImplementedError

        # calculate the mean of all augmented preds
        cls_preds = cls_preds.mean(dim=1)
        reg_preds = reg_preds.mean(dim=1)
        data_dict["batch_size"] = bs

        return cls_preds, reg_preds

    def assign_targets(self, gt_boxes):
        # obtain the basic params
        dtype = gt_boxes.dtype
        device = gt_boxes.device
        bs, max_num, _ = gt_boxes.shape
        gt_class = gt_boxes[:, :, -1]
        gt_boxes = gt_boxes[:, :, :-1]

        # calculte the coords on the final feat-map mapped from world
        x_lower, y_lower, z_lower = self.point_cloud_range[:3]
        x_num, y_num = int(self.grid_size[0] / self.stride), int(self.grid_size[1] / self.stride)
        x_size, y_size = self.voxel_size[0] * self.stride, self.voxel_size[1] * self.stride
        x_coord = (gt_boxes[:, :, 0] - x_lower) / x_size
        y_coord = (gt_boxes[:, :, 1] - y_lower) / y_size

        # init the targets
        hm_gt = np.zeros((bs, y_num, x_num, self.num_class), dtype=np.float32)
        cls_gt = torch.zeros((bs, max_num, 3), dtype=torch.int64, device=device)
        ofs_gt = torch.zeros((bs, max_num, self.tasks['ofs']), dtype=dtype, device=device)
        rot_gt = torch.zeros((bs, max_num, self.tasks['rot']), dtype=dtype, device=device)

        # centerpoint will throw out the objects outside of this range, rather than clamping
        cls_gt[:, :, 0] = torch.clamp(torch.floor(x_coord), 0, x_num-1)
        cls_gt[:, :, 1] = torch.clamp(torch.floor(y_coord), 0, y_num-1)
        cls_gt[:, :, 2] = gt_class

        ofs_gt[:, :, 0] = x_coord - cls_gt[:, :, 0]
        ofs_gt[:, :, 1] = y_coord - cls_gt[:, :, 1]

        height_gt = gt_boxes[:, :, 2].unsqueeze(2)
        dim_gt = torch.log(gt_boxes[:, :, 3:6])
        rot_gt[:, :, 0] = torch.cos(gt_boxes[:, :, 6])
        rot_gt[:, :, 1] = torch.sin(gt_boxes[:, :, 6])

        # draw heatmap target
        for k in range(bs):
            cur_gt = gt_boxes[k]
            cnt = len(cur_gt) - 1
            while cnt >= 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            for i in range(cnt+1):
                dx = gt_boxes[k, i, 3].item() / x_size
                dy = gt_boxes[k, i, 4].item() / y_size
                radius = centernet_utils.gaussian_radius_3dal((dx,dy), min_overlap=self.min_overlap)
                # the input and output in 3DAL is numpy, but in 3DTrans is torch.tensor
                radius = int(max(radius, self.min_radius))
                ct = np.array([cls_gt[k, i, 0].item(), cls_gt[k, i, 1].item()])
                centernet_utils.draw_gaussian(
                    hm_gt[k, :, :, cls_gt[k, i, 2].item()-1],
                    ct,
                    radius
                )
        hm_gt = torch.from_numpy(hm_gt).to(dtype=dtype, device=device)
        
        if self.pred_velo:
            velo_gt = gt_boxes[:, :, 7:9]
            reg_gt = torch.cat([ofs_gt, height_gt, dim_gt, rot_gt, velo_gt], dim=2)
        else:
            reg_gt = torch.cat([ofs_gt, height_gt, dim_gt, rot_gt], dim=2)
        
        all_targets_dict = {
            'hm_targets': hm_gt,
            'cls_targets': cls_gt,
            'reg_targets': reg_gt,
        }
        return all_targets_dict

    def assign_iou_target(self, indices):
        cls_preds = self.forward_ret_dict['cls_preds']
        reg_preds = self.forward_ret_dict['reg_preds']

        bs, _, H, W = cls_preds.size()

        # yy = torch.arange(H, dtype=dtype, device=device)
        # xx = torch.arange(W, dtype=dtype, device=device)
        # grid_y, grid_x = torch.meshgrid(yy, xx)

        # grid_y = (grid_y * y_size + y_lower).flatten()    # [188*188]
        # grid_x = (grid_x * x_size + x_lower).flatten()
        # grid = torch.stack([grid_x, grid_y], dim=1).unsqueeze(0).repeat(batch_size, 1, 1)     # [bs, 188*188, 2]

        
        # ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])   # [188, 188]
        # ys = ys.view(1, 1, H, W).repeat(bs, 1, 1, 1).to(cls_preds)          # [bs, 1, 188, 188]
        # xs = xs.view(1, 1, H, W).repeat(bs, 1, 1, 1).to(cls_preds)

        # cls_gt = self.forward_ret_dict['cls_targets']
        # reg_gt = self.forward_ret_dict['reg_targets']
        # hm_gt = self.forward_ret_dict['hm_targets']
        
        # batch_det_dim = torch.exp(preds_dict['dim'])        # [N, 3, H, W]
        # batch_det_rots = preds_dict['rot'][:, 0:1, :, :]    # [N, 1, H, W]
        # batch_det_rotc = preds_dict['rot'][:, 1:2, :, :]    # [N, 1, H, W]
        # batch_det_reg = preds_dict['reg']                   # [N, 2, H, W]
        # batch_det_hei = preds_dict['height']                # [N, 1, H, W]
        # batch_det_rot = torch.atan2(batch_det_rots, batch_det_rotc)
        # batch_det_xs = xs + batch_det_reg[:, 0:1, :, :]
        # batch_det_ys = ys + batch_det_reg[:, 1:2, :, :]
        # batch_det_xs = batch_det_xs * self.stride * self.voxel_size[0] + self.point_cloud_range[0]
        # batch_det_ys = batch_det_ys * self.stride * self.voxel_size[1] + self.point_cloud_range[1]
        # # (B, 7, H, W)
        # batch_box_preds = torch.cat([batch_det_xs, batch_det_ys, batch_det_hei, batch_det_dim, batch_det_rot], dim=1)

        # batch_box_preds = _transpose_and_gather_feat(batch_box_preds, example['ind'][task_id])

        batch_box_preds = self.forward_ret_dict['batch_box_preds']
        batch_box_targets = self.forward_ret_dict['batch_gt_boxes']
        batch_box_preds = centernet_utils._transpose_and_gather_feat(
            batch_box_preds.reshape(bs, -1, H, W),
            indices
        )

        # hm_gt = self.forward_ret_dict['hm_targets']
        # target_box = example['anno_box'][task_id]
        # batch_gt_dim = torch.exp(target_box[..., 3:6])
        # batch_gt_reg = target_box[..., 0:2]
        # batch_gt_hei = target_box[..., 2:3]
        # batch_gt_rot = torch.atan2(target_box[..., -2:-1], target_box[..., -1:])
        # batch_gt_xs = _transpose_and_gather_feat(xs, example['ind'][task_id]) + batch_gt_reg[..., 0:1]
        # batch_gt_ys = _transpose_and_gather_feat(ys, example['ind'][task_id]) + batch_gt_reg[..., 1:2]
        # batch_gt_xs = batch_gt_xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
        # batch_gt_ys = batch_gt_ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]
        # # (B, max_obj, 7)
        # batch_box_targets = torch.cat([batch_gt_xs, batch_gt_ys, batch_gt_hei, batch_gt_dim, batch_gt_rot], dim=-1)
        
        iou_targets = iou3d_nms_utils.boxes_iou3d_gpu(
            batch_box_preds.reshape(-1, 7),
            batch_box_targets.reshape(-1, 7)
        )[range(batch_box_preds.reshape(-1, 7).shape[0]), range(batch_box_targets.reshape(-1, 7).shape[0])]

        return iou_targets.reshape(bs, -1, 1)

    def generate_predicted_boxes(self, batch_size, cls_preds, reg_preds, local_maximum=True):
        """
        Args:
            batch_size: B
            cls_preds: B H W 3
            reg_preds: B H W 8

        Returns:
            batch_cls_preds N1+N2+N3 3
            batch_box_pred N1+N2+N3 7
            batch_index N1+N2+N3
        """
        dtype = cls_preds.dtype
        device = cls_preds.device
        B, H, W, _ = cls_preds.shape
        
        x_lower, y_lower, z_lower = self.point_cloud_range[:3]
        x_size, y_size = self.voxel_size[0] * self.stride, self.voxel_size[1] * self.stride

        yy = torch.arange(H, dtype=dtype, device=device)
        xx = torch.arange(W, dtype=dtype, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx)

        grid_y = (grid_y * y_size + y_lower).flatten()
        grid_x = (grid_x * x_size + x_lower).flatten()
        grid = torch.stack([grid_x, grid_y], dim=1).unsqueeze(0).repeat(batch_size, 1, 1)

        reg_preds = reg_preds.reshape(batch_size, -1, reg_preds.shape[3]).contiguous()

        x_preds = (grid[:, :, 0] + reg_preds[:, :, 0]*x_size).unsqueeze(2)
        y_preds = (grid[:, :, 1] + reg_preds[:, :, 1]*y_size).unsqueeze(2)

        rot_preds = torch.atan2(reg_preds[:, :, 7], reg_preds[:, :, 6]).unsqueeze(2)
        height_preds = reg_preds[:, :, 2].unsqueeze(2)
        dim_preds = torch.exp(reg_preds[:, :, 3:6])
        if self.pred_velo:
            velo_preds = reg_preds[:, :, 8:]
            box_preds = torch.cat([x_preds, y_preds, height_preds, dim_preds, rot_preds, velo_preds], dim=2)
        else:
            box_preds = torch.cat([x_preds, y_preds, height_preds, dim_preds, rot_preds], dim=2)

        cls_preds_pool = cls_preds.permute(0, 3, 1, 2).contiguous()
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)

        if local_maximum:
            target_ind = torch.arange(H*W, dtype=torch.int64, device=device).\
                            unsqueeze(0).unsqueeze(2).expand_as(cls_preds)
            _, ind = F.max_pool2d(cls_preds_pool, kernel_size=3, stride=1, padding=1, return_indices=True)
            ind = ind.permute(0, 2, 3, 1).reshape(batch_size, -1, 3).contiguous()  # B H*W 3
            positives = (target_ind == ind).any(dim=2)

            batch_box_preds = box_preds[positives, :].reshape(-1, box_preds.shape[2])
            batch_cls_preds = cls_preds[positives, :].reshape(-1, self.num_class)

            positives_sum = positives.sum(dim=1)
            batch_index = []

            for k in range(batch_size):
                batch_index.append(torch.ones(positives_sum[k], dtype=torch.int64, device=cls_preds.device)*k)
            batch_index = torch.cat(batch_index, dim=0)
        else:
            batch_cls_preds = cls_preds.reshape(-1, self.num_class)
            batch_box_preds = box_preds.reshape(-1, box_preds.shape[2])

            batch_index = []
            for k in range(batch_size):
                batch_index.append(torch.ones(cls_preds.shape[1], dtype=torch.int64, device=cls_preds.device)*k)
            batch_index = torch.cat(batch_index, dim=0)

        return batch_cls_preds, batch_box_preds, batch_index

    def get_loss(self, tb_dict=None):
        cls_preds = self.forward_ret_dict['cls_preds']
        reg_preds = self.forward_ret_dict['reg_preds']
        cls_gt = self.forward_ret_dict['cls_targets']
        reg_gt = self.forward_ret_dict['reg_targets']
        hm_gt = self.forward_ret_dict['hm_targets']

        bs, H, W, _ = reg_preds.shape
        pos_ind = (cls_gt[:, :, 1]*W + cls_gt[:, :, 0])
        pos_cls = cls_gt[:, :, 2]

        positives = pos_cls > 0
        negatives = pos_cls == 0
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        reg_gt[negatives, :] = 0
        
        # calculate regression loss
        if self.model_cfg.LOSS_CONFIG['REG_LOSS'] != 'RegLoss':
            reg_preds = reg_preds.view(bs, -1, reg_preds.shape[3]).contiguous()
            reg_preds = torch.gather(reg_preds, dim=1,\
                                     index=pos_ind.unsqueeze(2).expand_as(reg_gt))
            reg_preds[negatives, :] = 0
            reg_loss = self.reg_loss_func(reg_preds,
                                          reg_gt,
                                          reg_weights)
            reg_loss = reg_loss.sum() / bs
        else:
            reg_loss = self.reg_loss_func(reg_preds, reg_gt, pos_ind, positives)
            reg_loss = (reg_loss * reg_loss.new_tensor(self.code_weights)).sum()
        
        # calculate classifiction loss
        cls_preds = self._sigmoid(cls_preds)
        cls_loss = self.cls_loss_func(
            cls_preds,
            hm_gt,
            pos_ind,
            positives,
            torch.clamp(pos_cls-1, min=0)
        )
        
        tb_dict['rpn_loss_loc'] = reg_loss.item()
        tb_dict['rpn_loss_cls'] = cls_loss.item()

        rpn_loss = self.cls_weight*cls_loss + self.reg_weight*reg_loss
        
        # added iou loss
        if self.model_cfg.LOSS_CONFIG.get('IOU_LOSS', None) is not None:
            iou_target = self.assign_iou_target(pos_ind)

            iou_loss = self.reg_loss_func(iou_pred, iou_target, pos_ind, positives)
            tb_dict['rpn_loss_ious'] = iou_loss.sum()
            rpn_loss += self.iou_weight * iou_loss.sum()
            ret['iou_loss'] = iou_loss.detach().cpu()


        tb_dict['rpn_loss'] = rpn_loss.item()

        if math.isnan(tb_dict['rpn_loss']):
            pass

        return rpn_loss, tb_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        shared_features = self.shared_layers(spatial_features_2d)

        reg_list = []
        for task in self.tasks.keys():
            task_pred = self.task_layers[task](shared_features)
            if task == 'hm':
                cls_preds = task_pred.permute(0, 2, 3, 1).contiguous()
            else:
                reg_list.append(task_pred.permute(0, 2, 3, 1).contiguous())
        reg_preds = torch.cat(reg_list, dim=3)

        # if not self.training and self.tta:
        #     cls_preds, reg_preds = self.test_time_augment(cls_preds, reg_preds)

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['reg_preds'] = reg_preds

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds, batch_index = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds,
                reg_preds=reg_preds,
                local_maximum=self.model_cfg.get("LOCAL_MAXIMUM", True)
            )
            # attention: here exist a bug, please do the sigmoid opearation
            # during self.generate_predicted_boxes() functions
            # will set an experiment to prove the problem
            data_dict['batch_cls_preds'] = torch.sigmoid(batch_cls_preds)
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['batch_index'] = batch_index
            data_dict['cls_preds_normalized'] = True

            # self.forward_ret_dict['batch_box_preds'] = batch_box_preds
            # self.forward_ret_dict['batch_gt_boxes'] = data_dict['gt_boxes']

        return data_dict

    def build_losses(self, losses_cfg):
        cls_loss_name = 'SigmoidFocalClassificationLoss' if losses_cfg.get('CLS_LOSS', None) is None \
            else losses_cfg.CLS_LOSS
        self.add_module(
            'cls_loss_func',
            getattr(loss_utils, cls_loss_name)(alpha=losses_cfg.CLS_LOSS_ALPHA, beta=losses_cfg.CLS_LOSS_BETA)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS', None) is None \
            else losses_cfg.REG_LOSS
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

        self.cls_weight = losses_cfg.LOSS_WEIGHTS['cls_weight']
        self.reg_weight = losses_cfg.LOSS_WEIGHTS['reg_weight']
        self.code_weights = losses_cfg.LOSS_WEIGHTS['code_weights']
        if losses_cfg.get('IOU_LOSS', None) is not None:
            self.iou_weight = losses_cfg.LOSS_WEIGHTS['iou_weight']
