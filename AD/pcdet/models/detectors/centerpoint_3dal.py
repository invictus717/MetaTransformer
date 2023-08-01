"""
This file is borrowed from 3DAL-Project: https://gitlab.com/pjlab-adg/3dal-toolchain-v2
"""
import os

import torch
import torch.nn as nn
import numpy as np

from pcdet.utils import common_utils

from ...ops.iou3d_nms import iou3d_nms_utils

from ..model_utils import model_nms_utils, centernet_utils
from ..model_utils.ensemble import wbf_online
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_3d import vfe as pcd_vfe
from ..backbones_2d import map_to_bev as pcd_map_to_bev
from ...utils.spconv_utils import find_all_spconv_keys


class CenterPoint3DAL(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        # self.tta = self.dataset.tta
        self.tta = False
        # TODO Currently no support for the TTA operation
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.second_stage = model_cfg.SECOND_STAGE
        self.predict_boxes_when_training = self.second_stage or \
                                           model_cfg.DENSE_HEAD.LOSS_CONFIG.get('IOU_LOSS', None) is not None

        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        tb_dict = {}
        loss, tb_dict = self.dense_head.get_loss(tb_dict)

        if self.second_stage:
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss += loss_rcnn

        return loss, tb_dict, disp_dict

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
        }

        vfe = pcd_vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_point_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
        )

        model_info_dict['num_point_features'] = vfe.get_output_feature_dim()

        if self.model_cfg.BACKBONE_3D.NAME == "Backbone3D_align":
            backbones_3d_cp_3dal = "VoxelResBackBone8x"
        else:
            backbones_3d_cp_3dal = str(self.model_cfg.BACKBONE_3D.NAME)

        backbone3d = backbones_3d.__all__[backbones_3d_cp_3dal](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
        )
        model_info_dict['num_point_features'] = backbone3d.num_point_features

        map_to_bev = pcd_map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['num_bev_features'] = map_to_bev.num_bev_features

        if self.model_cfg.BACKBONE_2D.NAME == "Backbone2D":
            backbones_2d_cp_3dal = "BaseBEVBackbone"
        else:
            backbones_2d_cp_3dal = str(self.model_cfg.Backbone2D.NAME)

        backbone2d = backbones_2d.__all__[backbones_2d_cp_3dal](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['num_bev_features'] = backbone2d.num_bev_features

        dense_head = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            tta=self.tta,
            predict_boxes_when_training=self.predict_boxes_when_training
        )

        self.add_module('vfe', vfe)
        self.add_module('backbone3d', backbone3d)
        self.add_module('map_to_bev', map_to_bev)
        self.add_module('backbone2d', backbone2d)
        self.add_module('dense_head', dense_head)
        module_list = [vfe, backbone3d, map_to_bev, backbone2d, dense_head]

        if self.second_stage:
            roi_head = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
                model_cfg=self.model_cfg.ROI_HEAD,
                input_bev_channels=model_info_dict['num_bev_features'],
                num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
                grid_size=model_info_dict['grid_size'],
                voxel_size=model_info_dict['voxel_size'],
                point_cloud_range=model_info_dict['point_cloud_range']
            )
            self.add_module('roi_head', roi_head)
            module_list.append(roi_head)

        return module_list

    @staticmethod
    def test_time_augment(data_dict, pred_dicts):
        tta_ops = data_dict["tta_ops"]
        tta_num = len(tta_ops)
        bs = int(data_dict["batch_size"] // tta_num)
        max_num = max([len(x["pred_boxes"]) for x in pred_dicts])
        box_num = []

        # process the boxes from dict into Tensor
        boxes = torch.zeros((data_dict["batch_size"],
                             max_num,
                             pred_dicts[0]["pred_boxes"].shape[-1]),
                            dtype=pred_dicts[0]["pred_boxes"].dtype,
                            device=pred_dicts[0]["pred_boxes"].device)
        scores = torch.zeros((data_dict["batch_size"], max_num, 1),
                             dtype=pred_dicts[0]["pred_scores"].dtype,
                             device=pred_dicts[0]["pred_scores"].device)
        labels = torch.zeros((data_dict["batch_size"], max_num, 1),
                             dtype=pred_dicts[0]["pred_labels"].dtype,
                             device=pred_dicts[0]["pred_labels"].device)
        for i, pred in enumerate(pred_dicts):
            boxes[i, :pred["pred_boxes"].__len__(), :] = pred["pred_boxes"]
            scores[i, :pred["pred_scores"].__len__(), 0] = pred["pred_scores"]
            labels[i, :pred["pred_labels"].__len__(), 0] = pred["pred_labels"]
            box_num.append(pred["pred_boxes"].__len__())

        # scatter the original and augmented predict boxes
        boxes = boxes.reshape(bs, tta_num, max_num, -1)
        dim = boxes.shape[-1]

        # restore the augmented preds to original coordinates
        for i, tta_cfg in enumerate(tta_ops):
            if tta_cfg == "tta_original":
                continue
            name, param = tta_cfg.split("_")[1], tta_cfg.split("_")[2]

            if name == "flip":
                if param == "x":
                    boxes[:, i, :, 1] = -boxes[:, i, :, 1]
                    boxes[:, i, :, 6] = -boxes[:, i, :, 6]
                    if dim > 7:
                        boxes[:, i, :, 8] = -boxes[:, i, :, 8]
                elif param == "y":
                    boxes[:, i, :, 0] = -boxes[:, i, :, 0]
                    boxes[:, i, :, 6] = -(boxes[:, i, :, 6] + np.pi)
                    if dim > 7:
                        boxes[:, i, :, 7] = -boxes[:, i, :, 7]
                elif param == 'xy':
                    boxes[:, i, :, 0:2] = -boxes[:, i, :, 0:2]
                    boxes[:, i, :, 6] = boxes[:, i, :, 6] + np.pi
                    if dim > 7:
                        boxes[:, i, :, 7:9] = -boxes[:, i, :, 7:9]
            elif name == "rot":
                param = -float(param)
                boxes[:, i, :, 0:3] = common_utils.rotate_points_along_z(
                    boxes[:, i, :, 0:3],
                    np.repeat(np.array([param]), bs)
                )
                boxes[:, i, :, 6] += param
                if dim > 7:
                    boxes[:, i, :, 7:9] = common_utils.rotate_points_along_z(
                        torch.cat([boxes[:, i, :, 7:9],
                                   torch.zeros((bs, max_num, 1),
                                               dtype=pred_dicts[0]["pred_boxes"].dtype,
                                               device=pred_dicts[0]["pred_boxes"].device)], dim=-1),
                        # TODO: Add dtype and device here
                        np.repeat(np.array([param]), bs)
                    )[0][:, 0:2]
            elif name == "scale":
                param = float(param)
                boxes[:, i, :, :6] /= param
                if dim > 7:
                    boxes[:, i, :, 7:9] /= param

        # fuse all the results with weighted box fusion
        data_dict["batch_size"] = bs
        boxes = boxes.squeeze(0)
        boxes, scores, labels = wbf_online(boxes, scores, labels)
        return boxes, scores, labels

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        bs = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(bs):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [
                        torch.range(1, self.num_class, device=cls_preds[0].device, dtype=torch.int)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                score_preds, label_preds = torch.max(cls_preds, dim=-1)

                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1

                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=score_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            if self.tta:
                recall_dict = self.generate_recall_record(
                    box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                    recall_dict=recall_dict, batch_index=0, data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST
                )
            else:
                recall_dict = self.generate_recall_record(
                    box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                    recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST
                )
            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        # tta before output
        if not self.training and self.tta:
            # batch_dict['batch_size'] = batch_dict['batch_size'] // batch_dict['batch_size']

            final_boxes, final_scores, final_labels = self.test_time_augment(batch_dict, pred_dicts)
            pred_dicts = []
            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)
        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None

        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
