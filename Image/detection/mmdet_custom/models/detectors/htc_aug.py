# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.cascade_rcnn import CascadeRCNN
from mmdet.core import (bbox2result, bbox_mapping_back, multiclass_nms,
                        bbox2roi, merge_aug_masks, bbox_mapping)
import torch
import numpy as np
import torch.nn.functional as F


@DETECTORS.register_module()
class HybridTaskCascadeAug(CascadeRCNN):
    """Implementation of `HTC <https://arxiv.org/abs/1901.07518>`_"""
    
    def __init__(self, **kwargs):
        super(HybridTaskCascadeAug, self).__init__(**kwargs)
        
    @property
    def with_semantic(self):
        """bool: whether the detector has a semantic head"""
        return self.roi_head.with_semantic
    
    def aug_test(self, imgs, img_metas, rescale=False):
        return [self.aug_test_vote(imgs, img_metas, rescale)]
    
    def merge_aug_results(self, aug_bboxes, aug_scores, img_metas):
        recovered_bboxes = []
        for bboxes, img_info, scores in zip(aug_bboxes, img_metas, aug_scores):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            recovered_bboxes.append(bboxes)
        
        bboxes = torch.cat(recovered_bboxes, dim=0)
        scores = torch.cat(aug_scores, dim=0)
        
        return bboxes, scores
    
    def remove_boxes(self, boxes, scales=['s', 'm', 'l']):
        # print(boxes.shape, min_scale * min_scale, max_scale * max_scale)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        flag = areas < 0.0
        
        if 's' in scales:
            flag = flag | (areas <= 32.0 * 32.0)
        if 'm' in scales:
            flag = flag | ((areas > 32.0 * 32.0) & (areas <= 96.0 * 96.0))
        if 'm-' in scales:
            flag = flag | ((areas > 32.0 * 32.0) & (areas <= 64.0 * 64.0))
        if 'm+' in scales:
            flag = flag | ((areas > 64.0 * 64.0) & (areas <= 96.0 * 96.0))
        if 'l' in scales:
            flag = flag | (areas > 96.0 * 96.0)
        if 'l-' in scales:
            flag = flag | ((areas > 96.0 * 96.0) & (areas < 512.0 * 512.0))
        if 'l+' in scales:
            flag = flag | (areas > 512.0 * 512.0)
        keep = torch.nonzero(flag, as_tuple=False).squeeze(1)
        
        return keep

    
    def aug_bbox_forward(self, x, proposal_list, img_metas, rescale=False):
        
        if self.roi_head.with_semantic:
            _, semantic_feat = self.roi_head.semantic_head(x)
        else:
            semantic_feat = None
        
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        
        # "ms" in variable names means multi-stage
        ms_scores = []
        rcnn_test_cfg = self.roi_head.test_cfg
        
        rois = bbox2roi(proposal_list)
        
        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.roi_head.bbox_head[-1].num_classes)
            ]] * num_imgs
            
            if self.roi_head.with_mask:
                mask_classes = self.roi_head.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results
            
            return results
        
        for i in range(self.roi_head.num_stages):
            bbox_head = self.roi_head.bbox_head[i]
            bbox_results = self.roi_head._bbox_forward(
                i, x, rois, semantic_feat=semantic_feat)
            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            ms_scores.append(cls_score)
            
            if i < self.roi_head.num_stages - 1:
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refine_rois = bbox_head.regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refine_rois)
                rois = torch.cat(refine_rois_list)
        
        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]
        
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.roi_head.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=None)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            
        return det_bboxes[0], det_labels[0], semantic_feat
    
    def aug_segm_forward(self, img_feats, det_bboxes, det_labels, semantic_feats, img_metas):
        
        rcnn_test_cfg = self.roi_head.test_cfg
        
        if det_bboxes.shape[0] == 0:
            segm_results = [[]
                            for _ in range(self.roi_head.mask_head[-1].num_classes)]
        else:
            aug_masks = []
            aug_img_metas = []
            for x, img_meta, semantic in zip(img_feats, img_metas,
                                             semantic_feats):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                flip_direction = img_meta[0]['flip_direction']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip, flip_direction)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.roi_head.mask_roi_extractor[-1](
                    x[:len(self.roi_head.mask_roi_extractor[-1].featmap_strides)],
                    mask_rois)
                if self.roi_head.with_semantic:
                    semantic_feat = semantic
                    mask_semantic_feat = self.roi_head.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    if mask_semantic_feat.shape[-2:] != mask_feats.shape[
                                                        -2:]:
                        mask_semantic_feat = F.adaptive_avg_pool2d(
                            mask_semantic_feat, mask_feats.shape[-2:])
                    mask_feats += mask_semantic_feat
                last_feat = None
                for i in range(self.roi_head.num_stages):
                    mask_head = self.roi_head.mask_head[i]
                    if self.roi_head.mask_info_flow:
                        mask_pred, last_feat = mask_head(
                            mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                    aug_img_metas.append(img_meta)
            merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                           self.roi_head.test_cfg)
            
            ori_shape = img_metas[0][0]['ori_shape']
            segm_results = self.roi_head.mask_head[-1].get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                rcnn_test_cfg,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        return segm_results
    
    def aug_test_vote(self, imgs, img_metas, rescale=False):
        # recompute feats to save memory
        feats = self.extract_feats(imgs)
        aug_bboxes = []
        aug_scores = []
        semantic_feats = []
        
        for i, (x, img_meta) in enumerate(zip(feats, img_metas)):
            proposal_list = self.rpn_head.simple_test_rpn(x, img_meta)
            det_bboxes, det_scores, semantic_feat = self.aug_bbox_forward(x, proposal_list, img_meta, rescale=False)
            restored_bboxes, _ = self.merge_aug_results([det_bboxes], [det_scores], [img_meta])
            keeped = self.remove_boxes(restored_bboxes, self.test_cfg.aug.scale_ranges[i // 2])
            det_bboxes, det_scores = det_bboxes[keeped, :], det_scores[keeped, :]
            aug_bboxes.append(det_bboxes)
            aug_scores.append(det_scores)
            semantic_feats.append(semantic_feat)
        
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_results(
            aug_bboxes, aug_scores, img_metas)

        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                self.test_cfg.aug.score_thr,
                                                self.test_cfg.aug.nms,
                                                self.test_cfg.aug.max_per_img)
        
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.roi_head.bbox_head[-1].num_classes)
        
        if self.with_mask:
            segm_results = self.aug_segm_forward(feats, _det_bboxes, det_labels, semantic_feats, img_metas)
            return bbox_results, segm_results
        else:
            return bbox_results