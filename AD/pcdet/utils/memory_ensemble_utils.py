import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.models.model_utils.model_nms_utils import class_agnostic_nms


def consistency_ensemble(gt_infos_a, gt_infos_b, memory_ensemble_cfg):
    """
    Args:
        gt_infos_a:
            gt_boxes: (N, 9) [x, y, z, dx, dy, dz, heading, label, scores]  in LiDAR for previous pseudo boxes
            cls_scores: (N)
            iou_scores: (N)
            memory_counter: (N)

        gt_infos_b:
            gt_boxes: (M, 9) [x, y, z, dx, dy, dz, heading, label, scores]  in LiDAR for current pseudo boxes
            cls_scores: (M)
            iou_scores: (M)
            memory_counter: (M)

        memory_ensemble_cfg:

    Returns:
        gt_infos:
            gt_boxes: (K, 9) [x, y, z, dx, dy, dz, heading, label, scores]  in LiDAR for merged pseudo boxes
            cls_scores: (K)
            iou_scores: (K)
            memory_counter: (K)
    """
    gt_box_a, _ = common_utils.check_numpy_to_torch(gt_infos_a['gt_boxes'])
    gt_box_b, _ = common_utils.check_numpy_to_torch(gt_infos_b['gt_boxes'])
    gt_box_a, gt_box_b = gt_box_a.cuda(), gt_box_b.cuda()

    new_gt_box = gt_infos_a['gt_boxes']
    new_cls_scores = gt_infos_a['cls_scores']
    new_iou_scores = gt_infos_a['iou_scores']
    new_memory_counter = gt_infos_a['memory_counter']

    # if gt_box_b or gt_box_a don't have any predictions
    if gt_box_b.shape[0] == 0:
        gt_infos_a['memory_counter'] += 1
        return gt_infos_a
    elif gt_box_a.shape[0] == 0:
        return gt_infos_b

    # get ious
    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(gt_box_a[:, :7], gt_box_b[:, :7]).cpu()

    ious, match_idx = torch.max(iou_matrix, dim=1)
    ious, match_idx = ious.numpy(), match_idx.numpy()
    gt_box_a, gt_box_b = gt_box_a.cpu().numpy(), gt_box_b.cpu().numpy()

    match_pairs_idx = np.concatenate((
        np.array(list(range(gt_box_a.shape[0]))).reshape(-1, 1),
        match_idx.reshape(-1, 1)), axis=1)

    #########################################################
    # filter matched pair boxes by IoU
    # if matching succeeded, use boxes with higher confidence
    #########################################################

    iou_mask = (ious >= memory_ensemble_cfg.IOU_THRESH)

    matching_selected = match_pairs_idx[iou_mask]
    gt_box_selected_a = gt_box_a[matching_selected[:, 0]]
    gt_box_selected_b = gt_box_b[matching_selected[:, 1]]

    # assign boxes with higher confidence
    score_mask = gt_box_selected_a[:, 8] < gt_box_selected_b[:, 8]
    if memory_ensemble_cfg.get('WEIGHTED', None):
        weight = gt_box_selected_a[:, 8] / (gt_box_selected_a[:, 8] + gt_box_selected_b[:, 8])
        min_scores = np.minimum(gt_box_selected_a[:, 8], gt_box_selected_b[:, 8])
        max_scores = np.maximum(gt_box_selected_a[:, 8], gt_box_selected_b[:, 8])
        weighted_score = weight * (max_scores - min_scores) + min_scores
        new_gt_box[matching_selected[:, 0], :7] = weight.reshape(-1, 1) * gt_box_selected_a[:, :7] + \
                                                (1 - weight.reshape(-1, 1)) * gt_box_selected_b[:, :7]
        new_gt_box[matching_selected[:, 0], 8] = weighted_score
    else:
        new_gt_box[matching_selected[score_mask, 0], :] = gt_box_selected_b[score_mask, :]

    if gt_infos_a['cls_scores'] is not None:
        new_cls_scores[matching_selected[score_mask, 0]] = gt_infos_b['cls_scores'][
            matching_selected[score_mask, 1]]
    if gt_infos_a['iou_scores'] is not None:
        new_iou_scores[matching_selected[score_mask, 0]] = gt_infos_b['iou_scores'][
            matching_selected[score_mask, 1]]
    
    # for matching pairs, clear the ignore counter
    new_memory_counter[matching_selected[:, 0]] = 0

    #######################################################
    # If previous bboxes disappeared: ious <= 0.1
    #######################################################
    disappear_idx = (ious < memory_ensemble_cfg.IOU_THRESH).nonzero()[0]

    if memory_ensemble_cfg.get('MEMORY_VOTING', None) and memory_ensemble_cfg.MEMORY_VOTING.ENABLED:
        new_memory_counter[disappear_idx] += 1
        # ignore gt_boxes that ignore_count == IGNORE_THRESH
        ignore_mask = new_memory_counter >= memory_ensemble_cfg.MEMORY_VOTING.IGNORE_THRESH
        new_gt_box[ignore_mask, 7] = -1

        # remove gt_boxes that ignore_count >= RM_THRESH
        remain_mask = new_memory_counter < memory_ensemble_cfg.MEMORY_VOTING.RM_THRESH
        new_gt_box = new_gt_box[remain_mask]
        new_memory_counter = new_memory_counter[remain_mask]
        if gt_infos_a['cls_scores'] is not None:
            new_cls_scores = new_cls_scores[remain_mask]
        if gt_infos_a['iou_scores'] is not None:
            new_iou_scores = new_iou_scores[remain_mask]

    # Add new appear boxes
    ious_b2a, match_idx_b2a = torch.max(iou_matrix, dim=0)
    ious_b2a, match_idx_b2a = ious_b2a.numpy(), match_idx_b2a.numpy()

    newboxes_idx = (ious_b2a < memory_ensemble_cfg.IOU_THRESH).nonzero()[0]
    if newboxes_idx.shape[0] != 0:
        new_gt_box = np.concatenate((new_gt_box, gt_infos_b['gt_boxes'][newboxes_idx, :]), axis=0)
        if gt_infos_a['cls_scores'] is not None:
            new_cls_scores = np.concatenate((new_cls_scores, gt_infos_b['cls_scores'][newboxes_idx]), axis=0)
        if gt_infos_a['iou_scores'] is not None:
            new_iou_scores = np.concatenate((new_iou_scores, gt_infos_b['iou_scores'][newboxes_idx]), axis=0)
        new_memory_counter = np.concatenate((new_memory_counter, gt_infos_b['memory_counter'][newboxes_idx]), axis=0)

    new_gt_infos = {
        'gt_boxes': new_gt_box,
        'cls_scores': new_cls_scores if gt_infos_a['cls_scores'] is not None else None,
        'iou_scores': new_iou_scores if gt_infos_a['iou_scores'] is not None else None,
        'memory_counter': new_memory_counter
    }

    return new_gt_infos


def nms_ensemble(gt_infos_a, gt_infos_b, memory_ensemble_cfg):
    """
    Args:
        gt_infos_a:
            gt_boxes: (N, 9) [x, y, z, dx, dy, dz, heading, label, scores]  in LiDAR for previous pseudo boxes
            cls_scores: (N)
            iou_scores: (N)
            memory_counter: (N)

        gt_infos_b:
            gt_boxes: (M, 9) [x, y, z, dx, dy, dz, heading, label, scores]  in LiDAR for current pseudo boxes
            cls_scores: (M)
            iou_scores: (M)
            memory_counter: (M)

        memory_ensemble_cfg:

    Returns:
        gt_infos:
            gt_boxes: (K, 9) [x, y, z, dx, dy, dz, heading, label, scores]  in LiDAR for merged pseudo boxes
            cls_scores: (K)
            iou_scores: (K)
            memory_counter: (K)
    """
    gt_box_a, _ = common_utils.check_numpy_to_torch(gt_infos_a['gt_boxes'])
    gt_box_b, _ = common_utils.check_numpy_to_torch(gt_infos_b['gt_boxes'])

    if gt_box_b.shape[0] == 0:
        if memory_ensemble_cfg.get('MEMORY_VOTING', None) and memory_ensemble_cfg.MEMORY_VOTING.ENABLED:
            gt_infos_a['memory_counter'] += 1
        return gt_infos_a
    elif gt_box_a.shape[0] == 0:
        return gt_infos_b

    gt_box_a, gt_box_b = gt_box_a.cuda(), gt_box_b.cuda()

    gt_boxes = torch.cat((gt_box_a, gt_box_b), dim=0)
    if gt_infos_a['cls_scores'] is not None:
        new_cls_scores = np.concatenate((gt_infos_a['cls_scores'], gt_infos_b['cls_scores']), axis=0)
    if gt_infos_a['iou_scores'] is not None:
        new_iou_scores = np.concatenate((gt_infos_a['iou_scores'], gt_infos_b['iou_scores']), axis=0)
    new_memory_counter = np.concatenate((gt_infos_a['memory_counter'], gt_infos_b['memory_counter']), axis=0)

    selected, selected_scores = class_agnostic_nms(
        box_scores=gt_boxes[:, -1], box_preds=gt_boxes[:, :7], nms_config=memory_ensemble_cfg.NMS_CONFIG
    )

    gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(selected, list):
        selected = np.array(selected)
    else:
        selected = selected.cpu().numpy()

    if memory_ensemble_cfg.get('MEMORY_VOTING', None) and memory_ensemble_cfg.MEMORY_VOTING.ENABLED:
        iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(gt_box_a[:, :7], gt_box_b[:, :7])

        ious, _ = torch.max(iou_matrix, dim=1)
        ious = ious.cpu().numpy()

        gt_box_a_size = gt_box_a.shape[0]
        selected_a = selected[selected < gt_box_a_size]
        matched_mask = (ious[selected_a] > memory_ensemble_cfg.NMS_CONFIG.NMS_THRESH)
        match_idx = selected_a[matched_mask]
        new_memory_counter[match_idx] = 0

        # for previous bboxes disappeared
        disappear_idx = (ious < memory_ensemble_cfg.NMS_CONFIG.NMS_THRESH).nonzero()[0]
        new_memory_counter[disappear_idx] += 1
        # ignore gt_boxes that ignore_count == IGNORE_THRESH
        ignore_mask = new_memory_counter >= memory_ensemble_cfg.MEMORY_VOTING.IGNORE_THRESH
        gt_boxes[ignore_mask, 7] = -1

        # remove gt_boxes that ignore_count >= RM_THRESH
        rm_idx = (new_memory_counter >= memory_ensemble_cfg.MEMORY_VOTING.RM_THRESH).nonzero()[0]
        selected = np.setdiff1d(selected, rm_idx)

    selected_gt_boxes = gt_boxes[selected]

    new_gt_infos = {
        'gt_boxes': selected_gt_boxes,
        'cls_scores': new_cls_scores[selected] if gt_infos_a['cls_scores'] is not None else None,
        'iou_scores': new_iou_scores[selected] if gt_infos_a['iou_scores'] is not None else None,
        'memory_counter': new_memory_counter[selected]
    }

    return new_gt_infos


def bipartite_ensemble(gt_infos_a, gt_infos_b, memory_ensemble_cfg):
    """
    Args:
        gt_infos_a:
            gt_boxes: (N, 9) [x, y, z, dx, dy, dz, heading, label, scores]  in LiDAR for previous pseudo boxes
            cls_scores: (N)
            iou_scores: (N)
            memory_counter: (N)

        gt_infos_b:
            gt_boxes: (M, 9) [x, y, z, dx, dy, dz, heading, label, scores]  in LiDAR for current pseudo boxes
            cls_scores: (M)
            iou_scores: (M)
            memory_counter: (M)

        memory_ensemble_cfg:

    Returns:
        gt_infos:
            gt_boxes: (K, 9) [x, y, z, dx, dy, dz, heading, label, scores]  in LiDAR for merged pseudo boxes
            cls_scores: (K)
            iou_scores: (K)
            memory_counter: (K)
    """
    gt_box_a, _ = common_utils.check_numpy_to_torch(gt_infos_a['gt_boxes'])
    gt_box_b, _ = common_utils.check_numpy_to_torch(gt_infos_b['gt_boxes'])
    gt_box_a, gt_box_b = gt_box_a.cuda(), gt_box_b.cuda()

    new_gt_box = gt_infos_a['gt_boxes']
    new_cls_scores = gt_infos_a['cls_scores']
    new_iou_scores = gt_infos_a['iou_scores']
    new_memory_counter = gt_infos_a['memory_counter']

    # if gt_box_b or gt_box_a don't have any predictions
    if gt_box_b.shape[0] == 0:
        gt_infos_a['memory_counter'] += 1
        return gt_infos_a
    elif gt_box_a.shape[0] == 0:
        return gt_infos_b

    # bipartite matching
    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(gt_box_a[:, :7], gt_box_b[:, :7])

    iou_matrix = iou_matrix.cpu().numpy()

    a_idx, b_idx = linear_sum_assignment(-iou_matrix)

    gt_box_a, gt_box_b = gt_box_a.cpu().numpy(), gt_box_b.cpu().numpy()

    matching_paris_idx = np.concatenate((a_idx.reshape(-1, 1), b_idx.reshape(-1, 1)), axis=1)

    ious = iou_matrix[matching_paris_idx[:, 0], matching_paris_idx[:, 1]]

    # matched a boxes.
    matched_mask = ious > memory_ensemble_cfg.IOU_THRESH

    matching_selected = matching_paris_idx[matched_mask]
    gt_box_selected_a = gt_box_a[matching_selected[:, 0]]
    gt_box_selected_b = gt_box_b[matching_selected[:, 1]]

    # assign boxes with higher confidence
    score_mask = gt_box_selected_a[:, 8] < gt_box_selected_b[:, 8]
    new_gt_box[matching_selected[score_mask, 0], :] = gt_box_selected_b[score_mask, :]
    if gt_infos_a['cls_scores'] is not None:
        new_cls_scores[matching_selected[score_mask, 0]] = gt_infos_b['cls_scores'][
            matching_selected[score_mask, 1]]
    if gt_infos_a['iou_scores'] is not None:
        new_iou_scores[matching_selected[score_mask, 0]] = gt_infos_b['iou_scores'][
            matching_selected[score_mask, 1]]

    # for matched pairs, clear the ignore counter
    new_memory_counter[matching_selected[:, 0]] = 0

    ##############################################
    # disppeared boxes for previous pseudo boxes
    ##############################################
    gt_box_a_idx = np.array(list(range(gt_box_a.shape[0])))
    disappear_idx = np.setdiff1d(gt_box_a_idx, matching_selected[:, 0])

    if memory_ensemble_cfg.get('MEMORY_VOTING', None) and memory_ensemble_cfg.MEMORY_VOTING.ENABLED:
        new_memory_counter[disappear_idx] += 1
        # ignore gt_boxes that ignore_count == IGNORE_THRESH
        ignore_mask = new_memory_counter >= memory_ensemble_cfg.MEMORY_VOTING.IGNORE_THRESH
        new_gt_box[ignore_mask, 7] = -1

        # remove gt_boxes that ignore_count >= RM_THRESH
        remain_mask = new_memory_counter < memory_ensemble_cfg.MEMORY_VOTING.RM_THRESH
        new_gt_box = new_gt_box[remain_mask]
        new_memory_counter = new_memory_counter[remain_mask]
        if gt_infos_a['cls_scores'] is not None:
            new_cls_scores = new_cls_scores[remain_mask]
        if gt_infos_a['iou_scores'] is not None:
            new_iou_scores = new_iou_scores[remain_mask]

    ##############################################
    # new appear boxes for current pseudo boxes
    ##############################################
    gt_box_b_idx = np.array(list(range(gt_box_b.shape[0])))
    newboxes_idx = np.setdiff1d(gt_box_b_idx, matching_selected[:, 1])
    if newboxes_idx.shape[0] != 0:
        new_gt_box = np.concatenate((new_gt_box, gt_infos_b['gt_boxes'][newboxes_idx, :]), axis=0)
        if gt_infos_a['cls_scores'] is not None:
            new_cls_scores = np.concatenate((new_cls_scores, 
                                            gt_infos_b['cls_scores'][newboxes_idx]), axis=0)
        if gt_infos_a['iou_scores'] is not None:
            new_iou_scores = np.concatenate((new_iou_scores, 
                                            gt_infos_b['iou_scores'][newboxes_idx]), axis=0)
        new_memory_counter = np.concatenate((new_memory_counter,
                                            gt_infos_b['memory_counter'][newboxes_idx]), axis=0)

    new_gt_infos = {
        'gt_boxes': new_gt_box,
        'cls_scores': new_cls_scores if gt_infos_a['cls_scores'] is not None else None,
        'iou_scores': new_iou_scores if gt_infos_a['iou_scores'] is not None else None,
        'memory_counter': new_memory_counter
    }

    return new_gt_infos
