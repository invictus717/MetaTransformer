"""
This file is borrowed from 3DAL-Project: https://gitlab.com/pjlab-adg/3dal-toolchain-v2
"""

import copy

import numpy as np
import torch

from ...ops.iou3d_nms import iou3d_nms_utils


def prefilter_boxes(boxes, scores, labels, weights, thresh):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for i in range(len(boxes)):
        if len(boxes[i]) != len(scores[i]):
            raise ValueError("Length of boxes not equal to length of scores.")
        if len(boxes[i]) != len(labels[i]):
            raise ValueError("Length of boxes not equal to length of labels.")

        for j in range(len(boxes[i])):
            score = scores[i][j][0]
            # if score < thresh:
            #     continue
            label = int(labels[i][j][0])
            if label == 0:
                continue
            # import pdb; pdb.set_trace()
            box = boxes[i][j]
            x = float(box[0])
            y = float(box[1])
            z = float(box[2])
            dx = float(box[3])
            dy = float(box[4])
            dz = float(box[5])
            yaw = float(box[6])

            new_box = [int(label), float(score) * weights[i], x, y, z, dx, dy, dz, yaw]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(new_box)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1] >= thresh[k - 1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    Param:
        boxes: set of boxes to fuse
        conf_type: type of confidence, one of 'avg' or 'max'
    Return:
        weighted box
    """
    weighted_box = np.zeros(9, dtype=np.float32)
    conf = 0
    conf_list = []
    for box in boxes:
        weighted_box[2:] += (box[1] * box[2:])
        conf += box[1]
        conf_list.append(box[1])

    # assign label
    weighted_box[0] = boxes[0][0]

    # assign new score
    if conf_type == 'avg':
        weighted_box[1] = conf / len(boxes)
    elif conf_type == 'max':
        weighted_box[1] = np.array(conf_list).max()

    weighted_box[2:] /= conf
    weighted_box[-1] = boxes[conf_list.index(max(conf_list))][-1]

    return weighted_box


def find_matching_box(boxes_list, new_box, iou_thresh, iou_type):
    if len(boxes_list) == 0:
        return -1, iou_thresh

    boxes_list = np.array(boxes_list)

    boxes_gpu = copy.deepcopy(torch.from_numpy(boxes_list[:, 2:]).float().cuda())
    new_box = torch.from_numpy(new_box[2:]).unsqueeze(0).float().cuda()
    if iou_type == '3d':
        ious = iou3d_nms_utils.boxes_iou3d_gpu(new_box, boxes_gpu)
    elif iou_type == 'bev':
        ious = iou3d_nms_utils.boxes_iou_bev(new_box, boxes_gpu)

    best_idx = ious.argmax().item()
    best_iou = ious[0][best_idx].item()

    if best_iou <= iou_thresh:
        best_iou = iou_thresh
        best_idx = -1

    return best_idx, best_iou


def weighted_boxes_fusion_3d(boxes_list, scores_list, labels_list,
                             weights=None, iou_thr=None, skip_box_thr=None,
                             conf_type='avg', iou_type='3d',
                             allows_overflow=False):
    '''
    Param:
        boxes_list: list of boxes predictions from each model, each box is 7-dim
                    It has 3 dimensions (models_number, model_preds, 6)
                    Order of boxes: x,y,z,dx,dy,dz,yaw. We expect float normalized coordinates [0; 1]
        scores_list: list of scores of each box from each model
        labels_list: list of labels of each box from each model
        weights: list of weights for each model.
                 Default: None, which means weight == 1 for each model
        iou_thr: IoU threshold for boxes to be a match
        skip_box_thr: exclude boxes with score lower than this threshold
        conf_type: confidence calculation type
                   'avg': average value, 'max': maximum value
        allows_overflow: false if we want confidence score not exceed 1.0
    Return:
        boxes: new boxes coordinates (Order of boxes: x1, y1, z1, x2, y2, z2).
        scores: new confidence scores
        labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights),
                                                                                                     len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)
    if conf_type not in ['avg', 'max']:
        print('Error. Unknown conf_type: {}. Must be "avg" or "max". Use "avg"'.format(conf_type))
        conf_type = 'avg'
    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 7)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        # clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr[label - 1], iou_type)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())

        # rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            if not allows_overflow:
                weighted_boxes[i][1] = weighted_boxes[i][1] * min(weights.sum(), len(new_boxes[i])) / weights.sum()
            else:
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(new_boxes[i]) / weights.sum()

        if len(weighted_boxes) != 0:
            overall_boxes.append(np.array(weighted_boxes))

    if len(overall_boxes) == 0:
        return np.zeros((0, 7)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0].astype(int)

    return boxes, scores, labels
