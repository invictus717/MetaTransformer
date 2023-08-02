import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, box_utils, loss_utils, common_utils
from .point_head_template import PointHeadTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class IASSD_Head(PointHeadTemplate):
    """
    A simple point-based detect head, which are used for IA-SSD.
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        detector_dim = self.model_cfg.get('INPUT_DIM', input_channels) # for spec input_channel
        self.cls_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=detector_dim,
            output_channels=num_class
        )
        self.box_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=detector_dim,
            output_channels=self.box_coder.code_size
        )
        
        self.box_iou3d_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.IOU_FC,
            input_channels=detector_dim,
            output_channels=1
        ) if self.model_cfg.get('IOU_FC', None) is not None else None

        # self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def build_losses(self, losses_cfg):
        # classification loss
        if losses_cfg.LOSS_CLS.startswith('WeightedBinaryCrossEntropy'):
            self.add_module(
                'cls_loss_func',
                loss_utils.WeightedBinaryCrossEntropyLoss()
            )
        elif losses_cfg.LOSS_CLS.startswith('WeightedCrossEntropy'):
            self.add_module(
                'cls_loss_func',
                loss_utils.WeightedClassificationLoss()
            )
        elif losses_cfg.LOSS_CLS.startswith('FocalLoss'):
            self.add_module(
                'cls_loss_func',
                loss_utils.SigmoidFocalClassificationLoss(
                    **losses_cfg.get('LOSS_CLS_CONFIG', {})
                )
            )
        else:
            raise NotImplementedError

        # regression loss
        if losses_cfg.LOSS_REG == 'WeightedSmoothL1Loss':
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedSmoothL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None),
                    **losses_cfg.get('LOSS_REG_CONFIG', {})
                )
            )
        elif losses_cfg.LOSS_REG == 'WeightedL1Loss':
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
                )
            )
        else:
            raise NotImplementedError

        # instance-aware loss
        if losses_cfg.get('LOSS_INS', None) is not None:
            if losses_cfg.LOSS_INS.startswith('WeightedBinaryCrossEntropy'):
                self.add_module(
                    'ins_loss_func',
                    loss_utils.WeightedBinaryCrossEntropyLoss()
                )
            elif losses_cfg.LOSS_INS.startswith('WeightedCrossEntropy'):
                self.add_module(
                    'ins_loss_func',
                    loss_utils.WeightedClassificationLoss()
                )
            elif losses_cfg.LOSS_INS.startswith('FocalLoss'):
                self.add_module(
                    'ins_loss_func',
                    loss_utils.SigmoidFocalClassificationLoss(
                        **losses_cfg.get('LOSS_CLS_CONFIG', {})
                    )
                )
            else:
                raise NotImplementedError

    def assign_stack_targets_IASSD(self, points, gt_boxes, extend_gt_boxes=None, weighted_labels=False,
                             ret_box_labels=False, ret_offset_labels=True,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0,
                             use_query_assign=False, central_radii=2.0, use_ex_gt_assign=False, fg_pc_ignore=False,
                             binary_label=False):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        box_idxs_labels = points.new_zeros(points.shape[0]).long() 
        gt_boxes_of_fg_points = []
        gt_box_of_points = gt_boxes.new_zeros((points.shape[0], 8))

        for k in range(batch_size):            
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)

            if use_query_assign: ##
                centers = gt_boxes[k:k + 1, :, 0:3]
                query_idxs_of_pts = roiaware_pool3d_utils.points_in_ball_query_gpu(
                    points_single.unsqueeze(dim=0), centers.contiguous(), central_radii
                    ).long().squeeze(dim=0) 
                query_fg_flag = (query_idxs_of_pts >= 0)
                if fg_pc_ignore:
                    fg_flag = query_fg_flag ^ box_fg_flag 
                    extend_box_idxs_of_pts[box_idxs_of_pts!=-1] = -1
                    box_idxs_of_pts = extend_box_idxs_of_pts
                else:
                    fg_flag = query_fg_flag
                    box_idxs_of_pts = query_idxs_of_pts
            elif use_ex_gt_assign: ##
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                extend_fg_flag = (extend_box_idxs_of_pts >= 0)
                
                extend_box_idxs_of_pts[box_fg_flag] = box_idxs_of_pts[box_fg_flag] #instance points should keep unchanged

                if fg_pc_ignore:
                    fg_flag = extend_fg_flag ^ box_fg_flag
                    extend_box_idxs_of_pts[box_idxs_of_pts!=-1] = -1
                    box_idxs_of_pts = extend_box_idxs_of_pts
                else:
                    fg_flag = extend_fg_flag 
                    box_idxs_of_pts = extend_box_idxs_of_pts 
                                
            elif set_ignore_flag: 
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1

            elif use_ball_constraint: 
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag

            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 or binary_label else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single
            bg_flag = (point_cls_labels_single == 0) # except ignore_id
            # box_bg_flag
            fg_flag = fg_flag ^ (fg_flag & bg_flag)
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]

            gt_boxes_of_fg_points.append(gt_box_of_fg_points)
            box_idxs_labels[bs_mask] = box_idxs_of_pts
            gt_box_of_points[bs_mask] = gt_boxes[k][box_idxs_of_pts]

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single


        gt_boxes_of_fg_points = torch.cat(gt_boxes_of_fg_points, dim=0)
        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'gt_box_of_fg_points': gt_boxes_of_fg_points,
            'box_idxs_labels': box_idxs_labels,
            'gt_box_of_points': gt_box_of_points,
        }
        return targets_dict

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                batch_size: int
                centers: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                centers_origin: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                encoder_coords: List of point_coords in SA
                gt_boxes (optional): (B, M, 8)
        Returns:
            target_dict:
            ...
        """
        target_cfg = self.model_cfg.TARGET_CONFIG
        gt_boxes = input_dict['gt_boxes']
        if gt_boxes.shape[-1] == 10:   #nscence
            gt_boxes = torch.cat((gt_boxes[..., 0:7], gt_boxes[..., -1:]), dim=-1)

        targets_dict_center = {}
        # assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        # assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
        batch_size = input_dict['batch_size']      
        if target_cfg.get('EXTRA_WIDTH', False):  # multi class extension
            extend_gt = box_utils.enlarge_box3d_for_class(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=target_cfg.EXTRA_WIDTH
            ).view(batch_size, -1, gt_boxes.shape[-1])
        else:
            extend_gt = gt_boxes

        extend_gt_boxes = box_utils.enlarge_box3d(
            extend_gt.view(-1, extend_gt.shape[-1]), extra_width=target_cfg.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        center_targets_dict = self.assign_stack_targets_IASSD(
            points=input_dict['centers'].detach(), 
            gt_boxes=extend_gt, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_box_labels=True
        )
        targets_dict_center['center_gt_box_of_fg_points'] = center_targets_dict['gt_box_of_fg_points']
        targets_dict_center['center_cls_labels'] = center_targets_dict['point_cls_labels']
        targets_dict_center['center_box_labels'] = center_targets_dict['point_box_labels'] #only center assign
        targets_dict_center['center_gt_box_of_points'] = center_targets_dict['gt_box_of_points']
        if target_cfg.get('INS_AWARE_ASSIGN', False):
            sa_ins_labels, sa_gt_box_of_fg_points, sa_xyz_coords, sa_gt_box_of_points, sa_box_idxs_labels = [],[],[],[],[]
            sa_ins_preds = input_dict['sa_ins_preds']
            for i in range(1, len(sa_ins_preds)): # valid when i = 1,2 for IA-SSD
                # if sa_ins_preds[i].__len__() == 0:
                #     continue
                sa_xyz = input_dict['encoder_coords'][i]
                if i == 1:
                    extend_gt_boxes = box_utils.enlarge_box3d(
                        gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]  #[0.2, 0.2, 0.2]
                    ).view(batch_size, -1, gt_boxes.shape[-1])             
                    sa_targets_dict = self.assign_stack_targets_IASSD(
                        points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                        set_ignore_flag=True, use_ex_gt_assign= False 
                    )
                if i >= 2:
                # if False:
                    extend_gt_boxes = box_utils.enlarge_box3d(
                        gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]
                    ).view(batch_size, -1, gt_boxes.shape[-1])             
                    sa_targets_dict = self.assign_stack_targets_IASSD(
                        points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                        set_ignore_flag=False, use_ex_gt_assign= True 
                    )
                # else:
                #     extend_gt_boxes = box_utils.enlarge_box3d(
                #         gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]
                #     ).view(batch_size, -1, gt_boxes.shape[-1]) 
                #     sa_targets_dict = self.assign_stack_targets_IASSD(
                #         points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                #         set_ignore_flag=False, use_ex_gt_assign= True 
                #     )
                sa_xyz_coords.append(sa_xyz)
                sa_ins_labels.append(sa_targets_dict['point_cls_labels'])
                sa_gt_box_of_fg_points.append(sa_targets_dict['gt_box_of_fg_points'])
                sa_gt_box_of_points.append(sa_targets_dict['gt_box_of_points'])
                sa_box_idxs_labels.append(sa_targets_dict['box_idxs_labels'])                
                
            targets_dict_center['sa_ins_labels'] = sa_ins_labels
            targets_dict_center['sa_gt_box_of_fg_points'] = sa_gt_box_of_fg_points
            targets_dict_center['sa_xyz_coords'] = sa_xyz_coords
            targets_dict_center['sa_gt_box_of_points'] = sa_gt_box_of_points
            targets_dict_center['sa_box_idxs_labels'] = sa_box_idxs_labels

        extra_method = target_cfg.get('ASSIGN_METHOD', None)
        if extra_method is not None and extra_method.NAME == 'extend_gt':
            extend_gt_boxes = box_utils.enlarge_box3d(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=extra_method.EXTRA_WIDTH
            ).view(batch_size, -1, gt_boxes.shape[-1])

            if extra_method.get('ASSIGN_TYPE', 'centers') == 'centers_origin': 
                points = input_dict['centers_origin'].detach()
            else:
                points = input_dict['centers'].detach() #default setting

            targets_dict = self.assign_stack_targets_IASSD(
                points=points, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False,
                ret_box_labels=True,
                use_ex_gt_assign=True, fg_pc_ignore=extra_method.FG_PC_IGNORE,
            )
            targets_dict_center['center_origin_gt_box_of_fg_points'] = targets_dict['gt_box_of_fg_points']
            targets_dict_center['center_origin_cls_labels'] = targets_dict['point_cls_labels']
            targets_dict_center['center_origin_box_idxs_of_pts'] = targets_dict['box_idxs_labels']
            targets_dict_center['gt_box_of_center_origin'] = targets_dict['gt_box_of_points']

        elif extra_method is not None and extra_method.NAME == 'extend_gt_factor':
            extend_gt_boxes = box_utils.enlarge_box3d_with_factor(
                gt_boxes.view(-1, gt_boxes.shape[-1]), factor=extra_method.EXTRA_FACTOR).view(batch_size, -1, gt_boxes.shape[-1])

            if extra_method.get('ASSIGN_TYPE', 'centers') == 'centers_origin': 
                points = input_dict['centers_origin'].detach()
            else:
                points = input_dict['centers'].detach()

            targets_dict = self.assign_stack_targets_IASSD(
                points=points, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False,
                ret_box_labels=False,
                use_ex_gt_assign=True, fg_pc_ignore=extra_method.FG_PC_IGNORE,
            )
            targets_dict_center['center_origin_gt_box_of_fg_points'] = targets_dict['gt_box_of_fg_points']
            targets_dict_center['center_origin_cls_labels'] = targets_dict['point_cls_labels']

        elif extra_method is not None and extra_method.NAME == 'extend_gt_for_class':
            extend_gt_boxes = box_utils.enlarge_box3d_for_class(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=extra_method.EXTRA_WIDTH
            ).view(batch_size, -1, gt_boxes.shape[-1])

            if extra_method.get('ASSIGN_TYPE', 'centers') == 'centers_origin': 
                points = input_dict['centers_origin'].detach()
            else:
                points = input_dict['centers'].detach()

            targets_dict = self.assign_stack_targets_IASSD(
                points=points, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False,
                ret_box_labels=False,
                use_ex_gt_assign=True, fg_pc_ignore=extra_method.FG_PC_IGNORE,
            )
            targets_dict_center['center_origin_gt_box_of_fg_points'] = targets_dict['gt_box_of_fg_points']
            targets_dict_center['center_origin_cls_labels'] = targets_dict['point_cls_labels']            

        elif extra_method is not None and extra_method.NAME == 'extend_query':
            extend_gt_boxes = None
            if extra_method.get('ASSIGN_TYPE', 'centers') == 'centers_origin': 
                points = input_dict['centers_origin'].detach()
            elif extra_method.get('ASSIGN_TYPE', 'centers') == 'centers': 
                points = input_dict['centers'].detach()

            targets_dict = self.assign_stack_targets_IASSD(
                points=points, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False,
                ret_box_labels=False, 
                use_query_assign=True, central_radii=extra_method.RADII, fg_pc_ignore=extra_method.FG_PC_IGNORE,
            )
            targets_dict_center['center_origin_gt_box_of_fg_points'] = targets_dict['gt_box_of_fg_points']
            targets_dict_center['center_origin_cls_labels'] = targets_dict['point_cls_labels']
        
        return targets_dict_center

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        # vote loss
        if self.model_cfg.TARGET_CONFIG.get('ASSIGN_METHOD') is not None and \
            self.model_cfg.TARGET_CONFIG.ASSIGN_METHOD.get('ASSIGN_TYPE')== 'centers_origin':
            if self.model_cfg.LOSS_CONFIG.get('LOSS_VOTE_TYPE', 'none') == 'ver1':
                center_loss_reg, tb_dict_3 = self.get_contextual_vote_loss_ver1()
            elif self.model_cfg.LOSS_CONFIG.get('LOSS_VOTE_TYPE', 'none') == 'ver2':
                center_loss_reg, tb_dict_3 = self.get_contextual_vote_loss_ver2()
            else: # 'none'
                center_loss_reg, tb_dict_3 = self.get_contextual_vote_loss()
        else:
            center_loss_reg, tb_dict_3 = self.get_vote_loss_loss() # center assign
        tb_dict.update(tb_dict_3)

        # semantic loss in SA layers
        if self.model_cfg.LOSS_CONFIG.get('LOSS_INS', None) is not None:
            assert ('sa_ins_preds' in self.forward_ret_dict) and ('sa_ins_labels' in self.forward_ret_dict)
            sa_loss_cls, tb_dict_0 = self.get_sa_ins_layer_loss()
            tb_dict.update(tb_dict_0)
        else:
            sa_loss_cls = 0

        # cls loss
        center_loss_cls, tb_dict_4 = self.get_center_cls_layer_loss()
        tb_dict.update(tb_dict_4)

        # reg loss
        if self.model_cfg.TARGET_CONFIG.BOX_CODER == 'PointResidualCoder':
            center_loss_box, tb_dict_5 = self.get_box_layer_loss()
        else:
            center_loss_box, tb_dict_5 = self.get_center_box_binori_layer_loss()
        tb_dict.update(tb_dict_5)    
        
        # corner loss
        if self.model_cfg.LOSS_CONFIG.get('CORNER_LOSS_REGULARIZATION', False):
            corner_loss, tb_dict_6 = self.get_corner_layer_loss()
            tb_dict.update(tb_dict_6)

        # iou loss
        iou3d_loss = 0
        if self.model_cfg.LOSS_CONFIG.get('IOU3D_REGULARIZATION', False):
            iou3d_loss, tb_dict_7 = self.get_iou3d_layer_loss()          
            tb_dict.update(tb_dict_7)
        
        point_loss = center_loss_reg + center_loss_cls + center_loss_box + corner_loss + sa_loss_cls + iou3d_loss             
        return point_loss, tb_dict


    def get_contextual_vote_loss(self, tb_dict=None):        
        pos_mask = self.forward_ret_dict['center_origin_cls_labels'] > 0
        center_origin_loss_box = []
        for i in self.forward_ret_dict['center_origin_cls_labels'].unique():
            if i <= 0: continue
            simple_pos_mask = self.forward_ret_dict['center_origin_cls_labels'] == i
            center_box_labels = self.forward_ret_dict['center_origin_gt_box_of_fg_points'][:, 0:3][(pos_mask & simple_pos_mask)[pos_mask==1]]
            centers_origin = self.forward_ret_dict['centers_origin']
            ctr_offsets = self.forward_ret_dict['ctr_offsets']
            centers_pred = centers_origin + ctr_offsets
            centers_pred = centers_pred[simple_pos_mask][:, 1:4]
            simple_center_origin_loss_box = F.smooth_l1_loss(centers_pred, center_box_labels)
            center_origin_loss_box.append(simple_center_origin_loss_box.unsqueeze(-1))
        center_origin_loss_box = torch.cat(center_origin_loss_box, dim=-1).mean()
        center_origin_loss_box = center_origin_loss_box * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('vote_weight')
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_origin_loss_reg': center_origin_loss_box.item()})
        return center_origin_loss_box, tb_dict


    def get_contextual_vote_loss_ver1(self, tb_dict=None):  
        box_idxs_of_pts = self.forward_ret_dict['center_origin_box_idxs_of_pts']
        center_box_labels = self.forward_ret_dict['gt_box_of_center_origin']
        centers_origin = self.forward_ret_dict['centers_origin']
        ctr_offsets = self.forward_ret_dict['ctr_offsets']
        centers_pred = centers_origin[:, 1:] + ctr_offsets[:, 1:]
        centers_pred = torch.cat([centers_origin[:, :1], centers_pred], dim=-1)
        batch_idx = self.forward_ret_dict['centers'][:,0]
        ins_num, ins_vote_loss = [],[]
        for cur_id in batch_idx.unique():
            batch_mask = (batch_idx == cur_id)
            for ins_idx in box_idxs_of_pts[batch_mask].unique():
                if ins_idx < 0:
                    continue
                ins_mask = (box_idxs_of_pts[batch_mask] == ins_idx)
                ins_num.append(ins_mask.sum().long().unsqueeze(-1))
                ins_vote_loss.append(F.smooth_l1_loss(centers_pred[batch_mask][ins_mask, 1:4], center_box_labels[batch_mask][ins_mask, 0:3], reduction='sum').unsqueeze(-1))                
        ins_num = torch.cat(ins_num, dim=-1).float()
        ins_vote_loss = torch.cat(ins_vote_loss, dim=-1)
        ins_vote_loss = ins_vote_loss / ins_num.float().clamp(min=1.0)
        vote_loss = ins_vote_loss.mean()
        vote_loss_ver1 = vote_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'vote_loss_ver1': vote_loss_ver1.item()})
        return vote_loss_ver1, tb_dict


    def get_contextual_vote_loss_ver2(self, tb_dict=None):  
        box_idxs_of_pts = self.forward_ret_dict['center_origin_box_idxs_of_pts']
        center_box_labels = self.forward_ret_dict['gt_box_of_center_origin']
        centers_origin = self.forward_ret_dict['centers_origin']
        ctr_offsets = self.forward_ret_dict['ctr_offsets']
        centers_pred = centers_origin[:, 1:] + ctr_offsets[:, 1:]
        centers_pred = torch.cat([centers_origin[:, :1], centers_pred], dim=-1)        
        batch_idx = self.forward_ret_dict['centers'][:,0]
        ins_num, ins_vote_loss, ins_mean_vote_loss = [],[],[]
        for cur_id in batch_idx.unique():
            batch_mask = (batch_idx == cur_id)
            for ins_idx in box_idxs_of_pts[batch_mask].unique():
                if ins_idx < 0:
                    continue
                ins_mask = (box_idxs_of_pts[batch_mask] == ins_idx) # box_idxs_of_pts[batch_mask][ins_mask]
                ins_num.append(ins_mask.sum().unsqueeze(-1))
                ins_vote_loss.append(F.smooth_l1_loss(centers_pred[batch_mask][ins_mask, 1:4], center_box_labels[batch_mask][ins_mask, 0:3], reduction='sum').unsqueeze(-1))                     
                ins_mean_vote_loss.append(F.smooth_l1_loss(centers_pred[batch_mask][ins_mask, 1:4], centers_pred[batch_mask][ins_mask, 1:4].mean(dim=0).repeat(centers_pred[batch_mask][ins_mask, 1:4].shape[0],1), reduction='sum').unsqueeze(-1))                
        ins_num = torch.cat(ins_num, dim=-1).float()
        ins_vote_loss = torch.cat(ins_vote_loss, dim=-1)
        ins_mean_vote_loss = torch.cat(ins_mean_vote_loss, dim=-1)
        vote_loss = ins_vote_loss + ins_mean_vote_loss * 0.5
        vote_loss = vote_loss / ins_num.clamp(min=1.0)
        vote_loss = vote_loss.mean()
        vote_loss_ver2 = vote_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'vote_loss_ver2': vote_loss_ver2.item()})
        return vote_loss_ver2, tb_dict


    def get_vote_loss_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        center_box_labels = self.forward_ret_dict['center_gt_box_of_fg_points'][:, 0:3]
        centers_origin = self.forward_ret_dict['centers_origin']
        ctr_offsets = self.forward_ret_dict['ctr_offsets']
        centers_pred = centers_origin + ctr_offsets
        centers_pred = centers_pred[pos_mask][:, 1:4]

        vote_loss = F.smooth_l1_loss(centers_pred, center_box_labels, reduction='mean')
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'vote_loss': vote_loss.item()})
        return vote_loss, tb_dict


    def get_center_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['center_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['center_cls_preds'].view(-1, self.num_class)
        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0

        cls_weights = (1.0 *negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        
        if self.model_cfg.LOSS_CONFIG.CENTERNESS_REGULARIZATION:
            centerness_mask = self.generate_center_ness_mask()
            one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1).repeat(1, one_hot_targets.shape[1])

        point_loss_cls = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights).mean(dim=-1).sum()
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'center_loss_cls': point_loss_cls.item(),
            'center_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict
    

    def get_sa_ins_layer_loss(self, tb_dict=None):
        sa_ins_labels = self.forward_ret_dict['sa_ins_labels']
        sa_ins_preds = self.forward_ret_dict['sa_ins_preds']
        sa_centerness_mask = self.generate_sa_center_ness_mask()
        sa_ins_loss, ignore = 0, 0
        for i in range(len(sa_ins_labels)): # valid when i =1, 2
            if len(sa_ins_preds[i]) != 0:
                try:
                    point_cls_preds = sa_ins_preds[i][...,1:].view(-1, self.num_class)
                except:
                    point_cls_preds = sa_ins_preds[i][...,1:].view(-1, 1)

            else:
                ignore += 1
                continue
            point_cls_labels = sa_ins_labels[i].view(-1)
            positives = (point_cls_labels > 0)
            negative_cls_weights = (point_cls_labels == 0) * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            pos_normalizer = positives.sum(dim=0).float()
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)

            one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
            one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
            one_hot_targets = one_hot_targets[..., 1:]

            if ('ctr' in self.model_cfg.LOSS_CONFIG.SAMPLE_METHOD_LIST[i+1][0]):
                centerness_mask = sa_centerness_mask[i]
                one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1).repeat(1, one_hot_targets.shape[1])

            point_loss_ins = self.ins_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights).mean(dim=-1).sum()        
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            point_loss_ins = point_loss_ins * loss_weights_dict.get('ins_aware_weight',[1]*len(sa_ins_labels))[i]

            sa_ins_loss += point_loss_ins
            if tb_dict is None:
                tb_dict = {}
            tb_dict.update({
                'sa%s_loss_ins' % str(i): point_loss_ins.item(),
                'sa%s_pos_num' % str(i): pos_normalizer.item()
            })

        sa_ins_loss = sa_ins_loss / (len(sa_ins_labels) - ignore)
        tb_dict.update({
                'sa_loss_ins': sa_ins_loss.item(),
            })
        return sa_ins_loss, tb_dict


    def generate_center_ness_mask(self):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['center_gt_box_of_fg_points']
        centers = self.forward_ret_dict['centers'][:,1:]
        centers = centers[pos_mask].clone().detach()
        offset_xyz = centers[:, 0:3] - gt_boxes[:, 0:3]

        offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)

        template = gt_boxes.new_tensor(([1, 1, 1], [-1, -1, -1])) / 2
        margin = gt_boxes[:, None, 3:6].repeat(1, 2, 1) * template[None, :, :]
        distance = margin - offset_xyz_canical[:, None, :].repeat(1, 2, 1)
        distance[:, 1, :] = -1 * distance[:, 1, :]
        distance_min = torch.where(distance[:, 0, :] < distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])
        distance_max = torch.where(distance[:, 0, :] > distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])

        centerness = distance_min / distance_max
        centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
        centerness = torch.clamp(centerness, min=1e-6)
        centerness = torch.pow(centerness, 1/3)

        centerness_mask = pos_mask.new_zeros(pos_mask.shape).float()
        centerness_mask[pos_mask] = centerness
        return centerness_mask
    

    def generate_sa_center_ness_mask(self):
        sa_pos_mask = self.forward_ret_dict['sa_ins_labels']
        sa_gt_boxes = self.forward_ret_dict['sa_gt_box_of_fg_points']
        sa_xyz_coords = self.forward_ret_dict['sa_xyz_coords']
        sa_centerness_mask = []
        for i in range(len(sa_pos_mask)):
            pos_mask = sa_pos_mask[i] > 0
            gt_boxes = sa_gt_boxes[i]
            xyz_coords = sa_xyz_coords[i].view(-1,sa_xyz_coords[i].shape[-1])[:,1:]
            xyz_coords = xyz_coords[pos_mask].clone().detach()
            offset_xyz = xyz_coords[:, 0:3] - gt_boxes[:, 0:3]
            offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)

            template = gt_boxes.new_tensor(([1, 1, 1], [-1, -1, -1])) / 2
            margin = gt_boxes[:, None, 3:6].repeat(1, 2, 1) * template[None, :, :]
            distance = margin - offset_xyz_canical[:, None, :].repeat(1, 2, 1)
            distance[:, 1, :] = -1 * distance[:, 1, :]
            distance_min = torch.where(distance[:, 0, :] < distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])
            distance_max = torch.where(distance[:, 0, :] > distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])

            centerness = distance_min / distance_max
            centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
            centerness = torch.clamp(centerness, min=1e-6)
            centerness = torch.pow(centerness, 1/3)

            centerness_mask = pos_mask.new_zeros(pos_mask.shape).float()
            centerness_mask[pos_mask] = centerness

            sa_centerness_mask.append(centerness_mask)
        return sa_centerness_mask


    def get_center_box_binori_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['center_box_labels']
        point_box_preds = self.forward_ret_dict['center_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        pred_box_xyzwhl = point_box_preds[:, :6]
        label_box_xyzwhl = point_box_labels[:, :6]

        point_loss_box_src = self.reg_loss_func(
            pred_box_xyzwhl[None, ...], label_box_xyzwhl[None, ...], weights=reg_weights[None, ...]
        )
        point_loss_xyzwhl = point_loss_box_src.sum()

        pred_ori_bin_id = point_box_preds[:, 6:6+self.box_coder.bin_size]
        pred_ori_bin_res = point_box_preds[:, 6+self.box_coder.bin_size:]

        label_ori_bin_id = point_box_labels[:, 6]
        label_ori_bin_res = point_box_labels[:, 7]
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss_ori_cls = criterion(pred_ori_bin_id.contiguous(), label_ori_bin_id.long().contiguous())
        loss_ori_cls = torch.sum(loss_ori_cls * reg_weights)

        label_id_one_hot = F.one_hot(label_ori_bin_id.long().contiguous(), self.box_coder.bin_size)
        pred_ori_bin_res = torch.sum(pred_ori_bin_res * label_id_one_hot.float(), dim=-1)
        loss_ori_reg = F.smooth_l1_loss(pred_ori_bin_res, label_ori_bin_res)
        loss_ori_reg = torch.sum(loss_ori_reg * reg_weights)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        loss_ori_cls = loss_ori_cls * loss_weights_dict.get('dir_weight', 1.0)
        point_loss_box = point_loss_xyzwhl + loss_ori_reg + loss_ori_cls
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_box': point_loss_box.item()})
        tb_dict.update({'center_loss_box_xyzwhl': point_loss_xyzwhl.item()})
        tb_dict.update({'center_loss_box_ori_bin': loss_ori_cls.item()})
        tb_dict.update({'center_loss_box_ori_res': loss_ori_reg.item()})
        return point_loss_box, tb_dict


    def get_center_box_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['center_box_labels']
        point_box_preds = self.forward_ret_dict['center_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.reg_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        )
        point_loss = point_loss_box_src.sum()

        point_loss_box = point_loss
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict


    def get_corner_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['center_gt_box_of_fg_points']
        pred_boxes = self.forward_ret_dict['point_box_preds']
        pred_boxes = pred_boxes[pos_mask]
        loss_corner = loss_utils.get_corner_loss_lidar(
            pred_boxes[:, 0:7],
            gt_boxes[:, 0:7]
        )
        loss_corner = loss_corner.mean()
        loss_corner = loss_corner * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['corner_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'corner_loss_reg': loss_corner.item()})
        return loss_corner, tb_dict


    def get_iou3d_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['center_gt_box_of_fg_points']
        pred_boxes = self.forward_ret_dict['point_box_preds'].clone().detach()
        pred_boxes = pred_boxes[pos_mask]
        iou3d_targets, _ = loss_utils.generate_iou3d(pred_boxes[:, 0:7], gt_boxes[:, 0:7])

        iou3d_preds = self.forward_ret_dict['box_iou3d_preds'].squeeze(-1)
        iou3d_preds = iou3d_preds[pos_mask]

        loss_iou3d = F.smooth_l1_loss(iou3d_preds, iou3d_targets)

        loss_iou3d = loss_iou3d * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou3d_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'iou3d_loss_reg': loss_iou3d.item()})
        return loss_iou3d, tb_dict


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                centers_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                centers: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                encoder_xyz: List of points_coords in SA
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                batch_cls_preds: (N1 + N2 + N3 + ..., num_class)
                point_box_preds: (N1 + N2 + N3 + ..., 7)
        """

        center_features = batch_dict['centers_features']
        center_coords = batch_dict['centers']
        center_cls_preds = self.cls_center_layers(center_features)  # (total_centers, num_class)
        center_box_preds = self.box_center_layers(center_features)  # (total_centers, box_code_size)
        box_iou3d_preds = self.box_iou3d_layers(center_features) if self.box_iou3d_layers is not None else None

        ret_dict = {'center_cls_preds': center_cls_preds,
                    'center_box_preds': center_box_preds,
                    'ctr_offsets': batch_dict['ctr_offsets'],
                    'centers': batch_dict['centers'],
                    'centers_origin': batch_dict['centers_origin'],
                    'sa_ins_preds': batch_dict['sa_ins_preds'],
                    'box_iou3d_preds': box_iou3d_preds,
                    }
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training or \
                self.model_cfg.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION or \
                self.model_cfg.LOSS_CONFIG.CENTERNESS_REGULARIZATION or \
                self.model_cfg.LOSS_CONFIG.IOU3D_REGULARIZATION:

            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                    points=center_coords[:, 1:4],
                    point_cls_preds=center_cls_preds, point_box_preds=center_box_preds
                )

            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['box_iou3d_preds'] = box_iou3d_preds
            batch_dict['batch_index'] = center_coords[:,0]
            batch_dict['cls_preds_normalized'] = False

            ret_dict['point_box_preds'] = point_box_preds

        self.forward_ret_dict = ret_dict

        return batch_dict
