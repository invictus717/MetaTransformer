import torch
from .detector3d_template import Detector3DTemplate
from .detector3d_template_multi_db import Detector3DTemplate_M_DB
from .detector3d_template_multi_db_3 import Detector3DTemplate_M_DB_3
from pcdet.utils import common_utils

class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
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
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict

class SemiPVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.model_type = None

    def set_model_type(self, model_type):
        assert model_type in ['origin', 'teacher', 'student']
        self.model_type = model_type
        self.dense_head.model_type = model_type
        self.point_head.model_type = model_type
        self.roi_head.model_type = model_type

    def forward(self, batch_dict):
        # origin: (training, return loss) (testing, return final boxes)
        if self.model_type == 'origin':
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
        
        # teacher: (testing, return raw boxes)
        elif self.model_type == 'teacher':
            # assert not self.training
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)
            return batch_dict

        # student:
        elif self.model_type == 'student':
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)
            if self.training:
                if 'gt_boxes' in batch_dict: # for (pseudo-)labeled data
                    loss, tb_dict, disp_dict = self.get_training_loss()
                    ret_dict = {
                        'loss': loss
                    }
                    return batch_dict, ret_dict, tb_dict, disp_dict
                else:
                    return batch_dict
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts
        else:
            raise Exception('Unsupprted model type')

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d
            
        return loss, tb_dict, disp_dict

class PVRCNN_M_DB(Detector3DTemplate_M_DB):
    def __init__(self, model_cfg, num_class, num_class_s2, dataset, dataset_s2, source_one_name):
        super().__init__(model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, dataset=dataset,
                        dataset_s2=dataset_s2, source_one_name=source_one_name)
        self.module_list = self.build_networks()
        self.source_one_name = source_one_name

    def forward(self, batch_dict):
        
        # Split the Concat dataset batch into batch_1 and batch_2
        split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, batch_dict)

        batch_s1 = {}
        batch_s2 = {}

        len_of_module = len(self.module_list)
        for k, cur_module in enumerate(self.module_list):
            if k < len_of_module-6:
                batch_dict = cur_module(batch_dict)
            
            if k == len_of_module-6 or k == len_of_module-5 or k == len_of_module-4:
                if len(split_tag_s1) == batch_dict['batch_size']:
                    batch_dict = cur_module(batch_dict)
                elif len(split_tag_s2) == batch_dict['batch_size']:
                    continue
                else:
                    if k == len_of_module-6:
                        batch_s1, batch_s2 = common_utils.split_two_batch_dict_gpu(split_tag_s1, split_tag_s2, batch_dict)
                    batch_s1 = cur_module(batch_s1)

            if k == len_of_module-3 or k == len_of_module-2 or k == len_of_module-1:             
                if len(split_tag_s2) == batch_dict['batch_size']:
                    batch_dict = cur_module(batch_dict)
                elif len(split_tag_s1) == batch_dict['batch_size']:
                    continue
                else:
                    batch_s2 = cur_module(batch_s2)

        if self.training:
            split_tag_s1, split_tag_s2 = common_utils.split_batch_dict(self.source_one_name, batch_dict)
            if len(split_tag_s1) == batch_dict['batch_size']:
                loss, tb_dict, disp_dict = self.get_training_loss_s1()
            
                ret_dict = {
                    'loss': loss
                }
                return ret_dict, tb_dict, disp_dict
            elif len(split_tag_s2) == batch_dict['batch_size']:
                loss, tb_dict, disp_dict = self.get_training_loss_s2()
            
                ret_dict = {
                    'loss': loss
                }
                return ret_dict, tb_dict, disp_dict
            else:
                loss_1, tb_dict_1, disp_dict_1 = self.get_training_loss_s1()
                loss_2, tb_dict_2, disp_dict_2 = self.get_training_loss_s2()
                ret_dict = {
                    'loss': loss_1 + loss_2
                }
                return ret_dict, tb_dict_1, disp_dict_1

        else:
            # NOTE: When peform the inference, only one dataset can be accessed.
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss_s1(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head_s1.get_loss()
        loss_point, tb_dict = self.point_head_s1.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head_s1.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def get_training_loss_s2(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head_s2.get_loss()
        loss_point, tb_dict = self.point_head_s2.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head_s2.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

class PVRCNN_M_DB_3(Detector3DTemplate_M_DB_3):
    def __init__(self, model_cfg, num_class, num_class_s2, num_class_s3, dataset, dataset_s2, dataset_s3, source_one_name, source_1):
        super().__init__(model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, num_class_s3=num_class_s3, 
                        dataset=dataset, dataset_s2=dataset_s2, dataset_s3=dataset_s3, source_one_name=source_one_name, source_1=source_1)
        self.module_list = self.build_networks()
        self.source_one_name = source_one_name
        self.source_1 = source_1

    def forward(self, batch_dict):
        batch_s1 = {}
        batch_s2 = {}
        batch_s3 = {}

        if self.training:
            len_of_module = len(self.module_list)
            for k, cur_module in enumerate(self.module_list):
                if k < len_of_module-9:
                    batch_dict = cur_module(batch_dict)
                
                if k == len_of_module-9 or k == len_of_module-8 or k == len_of_module-7:
                    if k == len_of_module-9:
                        # Split the Concat dataset batch into batch_1, batch_2, and batch_3
                        split_tag_s1, split_tag_s2_pre = common_utils.split_batch_dict('waymo', batch_dict)
                        batch_s1, batch_s2_pre = common_utils.split_two_batch_dict_gpu(split_tag_s1, split_tag_s2_pre, batch_dict)
                        split_tag_s2, split_tag_s3 = common_utils.split_batch_dict(self.source_one_name, batch_s2_pre)
                        batch_s2, batch_s3 = common_utils.split_two_batch_dict_gpu(split_tag_s2, split_tag_s3, batch_s2_pre)
                    batch_s1 = cur_module(batch_s1)

                if k == len_of_module-6 or k == len_of_module-5 or k == len_of_module-4:
                    batch_s2 = cur_module(batch_s2)

                if k == len_of_module-3 or k == len_of_module-2 or k == len_of_module-1:
                    batch_s3 = cur_module(batch_s3)
        else:
            len_of_module = len(self.module_list)
            for k, cur_module in enumerate(self.module_list):
                if k < len_of_module-9:
                    batch_dict = cur_module(batch_dict)
                
                if k == len_of_module-9 or k == len_of_module-8 or k == len_of_module-7:
                    if self.source_1 == 1:
                        batch_dict = cur_module(batch_dict)
                    else:
                        continue
                if k == len_of_module-6 or k == len_of_module-5 or k == len_of_module-4:
                    if self.source_1 == 2:         
                        batch_dict = cur_module(batch_dict)
                    else:
                        continue

                if k == len_of_module-3 or k == len_of_module-2 or k == len_of_module-1:
                    if self.source_1 == 3:  
                        batch_dict = cur_module(batch_dict)
                    else:
                        continue

        if self.training:
            loss_1, tb_dict_1, disp_dict_1 = self.get_training_loss_s1()
            loss_2, tb_dict_2, disp_dict_2 = self.get_training_loss_s2()
            loss_3, tb_dict_3, disp_dict_3 = self.get_training_loss_s3()
            ret_dict = {
                'loss': loss_1 + loss_2 + loss_3
            }
            return ret_dict, tb_dict_1, disp_dict_1

        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss_s1(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head_s1.get_loss()
        loss_point, tb_dict = self.point_head_s1.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head_s1.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def get_training_loss_s2(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head_s2.get_loss()
        loss_point, tb_dict = self.point_head_s2.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head_s2.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
    
    def get_training_loss_s3(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head_s3.get_loss()
        loss_point, tb_dict = self.point_head_s3.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head_s3.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict