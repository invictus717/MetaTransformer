import torch
from .semi_utils import reverse_transform, load_data_to_gpu, construct_pseudo_label

def pseudo_label(teacher_model, student_model,
                  ld_teacher_batch_dict, ld_student_batch_dict,
                  ud_teacher_batch_dict, ud_student_batch_dict,
                  cfgs, epoch_id, dist
                 ):
    assert ld_teacher_batch_dict is None # Only generate labels for unlabeled data

    load_data_to_gpu(ld_student_batch_dict)
    load_data_to_gpu(ud_student_batch_dict)
    load_data_to_gpu(ud_teacher_batch_dict)

    if not dist:
        ud_teacher_batch_dict = teacher_model(ud_teacher_batch_dict)
        teacher_boxes, _ = teacher_model.post_processing(ud_teacher_batch_dict)
    else:
        _, ud_teacher_batch_dict = teacher_model(ld_teacher_batch_dict, ud_teacher_batch_dict)
        teacher_boxes, _ = teacher_model.module.onepass.post_processing(ud_teacher_batch_dict)

    teacher_boxes = reverse_transform(teacher_boxes, ud_teacher_batch_dict, ud_student_batch_dict)
    
    if cfgs.get('FILTER_BY_SCORE_THRESHOLD', False):
        pred_dicts = []
        for index in range(ud_teacher_batch_dict['batch_size']):
            selected = teacher_boxes[index]['pred_scores'] > cfgs.SCORE_THRESHOLD
            pred_boxes = teacher_boxes[index]['pred_boxes'][selected]
            pred_scores = teacher_boxes[index]['pred_scores'][selected]
            pred_labels = teacher_boxes[index]['pred_labels'][selected]

            record_dict = {
            'pred_boxes': pred_boxes,
            'pred_scores': pred_scores,
            'pred_labels': pred_labels
            }
            pred_dicts.append(record_dict)
        gt_boxes = construct_pseudo_label(pred_dicts)
    else:
        gt_boxes = construct_pseudo_label(teacher_boxes)
    
    ud_student_batch_dict['gt_boxes'] = gt_boxes

    if not dist:
        _, ld_ret_dict, _, _ = student_model(ld_student_batch_dict)
        _, ud_ret_dict, tb_dict, disp_dict = student_model(ud_student_batch_dict)
    else:
        (_, ld_ret_dict, _, _), (_, ud_ret_dict, tb_dict, disp_dict) = student_model(ld_student_batch_dict, ud_student_batch_dict)

    loss = ld_ret_dict['loss'].mean() + ud_ret_dict['loss'].mean()

    return loss, tb_dict, disp_dict