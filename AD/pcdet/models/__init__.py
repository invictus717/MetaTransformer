from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector, build_detector_multi_db, build_detector_multi_db_3

try:
    import kornia
except:
    pass
    # print('Warning: kornia is not installed. This package is only required by CaDDN')


def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model

# def build_network_multi_db_v2(model_cfg, num_class, dataset):
#     model = build_detector_multi_db_v2(
#         model_cfg=model_cfg, num_class=num_class, dataset=dataset
#     )
#     return model

def build_network_multi_db(model_cfg, num_class, num_class_s2, dataset, dataset_s2, source_one_name):
    model = build_detector_multi_db(
        model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, 
        dataset=dataset, dataset_s2=dataset_s2, source_one_name=source_one_name
    )
    return model

def build_network_multi_db_3(model_cfg, num_class, num_class_s2, num_class_s3, dataset, dataset_s2, dataset_s3, source_one_name, source_1):
    model = build_detector_multi_db_3(
        model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, num_class_s3=num_class_s3,
        dataset=dataset, dataset_s2=dataset_s2, dataset_s3=dataset_s3, source_one_name=source_one_name,
        source_1=source_1
    )
    return model

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        elif key in ['db_flag']:
            continue
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict, **forward_args):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict, **forward_args)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
