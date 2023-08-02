import copy
import numpy as np
import torch
import torch.nn as nn

from ....utils import uni3d_norm_2_in

class POINT_T(nn.Module):
    def __init__(self, model_cfg, **kwargs):
            super().__init__()
            self.model_cfg = model_cfg

            # using the domain-specific norm
            self.scale_bn = uni3d_norm_2_in.UniNorm1d(self.model_cfg.SHARED_CONV_CHANNEL,
                                        dataset_from_flag=int(self.model_cfg.db_source),
                                        eps=1e-3, momentum=0.01, voxel_coord=True)
            #self.scale_bn = nn.BatchNorm1d(self.model_cfg.SHARED_CONV_CHANNEL)
            

    # ---update the xyz coord---
    # def forward(self, data_dict):
    #     points = data_dict['points']
    #     points_idx = points[:,0].unsqueeze(1)
    #     points_coord = points[:,1:4]
    #     points_rescaled = self.scale_bn(points_coord)

    #     points = torch.cat([points_idx, points_rescaled], dim=1)

    #     data_dict['points'] = points
    #     return data_dict
    
    # ---only update the z coord---
    def forward(self, data_dict):
        points = data_dict['points']
        points_others = points[:,0:3]
        points_coord_z = points[:,3].unsqueeze(1)
        points_rescaled = self.scale_bn(points_coord_z, points[:,0].unsqueeze(1))

        points = torch.cat([points_others, points_rescaled], dim=1)

        data_dict['points'] = points
        return data_dict