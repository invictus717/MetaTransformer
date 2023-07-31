import torch
import emd_cuda


class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2




class earth_mover_distance(torch.nn.Module):
    f''' emd
    '''
    def __init__(self):
        super().__init__()

    def forward(self, xyz1, xyz2, transpose=False):
        """Earth Mover Distance (Approx)

        Args:
            xyz1 (torch.Tensor): (b, n1, 3)
            xyz2 (torch.Tensor): (b, n2, 3)
            transpose (bool): whether to transpose inputs as it might be BCN format.
                Extensions only support BNC format.

        Returns:
            cost (torch.Tensor): (b)

        """

        cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)
        cost = cost / xyz1.size(1)
        
        return cost.mean()
# def earth_mover_distance(xyz1, xyz2, transpose=True):
#     """Earth Mover Distance (Approx)

#     Args:
#         xyz1 (torch.Tensor): (b, 3, n1)
#         xyz2 (torch.Tensor): (b, 3, n1)
#         transpose (bool): whether to transpose inputs as it might be BCN format.
#             Extensions only support BNC format.

#     Returns:
#         cost (torch.Tensor): (b)

#     """
#     if xyz1.dim() == 2:
#         xyz1 = xyz1.unsqueeze(0)
#     if xyz2.dim() == 2:
#         xyz2 = xyz2.unsqueeze(0)
#     if transpose:
#         xyz1 = xyz1.transpose(1, 2)
#         xyz2 = xyz2.transpose(1, 2)
#     cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)
#     return cost

