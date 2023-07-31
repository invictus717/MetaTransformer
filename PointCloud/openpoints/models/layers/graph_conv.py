import torch
from torch import nn
from .group import grouping_operation
from .knn import DilatedKNN
from openpoints.models.layers.conv import create_convblock2d


def gather_features(features, indices, sparse_grad=True):
    """Gather the features specified by indices

    Args:
        features: tensor of shape [B, C, N, 1]
        indices: long tensor of shape [B, N, K]
        sparse_grad: whether to use a sparse tensor for the gradient

    Returns:
        gathered_features [B, C, N, K]
    """
    indices = indices.unsqueeze(1).long()
    features, indices = torch.broadcast_tensors(features, indices)
    return features.gather(dim=-2, index=indices, sparse_grad=sparse_grad)


class MRConv(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(MRConv, self).__init__()
        self.nn = create_convblock2d(in_channels*2, out_channels, **kwargs)

    def forward(self, x, edge_index):
        x_j = grouping_operation(x.squeeze(-1), edge_index)
        x_j, _ = torch.max(x_j - x.unsequence(-1), -1, keepdim=True)
        return self.nn(torch.cat([x, x_j], dim=1))


class EdgeConv(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(EdgeConv, self).__init__()
        self.nn = create_convblock2d(in_channels * 2, out_channels, **kwargs)

    def forward(self, x, edge_index):
        # x_j = gather_features(x,edge_index)
        x_j = grouping_operation(x.squeeze(-1), edge_index.int())
        max_value, _ = torch.max(self.nn(torch.cat([x.expand(-1, -1, -1, edge_index.shape[-1]), x_j - x],
                                                   dim=1)), -1, keepdim=True)
        return max_value


_GCN_LAYER_DEFAULT = dict(
    mrconv=MRConv, 
    edgeconv=EdgeConv, 
    edge=EdgeConv
)


class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv=EdgeConv, **kwargs):
        super(GraphConv, self).__init__()
        if isinstance(conv, str):
            conv = _GCN_LAYER_DEFAULT[conv] 
        self.gconv = conv(in_channels, out_channels, **kwargs)

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DynConv(GraphConv):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv=EdgeConv,
                 k=9, dilation=1, stochastic=False, epsilon=0.0, 
                 **kwargs):
        super(DynConv, self).__init__(in_channels, out_channels, conv, **kwargs)
        self.k = k
        self.d = dilation
        self.dilated_knn_graph = DilatedKNN(k, dilation, stochastic, epsilon)

    def forward(self, x):
        edge_index = self.dilated_knn_graph(x.squeeze(-1).transpose(1, 2))
        return super(DynConv, self).forward(x, edge_index)


class ResDynBlock(nn.Module):
    """
    Residual Dynamic graph convolution block
    """
    def __init__(self, in_channels, conv=EdgeConv,
                 k=9, dilation=1, stochastic=False, epsilon=0.0, 
                 **kwargs):
        super(ResDynBlock, self).__init__()
        self.body = DynConv(in_channels, in_channels, conv,
                            k, dilation, stochastic, epsilon, **kwargs)

    def forward(self, x):
        return self.body(x) + x


class DenseDynBlock(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, in_channels, out_channels,conv=EdgeConv,
                 k=9, dilation=1, stochastic=False, epsilon=0.0, 
                 **kwargs):
        super(DenseDynBlock, self).__init__()
        assert out_channels > in_channels, "#out channels should be larger than #in channels"
        self.body = DynConv(in_channels, out_channels-in_channels, conv,
                            k, dilation, stochastic, epsilon, **kwargs)

    def forward(self, x):
        dense = self.body(x).squeeze(-1)
        return torch.cat((x, dense), 1)

