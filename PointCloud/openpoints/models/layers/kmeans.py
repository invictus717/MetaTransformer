import torch
from torch import nn
from fast_pytorch_kmeans import KMeans, MultiKMeans
from torch_scatter import scatter
from .local_aggregation import CHANNEL_MAP


class KMeansEmbed(nn.Module):
    """ Point cloud to subsampled groups
    """

    def __init__(self,
                 in_chans=3, 
                 num_groups=256,
                 encoder_dim=256,
                 feature_type='dp', 
                 **kwargs
                 ):
        super().__init__()

        self.num_groups = num_groups

        self.kmeans = MultiKMeans(n_clusters=num_groups, n_kmeans=32, mode='euclidean', verbose=0)

        channels = CHANNEL_MAP[feature_type](in_chans)
        self.feature_type = feature_type
        self.conv1 = nn.Sequential(
            nn.Linear(channels, 128),   #   TODO: here, can be better, edgeconv.
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, encoder_dim)
        )

    def forward(self, xyz, features=None):
        B, N, _ = xyz.shape
        self.kmeans.centroids = None  #   re-init kmeans 
        self.kmeans.n_kmeans = xyz.shape[0]
        labels = self.kmeans.fit_predict(xyz)   # B,N

        # TODO: BUG, sometimes the value is even smaller than the number of centroids!! 
        centroids = self.kmeans.centroids   # B, K, 3
        idx = labels.unsqueeze(-1)

        # p_j = xyz
        p_i = torch.gather(centroids, 1, labels.unsqueeze(-1).expand(-1, -1, 3))   # B, N, 3
        relative_xyz = xyz - p_i  # p_j-p_i, B, N, 3

        if self.feature_type == 'dp':
            neighborhood_features = relative_xyz
        elif self.feature_type == 'pj_dp':
            neighborhood_features = torch.cat([xyz, relative_xyz], -1)
        elif self.feature_type == 'pi_dp':
            neighborhood_features = torch.cat([p_i, relative_xyz], -1)

        neighborhood_features = self.conv1(neighborhood_features)  # B, N, C
        pooled_feat = scatter(neighborhood_features, idx, dim=1, reduce='max')  # B, K, C
        reapted_feat = torch.gather(pooled_feat, 1, idx.expand(-1, -1, pooled_feat.shape[-1]))
        neighborhood_features = torch.cat([reapted_feat, neighborhood_features], dim=-1)
        out_features = scatter(self.conv2(neighborhood_features), idx, dim=1, reduce='max')
        return centroids, out_features, p_i, labels 


if __name__ == "__main__":
    import torch
    import os, sys
    sys.path.insert(0, os.path.join(os.path.abspath(__file__), "../../../../"))
    from openpoints.dataset import ModelNet, vis_points, vis_multi_points
    from openpoints.models.layers import fps

    B, C, N = 8, 3, 8196
    device = 'cuda'

    dataset = ModelNet("data/ModelNet40",
                        N, 40, split='test')
    test_datalodaer = torch.utils.data.DataLoader(dataset, batch_size=B, num_workers=1
                                                 )
    data = iter(test_datalodaer).next()[0]
    points = data['pos'].to(device)

    points = fps(points, N)
    print(points.shape)
    # debug one batch
    K = 12
    print(points.shape, points.device)

    #
    # kmeans_group = KMeansGroup(K).to(device)
    # kmeans_group(points)

    kmeans = KMeans(n_clusters=K, mode='euclidean', verbose=1)
    labels = kmeans.fit_predict(points[0])
    print(labels.shape)
    # 0.5207s for 10000 points, too slow!
    # vis_points(points[0], labels=labels)
    center_points = kmeans.centroids
    print(center_points.shape)
    # vis_multi_points([points.cpu().numpy(), center_points])
    vis_points(points[0], labels=labels)

    # B, N, 3
    # B, N, 1  (label index)

    # debug 8 batch
    # K = 8
    # print(points.shape, points.device)
    # kmeans = MultiKMeans(n_clusters=K, n_kmeans=B,
    #                      mode='euclidean', verbose=1)
    # labels = kmeans.fit_predict(points)
    # print(labels.shape)
    # # 0.5207s for 10000 points, too slow!
    # # vis_multi_points(points.cpu().numpy()[:4], labels=labels.cpu().numpy()[:4])
    # vis_points(points[0], labels=labels[0])
    # vis_points(points[1], labels=labels[1])
