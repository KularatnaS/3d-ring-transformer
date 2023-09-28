import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class RingEmbedding(nn.Module):
    def __init__(self, d_ring_embedding: int, n_point_features: int, n_extracted_point_features: int,
                 not_testing_padding: bool = True):
        super().__init__()

        self.d_ring_embedding = d_ring_embedding
        self.n_point_features = n_point_features

        self.conv1 = torch.nn.Conv1d(n_point_features, n_point_features, 1, bias=not_testing_padding)
        self.conv2 = torch.nn.Conv1d(n_point_features, n_extracted_point_features, 1, bias=not_testing_padding)
        self.conv3 = torch.nn.Conv1d(n_extracted_point_features, n_extracted_point_features, 1, bias=not_testing_padding)
        self.conv4 = torch.nn.Conv1d(n_extracted_point_features, 128, 1, bias=not_testing_padding)
        self.conv5 = torch.nn.Conv1d(128, d_ring_embedding, 1, bias=not_testing_padding)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(n_point_features, affine=not_testing_padding)
        self.bn2 = nn.BatchNorm1d(n_extracted_point_features, affine=not_testing_padding)
        self.bn3 = nn.BatchNorm1d(n_extracted_point_features, affine=not_testing_padding)
        self.bn4 = nn.BatchNorm1d(128, affine=not_testing_padding)
        self.bn5 = nn.BatchNorm1d(d_ring_embedding, affine=not_testing_padding)

    def forward(self, x):
        # x -> [batch, n_rings, n_points_per_ring, n_point_features]
        assert x.shape[-1] == self.n_point_features
        batch_size = x.shape[0]
        n_rings_per_bubble = x.shape[1]

        x = rearrange(x, 'batch a b c -> (batch a) c b')    # [batch * n_rings, n_point_features, n_points_per_ring]
        x = self.conv1(x)                                   # [batch * n_rings, n_point_features, n_points_per_ring]
        x = F.relu(self.bn1(x))                             # [batch * n_rings, n_points_per_ring, n_points_per_ring]

        x = self.conv2(x)                                   # [batch * n_rings, n_extracted_point_features, n_points_per_ring]
        x = F.relu(self.bn2(x))                             # [batch * n_rings, n_extracted_point_features, n_points_per_ring, ]

        x = self.conv3(x)                                   # [batch * n_rings, n_extracted_point_features, n_points_per_ring]
        per_point_embedded_features = rearrange(x, 'a b c -> a c b')  # [batch * n_rings, n_points_per_ring, n_extracted_point_features]
        x = F.relu(self.bn3(x))                             # [batch * n_rings, n_points_per_ring, n_extracted_point_features]

        x = self.conv4(x)                                   # [batch * n_rings, 128, n_points_per_ring]
        x = F.relu(self.bn4(x))                             # [batch * n_rings, 128, n_points_per_ring]

        x = self.conv5(x)                                   # [batch * n_rings, d_ring_embedding, n_points_per_ring]
        x = F.relu(self.bn5(x))                             # [batch * n_rings, d_ring_embedding, n_points_per_ring]

        x = torch.max(x, 2, keepdim=True)[0]                # [batch * n_rings, d_ring_embedding, 1]
        x = rearrange(x, 'a b 1 -> a b')                    # [batch * n_rings, d_ring_embedding]
        x = rearrange(x, '(batch n_rings) b -> batch n_rings b', batch=batch_size, n_rings=n_rings_per_bubble)  # [batch, n_rings, d_ring_embedding]

        return x, rearrange(per_point_embedded_features, '(batch n_rings) n d -> batch n_rings n d', batch=batch_size, n_rings=n_rings_per_bubble)