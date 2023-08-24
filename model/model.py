import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


import logging
LOGGER = logging.getLogger(__name__)


class RingEmbedding(nn.Module):
    def __init__(self, d_ring_embedding: int, n_point_features: int):
        super().__init__()
        self.d_ring_embedding = d_ring_embedding
        self.n_point_features = n_point_features

        self.conv1 = torch.nn.Conv1d(n_point_features, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, d_ring_embedding, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(d_ring_embedding)

    def forward(self, x):
        # make sure to pass the 0th batch from DataLoader into this function
        # x -> [n_rings, n_points_per_ring, n_point_features]
        assert x.shape[-1] == self.n_point_features
        x = rearrange(x, 'a b c -> a c b')    # [n_rings, n_point_features, n_points_per_ring]
        x = F.relu(self.bn1(self.conv1(x)))   # [n_rings, 64, n_points_per_ring]
        x = F.relu(self.bn2(self.conv2(x)))   # [n_rings, 128, n_points_per_ring]
        x = F.relu(self.bn3(self.conv3(x)))   # [n_rings, d_ring_embedding, n_points_per_ring]
        x = torch.max(x, 2, keepdim=True)[0]  # [n_rings, d_ring_embedding, 1]
        x = rearrange(x, 'a b 1 -> a b')      # [n_rings, d_ring_embedding]

        return x
