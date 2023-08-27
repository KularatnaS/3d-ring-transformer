import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange

import logging
LOGGER = logging.getLogger(__name__)


class RingEmbedding(nn.Module):
    def __init__(self, d_ring_embedding: int, n_point_features: int, not_testing_padding: bool = True):
        super().__init__()
        self.d_ring_embedding = d_ring_embedding
        self.n_point_features = n_point_features

        self.conv1 = torch.nn.Conv1d(n_point_features, 64, 1, bias=not_testing_padding)
        self.conv2 = torch.nn.Conv1d(64, 128, 1, bias=not_testing_padding)
        self.conv3 = torch.nn.Conv1d(128, self.d_ring_embedding, 1, bias=not_testing_padding)
        self.relu = nn.ReLU()
        self.ln1 = nn.LayerNorm(64, elementwise_affine=not_testing_padding)
        self.ln2 = nn.LayerNorm(128, elementwise_affine=not_testing_padding)
        self.ln3 = nn.LayerNorm(self.d_ring_embedding, elementwise_affine=not_testing_padding)

    def forward(self, x):
        # x -> [batch, n_rings, n_points_per_ring, n_point_features]
        assert x.shape[-1] == self.n_point_features
        batch_size = x.shape[0]
        n_rings_per_bubble = x.shape[1]

        x = rearrange(x, 'batch a b c -> (batch a) c b')    # [batch * n_rings, n_point_features, n_points_per_ring]
        x = self.conv1(x)                                   # [batch * n_rings, 64, n_points_per_ring]
        x = rearrange(x, 'a b c -> a c b')                  # [batch * n_rings, n_points_per_ring, 64]
        x = F.relu(self.ln1(x))                             # [batch * n_rings, n_points_per_ring, 64]

        x = rearrange(x, 'a b c -> a c b')                  # [batch * n_rings, 64, n_points_per_ring]
        x = self.conv2(x)                                   # [batch * n_rings, 128, n_points_per_ring]
        x = rearrange(x, 'a b c -> a c b')                  # [batch * n_rings, n_points_per_ring, 128]
        x = F.relu(self.ln2(x))                             # [batch * n_rings, n_points_per_ring, 128]

        x = rearrange(x, 'a b c -> a c b')                  # [batch * n_rings, 128, n_points_per_ring]
        x = self.conv3(x)                                   # [batch * n_rings, d_ring_embedding, n_points_per_ring]
        x = rearrange(x, 'a b c -> a c b')                  # [batch * n_rings, n_points_per_ring, d_ring_embedding]
        x = F.relu(self.ln3(x))                             # [batch * n_rings, n_points_per_ring, d_ring_embedding]

        x = rearrange(x, 'a b c -> a c b')                  # [batch * n_rings, d_ring_embedding, n_points_per_ring]
        x = torch.max(x, 2, keepdim=True)[0]                # [batch * n_rings, d_ring_embedding, 1]
        x = rearrange(x, 'a b 1 -> a b')                    # [batch * n_rings, d_ring_embedding]
        x = rearrange(x, '(batch n_rings) b -> batch n_rings b', batch=batch_size, n_rings=n_rings_per_bubble)  # [batch, n_rings, d_ring_embedding]

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_ring_embedding: int, rings_per_bubble: int, dropout: float) -> None:
        super().__init__()
        self.d_ring_embedding = d_ring_embedding
        self.rings_per_bubble = rings_per_bubble
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (rings_per_bubble, d_ring_embedding)
        pe = torch.zeros(rings_per_bubble, d_ring_embedding)
        # Create a vector of shape (rings_per_bubble, 1)
        position = torch.arange(0, rings_per_bubble, dtype=torch.float).unsqueeze(1)
        # Create a vector of shape (d_ring_embedding)
        div_term = torch.exp(torch.arange(0, d_ring_embedding, 2).float() * (-math.log(10000.0) / d_ring_embedding)) # (d_ring_embedding / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_ring_embedding))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_ring_embedding))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, rings_per_bubble, d_ring_embedding)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (batch, rings_per_bubble, d_ring_embedding)
        return self.dropout(x)

