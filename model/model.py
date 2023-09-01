import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange

import logging
LOGGER = logging.getLogger(__name__)


class RingEmbedding(nn.Module):
    def __init__(self, d_ring_embedding: int, n_point_features: int, point_features_div: int,
                 not_testing_padding: bool = True):
        super().__init__()
        layer2_div = int(point_features_div * 0.5)
        assert d_ring_embedding % point_features_div == 0
        assert d_ring_embedding % layer2_div == 0
        layer_1_features = int(d_ring_embedding/point_features_div)
        assert layer_1_features == 64
        layer_2_features = int(d_ring_embedding/layer2_div)
        assert layer_2_features == 128

        self.d_ring_embedding = d_ring_embedding
        self.n_point_features = n_point_features

        self.conv1 = torch.nn.Conv1d(n_point_features, layer_1_features, 1, bias=not_testing_padding)
        self.conv2 = torch.nn.Conv1d(layer_1_features, layer_2_features, 1, bias=not_testing_padding)
        self.conv3 = torch.nn.Conv1d(layer_2_features, self.d_ring_embedding, 1, bias=not_testing_padding)
        self.relu = nn.ReLU()
        self.ln1 = nn.LayerNorm(layer_1_features, elementwise_affine=not_testing_padding)
        self.ln2 = nn.LayerNorm(layer_2_features, elementwise_affine=not_testing_padding)
        self.ln3 = nn.LayerNorm(self.d_ring_embedding, elementwise_affine=not_testing_padding)

    def forward(self, x):
        # x -> [batch, n_rings, n_points_per_ring, n_point_features]
        assert x.shape[-1] == self.n_point_features
        batch_size = x.shape[0]
        n_rings_per_bubble = x.shape[1]

        x = rearrange(x, 'batch a b c -> (batch a) c b')    # [batch * n_rings, n_point_features, n_points_per_ring]
        x = self.conv1(x)                                   # [batch * n_rings, 64, n_points_per_ring]
        per_point_embedded_features = rearrange(x, 'a b c -> a c b') # [batch * n_rings, n_points_per_ring, 64]
        x = F.relu(self.ln1(per_point_embedded_features))            # [batch * n_rings, n_points_per_ring, 64]

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

        return x, rearrange(per_point_embedded_features, '(batch n_rings) n d -> batch n_rings n d', batch=batch_size, n_rings=n_rings_per_bubble)


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


class FeedForwardBlock(nn.Module):

    def __init__(self, d_ring_embedding: int, dropout: float) -> None:
        super().__init__()
        self.d_ff = d_ring_embedding * 4
        self.linear_1 = nn.Linear(d_ring_embedding, self.d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(self.d_ff, d_ring_embedding)  # w2 and b2

    def forward(self, x):
        # (batch, rings_per_bubble, d_ring_embedding) --> (batch, rings_per_bubble, d_ff) -->
        # (batch, rings_per_bubble, d_ring_embedding)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_ring_embedding: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_ring_embedding = d_ring_embedding  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_ring_embedding is divisible by h
        assert d_ring_embedding % h == 0, "d_ring_embedding is not divisible by h"

        self.d_k = d_ring_embedding // h  # Dimension of vector seen by each head

        self.w_q = nn.Linear(d_ring_embedding, d_ring_embedding, bias=False)  # Wq
        self.w_k = nn.Linear(d_ring_embedding, d_ring_embedding, bias=False)  # Wk
        self.w_v = nn.Linear(d_ring_embedding, d_ring_embedding, bias=False)  # Wv

        self.w_o = nn.Linear(d_ring_embedding, d_ring_embedding, bias=False)  # Wo

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, rings_per_bubble, d_k) --> (batch, h, rings_per_bubble, rings_per_bubble)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, rings_per_bubble, rings_per_bubble)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, rings_per_bubble, rings_per_bubble) --> (batch, h, rings_per_bubble, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, rings_per_bubble, d_ring_embedding) --> (batch, rings_per_bubble, d_ring_embedding)
        key = self.w_k(k)  # (batch, rings_per_bubble, d_ring_embedding) --> (batch, rings_per_bubble, d_ring_embedding)
        value = self.w_v(v)  # (batch, rings_per_bubble, d_ring_embedding) --> (batch, rings_per_bubble, d_ring_embedding)

        query = rearrange(query, 'batch rings_per_bubble (h d_k) -> batch h rings_per_bubble d_k', h=self.h, d_k=self.d_k)
        key = rearrange(key, 'batch rings_per_bubble (h d_k) -> batch h rings_per_bubble d_k', h=self.h, d_k=self.d_k)
        value = rearrange(value, 'batch rings_per_bubble (h d_k) -> batch h rings_per_bubble d_k', h=self.h, d_k=self.d_k)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        x = rearrange(x, 'batch h rings_per_bubble d_k -> batch rings_per_bubble (h d_k)')

        # Multiply by Wo
        # (batch, rings_per_bubble, d_ring_embedding) --> (batch, rings_per_bubble, d_ring_embedding)
        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, d_ring_embedding: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_ring_embedding)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float, d_ring_embedding: int) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.d_ring_embedding = d_ring_embedding
        self.residual_connections = nn.ModuleList([ResidualConnection(d_ring_embedding, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList, d_ring_embedding: int) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(d_ring_embedding)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ClassificationLayer(nn.Module):

    def __init__(self, d_ring_embedding: int, point_features_div: int, n_classes_model: int) -> None:
        super().__init__()
        assert d_ring_embedding % point_features_div == 0
        self.linear = nn.Linear(int(d_ring_embedding/point_features_div) + d_ring_embedding, n_classes_model)

    @staticmethod
    def append_ring_embedding_to_per_point_embedded_features(x, per_point_embedded_features):
        # x -> [batch, n_rings, d_ring_embedding]
        # per_point_embedded_features -> [batch, n_rings, n_points_per_ring, d_ring_embedding/4]
        # returns -> [batch, n_rings, n_points_per_ring, d_ring_embedding/4 + d_ring_embedding]
        return torch.cat((per_point_embedded_features,
                          x.unsqueeze(2).repeat(1, 1, per_point_embedded_features.shape[2], 1)), dim=-1)

    def forward(self, x, per_point_embedded_features):
        x = ClassificationLayer.append_ring_embedding_to_per_point_embedded_features(x, per_point_embedded_features)
        return self.linear(x)  # [batch, n_rings, n_points_per_ring, n_classes_model]


class RingTransformerClassification(nn.Module):

    def __init__(self, src_embed: RingEmbedding, src_pos: PositionalEncoding, encoder: Encoder,
                 classification_layer: ClassificationLayer) -> None:
        super().__init__()
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.encoder = encoder
        self.classification_layer = classification_layer

    def forward(self, src, src_mask):
        # [batch, n_rings, d_ring_embedding]
        src, per_point_embedded_features = self.src_embed(src)
        src = self.src_pos(src)
        src = self.encoder(src, src_mask)
        return self.classification_layer(src, per_point_embedded_features)  # [batch, n_rings, n_points_per_ring, n_classes_model]


def build_classification_model(d_ring_embedding: int, n_point_features: int, point_features_div: int,
                               rings_per_bubble: int, dropout: float, n_encoder_blocks: int, heads: int,
                               n_classes_model: int):

    # Embedding layer
    src_embed = RingEmbedding(d_ring_embedding=d_ring_embedding, n_point_features=n_point_features,
                              point_features_div=point_features_div)

    # Positional encoding layer
    src_pos = PositionalEncoding(d_ring_embedding=d_ring_embedding, rings_per_bubble=rings_per_bubble, dropout=dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(n_encoder_blocks):
        self_attention_block = MultiHeadAttentionBlock(d_ring_embedding=d_ring_embedding, h=heads, dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_ring_embedding=d_ring_embedding, dropout=dropout)
        encoder_block = EncoderBlock(self_attention_block=self_attention_block, feed_forward_block=feed_forward_block,
                                     dropout=dropout, d_ring_embedding=d_ring_embedding)
        encoder_blocks.append(encoder_block)

    # Create the encoder
    encoder = Encoder(layers=nn.ModuleList(encoder_blocks), d_ring_embedding=d_ring_embedding)

    # Create the classification layer
    classification_layer = ClassificationLayer(d_ring_embedding=d_ring_embedding, point_features_div=point_features_div,
                                               n_classes_model=n_classes_model)

    classification_model = RingTransformerClassification(src_embed=src_embed, src_pos=src_pos, encoder=encoder,
                                                         classification_layer=classification_layer)

    # Initialize the weights
    for p in classification_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return classification_model
