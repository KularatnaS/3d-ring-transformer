import unittest

import numpy as np
import torch
import torch.nn as nn

from model.model import PositionalEncoding, FeedForwardBlock, MultiHeadAttentionBlock, \
    ResidualConnection, EncoderBlock, Encoder, ClassificationLayer, build_classification_model

from model.spconv_unet_ring_embedding import RingEmbedding, batch_to_spconv_tensor


import logging
LOGGER = logging.getLogger(__name__)


class Test_3d_transformer_model(unittest.TestCase):

    def test_build_classification_model(self):
        # GIVEN
        device = torch.device("cuda")
        d_ring_embedding = 256
        n_point_features = 3
        n_extracted_point_features = 1
        batch_size = 2
        rings_per_bubble = 5
        points_per_ring = 100
        dropout = 0.1
        n_encoder_blocks = 6
        heads = 8
        n_classes_model = 4
        model_resolution = 0.01

        # WHEN
        model = build_classification_model(d_ring_embedding=d_ring_embedding, n_point_features=n_point_features,
                                           n_extracted_point_features=n_extracted_point_features, rings_per_bubble=rings_per_bubble,
                                           dropout=dropout, n_encoder_blocks=n_encoder_blocks, heads=heads,
                                           n_classes_model=n_classes_model, model_resolution=model_resolution)
        model.to(device)
        mask = torch.ones(batch_size, 1, 1, rings_per_bubble).to(device)
        x = torch.rand(batch_size, rings_per_bubble, points_per_ring, n_point_features).to(device)
        y = model(x, mask)

        # THEN
        assert y.shape == (batch_size, rings_per_bubble, points_per_ring, n_classes_model)

    def test_classification_layer_append_method(self):
        # GIVEN
        # x -> [batch, n_rings, d_ring_embedding]
        # per_point_embedded_features -> [batch, n_rings, n_points_per_ring, d_ring_embedding/4]
        # returns ->                     [batch, n_rings, n_points_per_ring, d_ring_embedding/4 + d_ring_embedding]
        batch_size = 2
        n_rings = 2
        d_ring_embedding = 4
        n_points_per_ring = 2
        x = torch.tensor([
                            [  # bubble 0
                             [0.1, 0.2, 0.3, 0.4],  # ring 0
                             [1.0, 2.0, 3.0, 4.0]   # ring 1
                            ],
                            [
                             [-0.1, -0.2, 0.0, 1.0],  # ring 0
                             [0.0, 0.0, 0.0, 1.0]   # ring 1
                            ]   # bubble 1
                        ])
        per_point_embedded_features = torch.tensor([
                                                    [  # bubble 0
                                                        [  # ring 0
                                                            [3.0, 4.0],  # point 0
                                                            [0.0, -1.0]   # point 1
                                                        ],
                                                        [  # ring 1
                                                            [-2.0, -3.0],  # point 0
                                                            [5.0, 6.0]   # point 1
                                                        ]
                                                    ],
                                                    [  # bubble 1
                                                        [  # ring 0
                                                            [10.0, 11.0],  # point 0
                                                            [-9.0, 1.0]   # point 1
                                                        ],
                                                        [  # ring 1
                                                            [2.0, 4.5],  # point 0
                                                            [5.5, 9.5]   # point 1
                                                        ]
                                                    ]
                                                ])

        # WHEN
        appended_features = \
            ClassificationLayer.append_ring_embedding_to_per_point_embedded_features(x, per_point_embedded_features)
        expected_appended_features = torch.tensor([
                                                    [  # bubble 0
                                                        [  # ring 0
                                                            [3.0, 4.0, 0.1, 0.2, 0.3, 0.4],  # point 0
                                                            [0.0, -1.0, 0.1, 0.2, 0.3, 0.4]   # point 1
                                                        ],
                                                        [  # ring 1
                                                            [-2.0, -3.0, 1.0, 2.0, 3.0, 4.0],  # point 0
                                                            [5.0, 6.0, 1.0, 2.0, 3.0, 4.0]   # point 1
                                                        ]
                                                    ],
                                                    [  # bubble 1
                                                        [  # ring 0
                                                            [10.0, 11.0, -0.1, -0.2, 0.0, 1.0],  # point 0
                                                            [-9.0, 1.0, -0.1, -0.2, 0.0, 1.0]   # point 1
                                                        ],
                                                        [  # ring 1
                                                            [2.0, 4.5, 0.0, 0.0, 0.0, 1.0],  # point 0
                                                            [5.5, 9.5, 0.0, 0.0, 0.0, 1.0]   # point 1
                                                        ]
                                                    ]
                                                ])

        # THEN
        assert torch.equal(appended_features, expected_appended_features)

    def test_classification_layer(self):
        # GIVEN
        batch_size = 2
        d_ring_embedding = 256
        n_extracted_point_features = 64
        n_classes_model = 4
        rings_per_bubble = 5
        points_per_ring = 10

        # WHEN/THEN
        x = torch.rand(batch_size, rings_per_bubble, d_ring_embedding)
        per_point_embedded_features = torch.rand(batch_size, rings_per_bubble, points_per_ring,
                                                 n_extracted_point_features)
        classification_layer = ClassificationLayer(d_ring_embedding, n_extracted_point_features, n_classes_model)
        y = classification_layer(x, per_point_embedded_features)
        assert y.shape == (batch_size, rings_per_bubble, points_per_ring, n_classes_model)
        # # assert that sum along last dimension of y 1
        # sum_along_last_dim = torch.sum(y, dim=-1)
        # # get values
        # sum_along_last_dim = sum_along_last_dim.detach().numpy()
        # # asser all values are 1
        # assert np.allclose(sum_along_last_dim, np.ones_like(sum_along_last_dim))

    def test_encoder(self):
        # GIVEN
        batch_size = 2
        rings_per_bubble = 4
        d_ring_embedding = 256
        dropout = 0.0
        h = 8
        N = 8
        mask = torch.ones(batch_size, 1, 1, rings_per_bubble)

        # WHEN/THEN
        encoder_blocks = []
        for _ in range(N):
            multi_head_attention_block = MultiHeadAttentionBlock(d_ring_embedding, h, dropout)
            feed_forward_block = FeedForwardBlock(d_ring_embedding, dropout)
            encoder_block = EncoderBlock(multi_head_attention_block, feed_forward_block, dropout, d_ring_embedding)
            encoder_blocks.append(encoder_block)

        encoder = Encoder(nn.ModuleList(encoder_blocks), d_ring_embedding)
        x = torch.rand(batch_size, rings_per_bubble, d_ring_embedding)
        y = encoder(x, mask)

        assert y.shape == x.shape

    def test_encoder_block(self):
        # GIVEN
        batch_size = 2
        rings_per_bubble = 4
        d_ring_embedding = 256
        dropout = 0.0
        h = 8
        mask = torch.ones(batch_size, 1, 1, rings_per_bubble)
        multi_head_attention_block = MultiHeadAttentionBlock(d_ring_embedding, h, dropout)
        feed_forward_block = FeedForwardBlock(d_ring_embedding, dropout)

        # WHEN/THEN
        x = torch.rand(batch_size, rings_per_bubble, d_ring_embedding)
        encoder_block = EncoderBlock(multi_head_attention_block, feed_forward_block, dropout, d_ring_embedding)
        y = encoder_block(x, mask)
        assert y.shape == x.shape

    def test_residual_connection(self):
        # GIVEN
        batch_size = 2
        rings_per_bubble = 4
        d_ring_embedding = 256
        dropout = 0.0
        sublayer = FeedForwardBlock(d_ring_embedding, dropout)

        # WHEN/THEN
        residual_connection = ResidualConnection(d_ring_embedding, dropout)
        x = torch.rand(batch_size, rings_per_bubble, d_ring_embedding)
        y = residual_connection(x, sublayer)
        assert y.shape == x.shape

    def test_multi_head_attention_block_attention_scores(self):
        # GIVEN
        d_ring_embedding = 256
        h = 1
        batch_size = 2
        rings_per_bubble = 4
        mask = torch.ones(batch_size, 1, 1, rings_per_bubble)
        mask[0, :, :, -1] = 0.0
        mask[1, :, :, -1] = 0.0
        mask[1, :, :, -2] = 0.0

        # WHEN/THEN
        x = torch.rand(batch_size, rings_per_bubble, d_ring_embedding)
        multi_head_attention_block = MultiHeadAttentionBlock(d_ring_embedding, h, dropout=0.0)
        _ = multi_head_attention_block.forward(x, x, x, mask)
        attention_scores = multi_head_attention_block.attention_scores
        assert attention_scores.shape == (batch_size, h, rings_per_bubble, rings_per_bubble)
        assert torch.equal(attention_scores[0, 0, :, -1], torch.zeros(rings_per_bubble))
        assert torch.equal(attention_scores[1, 0, :, -1], torch.zeros(rings_per_bubble))
        assert torch.equal(attention_scores[1, 0, :, -2], torch.zeros(rings_per_bubble))

    def test_multi_head_attention_block_output_shape(self):
        # GIVEN
        d_ring_embedding = 256
        h = 8
        batch_size = 2
        rings_per_bubble = 5
        mask = torch.ones(batch_size, 1, 1, rings_per_bubble)

        # WHEN/THEN
        x = torch.rand(batch_size, rings_per_bubble, d_ring_embedding)
        multi_head_attention_block = MultiHeadAttentionBlock(d_ring_embedding, h, dropout=0.0)
        y = multi_head_attention_block.forward(x, x, x, mask)
        assert y.shape == x.shape

    def test_feed_forward_block(self):
        # GIVEN
        batch_size = 2
        rings_per_bubble = 5
        d_ring_embedding = 256

        # WHEN/THEN
        x = torch.rand(batch_size, rings_per_bubble, d_ring_embedding)
        feed_forward_block = FeedForwardBlock(d_ring_embedding, dropout=0.0)
        y = feed_forward_block(x)
        assert y.shape == x.shape

    def test_ring_embedding(self):
        # GIVEN
        device = torch.device("cuda")
        n_point_features = 3
        n_extracted_point_features = 64
        d_ring_embedding = 512
        model_resolution = 0.01
        x = torch.randn(2, 2, 1000, 3)

        x = x.to(torch.float32).to(device)

        # WHEN/THEN
        ring_embedding = RingEmbedding(n_point_features, n_extracted_point_features, model_resolution, d_ring_embedding)
        ring_embedding.to(device)
        ring_embeddings, per_point_features_extracted = ring_embedding(x)

        assert ring_embeddings.shape == (x.shape[0], x.shape[1], d_ring_embedding)
        assert per_point_features_extracted.shape == (x.shape[0], x.shape[1], x.shape[2], n_extracted_point_features)

    def test_batch_to_spconv_tensor(self):

        # GIVEN
        model_resolution = 0.1
        x = torch.tensor \
                ([
                    [  # bubble 0
                        [[0., 0., 0.], [0., 0., -0.2], [0.1, 0.0, 0.0]],  # ring 0
                        [[0., 0., 0.1], [0., 0., 0.2], [0.0, 0.2, 0.0]]  # ring 1
                    ],
                    [  # bubble 1
                        [[0., 0., 0.01], [0., 0., -0.11], [0.2, 0.0, 0.0]],  # ring 0
                        [[0., 0., 0.11], [0., 0., 0.21], [0.0, 0.3, 0.0]]  # ring 1
                    ]
                ], dtype=torch.float32)

        # WHEN
        features, indices, spatial_shape, batch_size = batch_to_spconv_tensor(x, model_resolution)
        expected_features = x.reshape(-1, 3).to(torch.float32)
        expected_indices = torch.tensor([[0, 0, 0, 2],
                                        [0, 0, 0, 0],
                                        [0, 1, 0, 2],
                                        [1, 0, 0, 3],
                                        [1, 0, 0, 4],
                                        [1, 0, 2, 2],
                                        [2, 0, 0, 2],
                                        [2, 0, 0, 0],
                                        [2, 2, 0, 2],
                                        [3, 0, 0, 3],
                                        [3, 0, 0, 4],
                                        [3, 0, 3, 2]])
        expected_indices = expected_indices.to(torch.int32)
        expected_spatial_shape = torch.tensor([3, 4, 5]).to(torch.int32)
        expected_batch_size = torch.tensor(4).to(torch.int32)

        assert torch.equal(features, expected_features)
        assert torch.equal(indices, expected_indices)
        assert torch.equal(spatial_shape, expected_spatial_shape)
        assert torch.equal(batch_size, expected_batch_size)

    def test_positional_encoding(self):

        # GIVEN
        device = torch.device("cuda")
        batch_size = 2
        rings_per_bubble = 5
        points_per_ring = 500
        d_ring_embedding = 256
        n_extracted_point_features = 64
        n_point_features = 3
        model_resolution = 0.01

        # WHEN/THEN
        positional_encoding = PositionalEncoding(d_ring_embedding, rings_per_bubble, dropout=0.0).to(device)
        assert positional_encoding.pe.shape == (1, rings_per_bubble, d_ring_embedding)

        x = torch.rand(batch_size, rings_per_bubble, points_per_ring, n_point_features).to(device)
        ring_embedding = \
            RingEmbedding(n_point_features, n_extracted_point_features, model_resolution, d_ring_embedding).to(device)

        y, _ = ring_embedding(x)

        assert y.shape == (batch_size, rings_per_bubble, d_ring_embedding)
        z = positional_encoding(y)
        assert z.shape == (batch_size, rings_per_bubble, d_ring_embedding)
        assert torch.equal(z, positional_encoding.pe + y)


if __name__ == '__main__':
    unittest.main()
