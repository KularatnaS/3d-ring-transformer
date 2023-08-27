import unittest

import numpy as np
import torch

from model.model import RingEmbedding, PositionalEncoding


import logging
LOGGER = logging.getLogger(__name__)


class Test_3d_transformer_model(unittest.TestCase):

    def test_ring_embedding(self):
        # GIVEN
        batch_size = 2
        rings_per_bubble = 5
        points_per_ring = 500
        d_ring_embedding = 256
        n_point_features = 3

        # WHEN/THEN -> no missing rings
        x = torch.rand(batch_size, rings_per_bubble, points_per_ring, n_point_features)
        ring_embedding = RingEmbedding(d_ring_embedding, n_point_features)
        assert ring_embedding.conv1.bias is not None
        assert ring_embedding.conv2.bias is not None
        assert ring_embedding.conv3.bias is not None
        assert ring_embedding.ln1.elementwise_affine is True
        assert ring_embedding.ln2.elementwise_affine is True
        assert ring_embedding.ln3.elementwise_affine is True

        y = ring_embedding(x)
        self.assertEqual(y.shape, (batch_size, rings_per_bubble, d_ring_embedding))

        # WHEN/THEN -> with missing rings
        x = torch.rand(batch_size, rings_per_bubble, points_per_ring, n_point_features)
        x[:, -1, :, :] = 0.0
        ring_embedding = RingEmbedding(d_ring_embedding, n_point_features, not_testing_padding=False)
        assert ring_embedding.conv1.bias is None
        assert ring_embedding.conv2.bias is None
        assert ring_embedding.conv3.bias is None
        assert ring_embedding.ln1.elementwise_affine is False
        assert ring_embedding.ln2.elementwise_affine is False
        assert ring_embedding.ln3.elementwise_affine is False

        y = ring_embedding(x)
        assert y.shape == (batch_size, rings_per_bubble, d_ring_embedding)
        assert torch.equal(y[:, -1, :], torch.zeros(batch_size, d_ring_embedding))

    def test_positional_encoding(self):

        # GIVEN
        batch_size = 2
        rings_per_bubble = 5
        points_per_ring = 500
        d_ring_embedding = 256
        n_point_features = 3

        # WHEN/THEN
        positional_encoding = PositionalEncoding(d_ring_embedding, rings_per_bubble, dropout=0.0)
        assert positional_encoding.pe.shape == (1, rings_per_bubble, d_ring_embedding)

        x = torch.rand(batch_size, rings_per_bubble, points_per_ring, n_point_features)
        ring_embedding = RingEmbedding(d_ring_embedding, n_point_features)
        y = ring_embedding(x)

        assert y.shape == (batch_size, rings_per_bubble, d_ring_embedding)
        z = positional_encoding(y)
        assert z.shape == (batch_size, rings_per_bubble, d_ring_embedding)
        assert torch.equal(z, positional_encoding.pe + y)


if __name__ == '__main__':
    unittest.main()
