import unittest

import numpy as np
import torch

from model.model import RingEmbedding


import logging
LOGGER = logging.getLogger(__name__)


class Test_3d_transformer_model(unittest.TestCase):

    def test_ring_embedding(self):
        # GIVEN
        d_ring_embedding = 256
        n_point_features = 3
        x = torch.rand(5, 500, n_point_features)

        # WHEN
        ring_embedding = RingEmbedding(d_ring_embedding, n_point_features)
        y = ring_embedding(x)

        # THEN
        self.assertEqual(y.shape, (5, d_ring_embedding))


if __name__ == '__main__':
    unittest.main()