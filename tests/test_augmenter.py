import unittest
import numpy as np

import torch

from train_utils.bubble_augmenter import BatchAugmenter

import logging
LOGGER = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)


class Test_bubble_augmenter(unittest.TestCase):

    def test_remove_padding_points_from_bubble(self):
        # GIVEN
        batch = torch.rand((2, 4, 10, 6))

        # WHEN
        batch_augmenter = BatchAugmenter(batch, model_resolution=0.01)
        batch_augmented = batch_augmenter.augment()

        # THEN
        self.assertEqual(batch_augmented.shape, batch.shape)
        assert type(batch_augmented) == torch.Tensor


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()

