import unittest
import os
import glob
import tempfile

import numpy as np
import laspy

import torch
from torch.utils.data import DataLoader

from dataset.datautils import get_data_from_laz_file, down_sample_point_data, get_down_sampled_points_and_classification, \
    bounding_box_calculator, calc_train_bubble_centres, save_as_laz_file, remove_padding_points_from_bubble
from dataset.dataset import TrainingBubblesCreator, TokenizedBubbleDataset, collate_fn

np.random.seed(0)

import logging
LOGGER = logging.getLogger(__name__)


class Test_3d_transformer_dataset(unittest.TestCase):

    def test_remove_padding_points_from_bubble(self):
        # GIVEN
        points = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.1],
                           [0.0, 0.0, 0.2],
                           [0.0, 0.0, 0.3],
                           [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5],
                           [0.0, 0.0, 0.6],
                           [0.0, 0.0, 0.7],
                           ])
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        points_per_ring = 4
        n_rings_per_bubble = 2
        ring_padding = 0.5

        points_out, labels_out = remove_padding_points_from_bubble(points, labels, n_rings_per_bubble, points_per_ring,
                                                                   ring_padding)
        expected_points_out = np.array([[0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.1],
                                        [0.0, 0.0, 0.4],
                                        [0.0, 0.0, 0.5]
                                       ])
        expected_labels_out = np.array([0, 1, 4, 5])
        assert np.allclose(points_out, expected_points_out)
        assert np.allclose(labels_out, expected_labels_out)

    def test_collate_fn(self):
        # GIVEN
        bubble_0 = torch.tensor \
                ([
                    [[0., 0., 0.], [0., 0., -0.1], [0., 0., -0.2], [0., 0., -0.3], [0, 0., -0.4]],  # ring 0
                    [[0., 0., -0.5], [0., 0., -0.6], [0., 0., -0.7], [0., 0., -0.8], [0., 0., -0.9]]  # ring 1
                ])
        bubble_1 = torch.tensor \
                ([
                    [[0., 0., 0.], [0.1, 0., -0.1], [0., 0., -0.2], [0., 0., -0.3], [0, 0., -0.4]],  # ring 0
                    [[0., 0., -0.5], [0.1, 0., -0.6], [0., 0., -0.7], [0.2, 0., -0.8], [0., 0., -0.9]]  # ring 1
                ])
        label_token_0 = \
            torch.tensor([  # bubble 0
                [[0., 0., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.], [1., 0., 0., 0.]],  # ring 0
                [[0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.], [0., 0., 1., 0.]]  # ring 1
            ])
        label_token_1 = \
            torch.tensor([  # bubble 0
                [[0., 0., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.], [0., 1., 0., 0.]],  # ring 0
                [[0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 1., 0.], [0., 0., 1., 0.]]  # ring 1
            ])
        n_missing_rings_0 = torch.tensor([[[0]]])
        n_missing_rings_1 = torch.tensor([[[1]]])
        batch = [(bubble_0, label_token_0, n_missing_rings_0), (bubble_1, label_token_1, n_missing_rings_1)]

        # WHEN
        point_tokens, label_tokens, n_missing_rings = collate_fn(batch)

        # THEN
        assert torch.equal(point_tokens, torch.stack([bubble_0, bubble_1]))
        assert torch.equal(label_tokens, torch.stack([label_token_0, label_token_1]))
        assert torch.equal(n_missing_rings, torch.stack([n_missing_rings_0, n_missing_rings_1]))

    def test_ring_padding(self):
        # GIVEN
        max_points_per_bubble = 10
        points_per_ring = 4
        rings_per_bubble = 2
        n_point_features = 3
        model_resolution = 0.01
        n_classes_model = 4
        ignore_index = -100
        ring_padding = 0.25

        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], [0.0, 0.0, 0.7], [0.0, 0.0, 0.8], [0.0, 0.0, 0.9],
                           [0.0, 0.0, 1.0]])
        classification = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # WHEN / THEN
        training_bubbles_creator = \
            TrainingBubblesCreator(max_points_per_bubble=max_points_per_bubble,
                                   points_per_ring=points_per_ring,
                                   rings_per_bubble=rings_per_bubble,
                                   n_point_features=n_point_features,
                                   model_resolution=model_resolution,
                                   n_classes_model=n_classes_model,
                                   ignore_index=ignore_index,
                                   ring_padding=ring_padding)

        point_tokens, label_tokens, n_missing_rings = training_bubbles_creator._split_bubble_to_rings(points,
                                                                                                      classification)
        expected_point_tokens = np.array([[[0., 0., 0.], [0., 0., -0.1], [0., 0., 0.1], [0., 0., -0.3]],
                                          [[0., 0., 0.2], [0., 0., -0.2], [0., 0., -0.3], [0., 0., 0.0]]])
        expected_label_tokens = np.array([[5, 4, 6, ignore_index], [7, 3, 2, ignore_index]])

        point_token_1 = point_tokens[0]
        point_token_2 = point_tokens[1]
        assert np.any(np.isin(point_token_1, point_token_2))
        assert np.any(np.isin(point_token_2, point_token_1))

        assert np.allclose(label_tokens, expected_label_tokens)

    def test_dataloader_batch_size_2(self):
        # GIVEN
        max_points_per_bubble = 10
        points_per_ring = 5
        rings_per_bubble = 4
        n_point_features = 3
        model_resolution = 0.01
        n_classes_model = 4
        ignore_index = -100
        ring_padding = 0.0

        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], [0.0, 0.0, 0.7], [0.0, 0.0, 0.8], [0.0, 0.0, 0.9],
                           [0.0, 0.0, 1.0]])
        classification = np.array([0, 2, 1, 3, 0, 1, 0, 1, 3, 0, 2])

        # WHEN / THEN
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            save_as_laz_file(points, os.path.join(tmp_local_dir, "test.laz"), classification)
            training_bubbles_creator = \
                TrainingBubblesCreator(max_points_per_bubble=max_points_per_bubble,
                                       points_per_ring=points_per_ring,
                                       rings_per_bubble=rings_per_bubble,
                                       n_point_features=n_point_features,
                                       model_resolution=model_resolution,
                                       n_classes_model=n_classes_model,
                                       ignore_index=ignore_index,
                                       ring_padding=ring_padding)
            training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0,
                                         min_rings_per_laz=1)

            dataset = TokenizedBubbleDataset(os.path.join(tmp_local_dir, 'bubbles'), n_classes_model, rings_per_bubble)

            # test for batch size 2
            dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)
            iterator = iter(dataloader)
            point_tokens, label_tokens, encoder_mask = next(iterator)
            assert point_tokens.shape[0] == 2
            assert label_tokens.shape[0] == 2
            assert encoder_mask.shape[0] == 2

            expected_point_tokens = \
                torch.tensor \
                        ([  # batch
                            [  # bubble 0
                                [[0., 0., 0.05], [0., 0., -0.05], [0., 0., 0.15], [0., 0., -0.15], [0, 0., 0.25]],  # ring 0
                                [[0., 0., -0.25], [0., 0., -0.35], [0., 0., 0.35], [0., 0., 0.45], [0., 0., -0.45]],  # ring 1
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0, 0., 0.], [0, 0., 0.]],  # ring 2 padding
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0, 0., 0.], [0, 0., 0.]]  # ring 3 padding
                            ],
                            [  # bubble 1
                                [[0., 0., -0.05], [0., 0., 0.05], [0., 0., 0.15], [0., 0., -0.15], [0, 0., -0.25]],  # ring 0
                                [[0., 0., 0.25], [0., 0., -0.35], [0., 0., 0.35], [0., 0., -0.45], [0., 0., 0.45]],  # ring 1
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0, 0., 0.], [0, 0., 0.]],  # ring 2 padding
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0, 0., 0.], [0, 0., 0.]]  # ring 3 padding
                            ]
                        ])
            # use torch to check if two tensors are almost equal
            assert torch.equal(point_tokens, expected_point_tokens)
            assert point_tokens.dtype == torch.float32

            # [np.array([2, 0, 3, 1, 0]), np.array([1, 0, 3, 1, 2])] -> one hot encoded
            expected_label_tokens = \
                torch.tensor([  # batch
                    [  # bubble 0
                        [0., 1., 1., 0., 3.],  # ring 0
                        [3., 1., 0., 2., 2.],  # ring 1
                        [0., 0., 0., 0., 0.],  # ring 2 padding
                        [0., 0., 0., 0., 0.]  # ring 3 padding
                    ],
                    [  # bubble 1  classification = np.array([0, 2, 1, 3, 0, 1, 0, 1, 3, 0, 2])
                        [0., 1., 0., 3., 1.],  # ring 0
                        [1., 2., 3., 0., 0.],  # ring 1
                        [0., 0., 0., 0., 0.],  # ring 2 padding
                        [0., 0., 0., 0., 0.]   # ring 3 padding
                    ]
                ])
            assert torch.equal(label_tokens, expected_label_tokens)
            assert label_tokens.dtype == torch.float32

            expected_encoder_mask = torch.tensor([
                                                  [[[1, 1, 0, 0]]],
                                                  [[[1, 1, 0, 0]]]
                                                 ], dtype=torch.int32)
            assert torch.equal(encoder_mask, expected_encoder_mask)
            assert encoder_mask.dtype == torch.int32

    def test_dataloader_batch_size_1(self):
        # GIVEN
        max_points_per_bubble = 10
        points_per_ring = 5
        rings_per_bubble = 2
        n_point_features = 3
        model_resolution = 0.01
        n_classes_model = 4
        ignore_index = -100
        ring_padding = 0.0

        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], [0.0, 0.0, 0.7], [0.0, 0.0, 0.8], [0.0, 0.0, 0.9],
                           [0.0, 0.0, 1.0]])
        classification = np.array([0, 2, 1, 3, 0, 1, 0, 1, 3, 0, 2])

        # WHEN / THEN
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            save_as_laz_file(points, os.path.join(tmp_local_dir, "test.laz"), classification)
            training_bubbles_creator = \
                TrainingBubblesCreator(max_points_per_bubble=max_points_per_bubble,
                                       points_per_ring=points_per_ring,
                                       rings_per_bubble=rings_per_bubble,
                                       n_point_features=n_point_features,
                                       model_resolution=model_resolution,
                                       n_classes_model=n_classes_model,
                                       ignore_index=ignore_index,
                                       ring_padding=ring_padding)
            training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0,
                                         min_rings_per_laz=1)

            dataset = TokenizedBubbleDataset(os.path.join(tmp_local_dir, 'bubbles'), n_classes_model, rings_per_bubble)
            dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

            # get next batch
            iterator = iter(dataloader)
            point_tokens, label_tokens, encoder_mask = next(iterator)
            assert len(point_tokens) == 1
            assert len(label_tokens) == 1
            assert len(encoder_mask) == 1

            expected_point_tokens = \
                torch.tensor \
                        ([[  # bubble 0
                        [[0., 0., 0.05], [0., 0., -0.05], [0., 0., 0.15], [0., 0., -0.15], [0, 0., 0.25]],  # ring 0
                        [[0., 0., -0.25], [0., 0., -0.35], [0., 0., 0.35], [0., 0., 0.45], [0., 0., -0.45]]  # ring 1
                    ]])
            # use torch to check if two tensors are almost equal
            assert torch.equal(point_tokens, expected_point_tokens)
            assert point_tokens.dtype == torch.float32

            # [np.array([2, 0, 3, 1, 0]), np.array([1, 0, 3, 1, 2])] -> one hot encoded
            expected_label_tokens = \
                torch.tensor(  # bubble 0
                    [[[0., 1., 1., 0., 3.],  # ring 0
                     [3., 1., 0., 2., 2.]]]  # ring 1
                )
            assert torch.equal(label_tokens, expected_label_tokens)
            assert label_tokens.dtype == torch.float32

            expected_encoder_mask = torch.tensor([[[[1, 1]]]], dtype=torch.int32)
            assert torch.equal(encoder_mask, expected_encoder_mask)
            assert encoder_mask.dtype == torch.int32

    def test_dataset_get_item_with_ring_padding(self):
        # GIVEN
        max_points_per_bubble = 10
        points_per_ring = 5
        rings_per_bubble = 4
        n_point_features = 3
        model_resolution = 0.01
        n_classes_model = 4
        ignore_index = -100
        ring_padding = 0.0

        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], [0.0, 0.0, 0.7], [0.0, 0.0, 0.8], [0.0, 0.0, 0.9],
                           [0.0, 0.0, 1.0]])
        classification = np.array([0, 2, 1, 3, 0, 1, 0, 1, 3, 0, 2])

        # WHEN / THEN
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            save_as_laz_file(points, os.path.join(tmp_local_dir, "test.laz"), classification)
            training_bubbles_creator = \
                TrainingBubblesCreator(max_points_per_bubble=max_points_per_bubble,
                                       points_per_ring=points_per_ring,
                                       rings_per_bubble=rings_per_bubble,
                                       n_point_features=n_point_features,
                                       model_resolution=model_resolution,
                                       n_classes_model=n_classes_model,
                                       ignore_index=ignore_index,
                                       ring_padding=ring_padding)
            training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0,
                                         min_rings_per_laz=1)

            dataset = TokenizedBubbleDataset(os.path.join(tmp_local_dir, 'bubbles'), n_classes_model, rings_per_bubble)
            point_tokens, label_tokens, encoder_mask = dataset.__getitem__(0)
            expected_point_tokens = \
                torch.tensor \
                        ([  # bubble 0
                            [[0., 0., 0.05], [0., 0., -0.05], [0., 0., 0.15], [0., 0., -0.15], [0, 0., 0.25]],  # ring 0
                            [[0., 0., -0.25], [0., 0., -0.35], [0., 0., 0.35], [0., 0., 0.45], [0., 0., -0.45]],  # ring 1
                            [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0, 0., 0.], [0, 0., 0.]],  # ring 2 padding
                            [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0, 0., 0.], [0, 0., 0.]]  # ring 3 padding
                        ])
            # use torch to check if two tensors are almost equal
            assert torch.equal(point_tokens, expected_point_tokens)
            assert point_tokens.dtype == torch.float32

            # [np.array([2, 0, 3, 1, 0]), np.array([1, 0, 3, 1, 2])] -> one hot encoded
            expected_label_tokens = \
                torch.tensor(  # bubble 0
                    [[0., 1., 1., 0., 3.],  # ring 0
                     [3., 1., 0., 2., 2.],  # ring 1
                     [0., 0., 0., 0., 0.],  # ring 2 padding
                     [0., 0., 0., 0., 0.]]  # ring 3 padding
                )
            assert torch.equal(label_tokens, expected_label_tokens)
            assert label_tokens.dtype == torch.float32

            expected_encoder_mask = torch.tensor([[[1, 1, 0, 0]]], dtype=torch.int32)
            assert torch.equal(encoder_mask, expected_encoder_mask)
            assert encoder_mask.dtype == torch.int32

    def test_dataset_get_item_no_ring_padding(self):
        # GIVEN
        max_points_per_bubble = 10
        points_per_ring = 5
        rings_per_bubble = 2
        n_point_features = 3
        model_resolution = 0.01
        n_classes_model = 4
        ignore_index = -100
        ring_padding = 0.0

        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], [0.0, 0.0, 0.7], [0.0, 0.0, 0.8], [0.0, 0.0, 0.9],
                           [0.0, 0.0, 1.0]])
        classification = np.array([0, 2, 1, 3, 0, 1, 0, 1, 3, 0, 2])

        # WHEN / THEN
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            save_as_laz_file(points, os.path.join(tmp_local_dir, "test.laz"), classification)
            training_bubbles_creator = \
                TrainingBubblesCreator(max_points_per_bubble=max_points_per_bubble,
                                       points_per_ring=points_per_ring,
                                       rings_per_bubble=rings_per_bubble,
                                       n_point_features=n_point_features,
                                       model_resolution=model_resolution,
                                       n_classes_model=n_classes_model,
                                       ignore_index=ignore_index,
                                       ring_padding=ring_padding)
            training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0,
                                         min_rings_per_laz=1)

            dataset = TokenizedBubbleDataset(os.path.join(tmp_local_dir, 'bubbles'), n_classes_model, rings_per_bubble)
            point_tokens, label_tokens, encoder_mask = dataset.__getitem__(0)
            expected_point_tokens = \
                torch.tensor \
                        ([  # bubble 0
                            [[0., 0., 0.05], [0., 0., -0.05], [0., 0., 0.15], [0., 0., -0.15], [0, 0., 0.25]],  # ring 0
                            [[0., 0., -0.25], [0., 0., -0.35], [0., 0., 0.35], [0., 0., 0.45], [0., 0., -0.45]]  # ring 1
                        ])
            # use torch to check if two tensors are almost equal
            assert torch.equal(point_tokens, expected_point_tokens)
            assert point_tokens.dtype == torch.float32

            # [np.array([2, 0, 3, 1, 0]), np.array([1, 0, 3, 1, 2])] -> one hot encoded
            expected_label_tokens = \
                torch.tensor(  # bubble 0
                                [[0., 1., 1., 0., 3.], # ring 0
                                [3., 1., 0., 2., 2.]]  # ring 1
                            )
            assert torch.equal(label_tokens, expected_label_tokens)
            assert label_tokens.dtype == torch.float32
            expected_encoder_mask = torch.tensor([[[1, 1]]], dtype=torch.int32)
            assert torch.equal(encoder_mask, expected_encoder_mask)
            assert encoder_mask.dtype == torch.int32

            # print total number of bubbles at output directory
            all_bubbles = glob.glob(os.path.join(tmp_local_dir, 'bubbles', "*.pt"))
            assert len(all_bubbles) == dataset.__len__()

    def test_training_bubbles_creator_laz_file_with_not_enough_points(self):
        # GIVEN
        max_points_per_bubble = 10
        points_per_ring = 5
        rings_per_bubble = 2
        n_point_features = 3
        model_resolution = 0.01
        n_classes_model = 7
        ignore_index = -100
        ring_padding = 0.0

        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6]])
        classification = np.array([0, 1, 2, 3, 4, 5, 6])

        file_name = 'test.laz'
        # WHEN / THEN
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            save_as_laz_file(points, os.path.join(tmp_local_dir, file_name), classification)
            training_bubbles_creator = \
                TrainingBubblesCreator(max_points_per_bubble=max_points_per_bubble,
                                       points_per_ring=points_per_ring,
                                       rings_per_bubble=rings_per_bubble,
                                       n_point_features=n_point_features,
                                       model_resolution=model_resolution,
                                       n_classes_model=n_classes_model,
                                       ignore_index=ignore_index,
                                       ring_padding=ring_padding)
            with self.assertLogs() as captured:
                training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0,
                                             min_rings_per_laz=3)
                logs = [captured.records[i].message for i in range(len(captured.records))]
                assert f"Skipping {os.path.join(tmp_local_dir, file_name)} because it has too few points" in logs

    def test_training_bubbles_creator_total_points_less_than_max_points_per_bubble(self):
        # GIVEN
        max_points_per_bubble = 10
        points_per_ring = 5
        rings_per_bubble = 2
        n_point_features = 3
        model_resolution = 0.01
        n_classes_model = 7
        ignore_index = -100
        ring_padding = 0.0

        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6]])
        classification = np.array([0, 1, 2, 3, 4, 5, 6])

        # WHEN / THEN
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            save_as_laz_file(points, os.path.join(tmp_local_dir, "test.laz"), classification)
            training_bubbles_creator = \
                TrainingBubblesCreator(max_points_per_bubble=max_points_per_bubble,
                                       points_per_ring=points_per_ring,
                                       rings_per_bubble=rings_per_bubble,
                                       n_point_features=n_point_features,
                                       model_resolution=model_resolution,
                                       n_classes_model=n_classes_model,
                                       ignore_index=ignore_index,
                                       ring_padding=ring_padding)
            training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0,
                                         min_rings_per_laz=1)
            # get all files at output directory
            all_bubbles = glob.glob(os.path.join(tmp_local_dir, 'bubbles', "*.pt"))
            test_bubble = torch.load(all_bubbles[1])
            point_tokens = test_bubble[0]
            expected_point_tokens = np.array([
                [[0., 0., 0.],
                 [0., 0., -0.1],
                 [0., 0., 0.1],
                 [0., 0., -0.2],
                 [0, 0., 0.2]],
                # mask ring
                [[0., 0., 0.0],
                 [0., 0., 0.0],
                 [0., 0., 0.0],
                 [0., 0., 0.0],
                 [0, 0., 0.0]]
            ])
            assert point_tokens.shape == expected_point_tokens.shape
            assert np.allclose(point_tokens, expected_point_tokens)

            label_tokens = test_bubble[1]
            expected_label_tokens = np.array([[3, 2, 4, 1, 5], [0, 0, 0, 0, 0]])
            assert label_tokens.shape == expected_label_tokens.shape
            assert np.array_equal(label_tokens, expected_label_tokens)

            test_bubble = torch.load(all_bubbles[0])
            point_tokens = test_bubble[0]
            expected_point_tokens = np.array([
                [[0., 0., 0.], [0., 0., -0.1], [0., 0., 0.1], [0., 0., -0.2], [0, 0., 0.2]],
                # mask ring
                [[0., 0., 0.0], [0., 0., 0.0], [0., 0., 0.0], [0., 0., 0.0], [0, 0., 0.0]]
            ])
            assert point_tokens.shape == expected_point_tokens.shape
            assert np.allclose(point_tokens, expected_point_tokens)

            label_tokens = test_bubble[1]
            expected_label_tokens = np.array([
                                                [3, 2, 4, 1, 5],
                                                [0, 0, 0, 0, 0]
                                            ])
            assert label_tokens.shape == expected_label_tokens.shape
            assert np.array_equal(label_tokens, expected_label_tokens)

    def test_training_bubbles_creator_total_points_more_than_max_points_per_bubble(self):
        # GIVEN
        max_points_per_bubble = 10
        points_per_ring = 5
        rings_per_bubble = 2
        n_point_features = 3
        model_resolution = 0.01
        n_classes_model = 11
        ignore_index = -100
        ring_padding = 0.0

        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], [0.0, 0.0, 0.7], [0.0, 0.0, 0.8], [0.0, 0.0, 0.9],
                           [0.0, 0.0, 1.0]])
        classification = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # WHEN / THEN
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            save_as_laz_file(points, os.path.join(tmp_local_dir, "test.laz"), classification)
            training_bubbles_creator = \
                TrainingBubblesCreator(max_points_per_bubble=max_points_per_bubble,
                                       points_per_ring=points_per_ring,
                                       rings_per_bubble=rings_per_bubble,
                                       n_point_features=n_point_features,
                                       model_resolution=model_resolution,
                                       n_classes_model=n_classes_model,
                                       ignore_index=ignore_index,
                                       ring_padding=ring_padding)
            training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0,
                                         min_rings_per_laz=1)
            # get all files at output directory
            all_bubbles = glob.glob(os.path.join(tmp_local_dir, 'bubbles', "*.pt"))
            test_bubble = torch.load(all_bubbles[1])
            point_tokens = test_bubble[0]
            expected_point_tokens = np.array([
                [[0., 0., -0.05], [0., 0., 0.05], [0., 0., 0.15], [0., 0., -0.15], [0, 0., -0.25]],
                [[0., 0., 0.25], [0., 0., -0.35], [0., 0., 0.35], [0., 0., -0.45], [0., 0., 0.45]]
            ])
            assert point_tokens.shape == expected_point_tokens.shape
            assert np.allclose(point_tokens, expected_point_tokens)

            label_tokens = test_bubble[1]
            expected_label_tokens = np.array([[4, 5, 6, 3, 2], [7, 1, 8, 0, 9]])
            assert label_tokens.shape == expected_label_tokens.shape
            assert np.array_equal(label_tokens, expected_label_tokens)

            test_bubble = torch.load(all_bubbles[0])
            point_tokens = test_bubble[0]
            expected_point_tokens = np.array([
                [[0., 0., 0.05], [0., 0., -0.05], [0., 0., 0.15], [0., 0., -0.15], [0, 0., 0.25]],
                [[0., 0., -0.25], [0., 0., -0.35], [0., 0., 0.35], [0., 0., 0.45], [0., 0., -0.45]]
            ])
            assert point_tokens.shape == expected_point_tokens.shape
            assert np.allclose(point_tokens, expected_point_tokens)

            label_tokens = test_bubble[1]
            expected_label_tokens = np.array([[6, 5, 7, 4, 8], [3, 2, 9, 10, 1]])
            assert label_tokens.shape == expected_label_tokens.shape
            assert np.array_equal(label_tokens, expected_label_tokens)

    def test_split_bubble_to_rings(self):

        # GIVEN
        max_points_per_bubble = 25_000
        points_per_ring = 3
        rings_per_bubble = 2
        n_point_features = 3
        n_classes_model = 8
        ignore_index = -100
        ring_padding = 0.0

        model_resolution = 0.08
        points = np.array([[-1.0, 0.0, 0.0],
                           [0.5, 0.0, 0.0],
                           [0.51, 0.0, 0.0],
                           [0.52, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0]])
        classification = np.array([0, 1, 2, 3, 4, 5])

        # WHEN
        training_bubbles_creator = \
            TrainingBubblesCreator(max_points_per_bubble=max_points_per_bubble,
                                   points_per_ring=points_per_ring,
                                   rings_per_bubble=rings_per_bubble,
                                   n_point_features=n_point_features,
                                   model_resolution=model_resolution,
                                   n_classes_model=n_classes_model,
                                   ignore_index=ignore_index,
                                   ring_padding=ring_padding)

        point_tokens, label_tokens, n_missing_rings = training_bubbles_creator._split_bubble_to_rings(points,
                                                                                                      classification)
        print(point_tokens)

    def test_training_bubbles_creator_bubble_to_rings_without_mask(self):
        # GIVEN
        max_points_per_bubble = 25_000
        points_per_ring = 3
        rings_per_bubble = 2
        n_point_features = 3
        n_classes_model = 8
        ignore_index = -100
        ring_padding = 0.0

        model_resolution = 0.08
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6]])
        classification = np.array([1, 2, 3, 4, 5, 6, 7])

        # WHEN
        training_bubbles_creator = \
            TrainingBubblesCreator(max_points_per_bubble=max_points_per_bubble,
                                   points_per_ring=points_per_ring,
                                   rings_per_bubble=rings_per_bubble,
                                   n_point_features=n_point_features,
                                   model_resolution=model_resolution,
                                   n_classes_model=n_classes_model,
                                   ignore_index=ignore_index,
                                   ring_padding=ring_padding)
        point_tokens, label_tokens, n_missing_rings = \
            training_bubbles_creator._split_bubble_to_rings(points, classification)

        # THEN
        expected_point_tokens = np.array([
            # ring 0
            [[0.0, 0.0, 0.0],
             [0.0, 0.0, -0.1],
             [0.0, 0.0, 0.1]],
            # ring 1
            [[0.0, 0.0, -0.2],
             [0.0, 0.0, 0.2],
             [0.0, 0.0, 0.3]],
        ])
        expected_label_tokens = np.array([
            # ring 0
            [4, 3, 5],
            # ring 1
            [2, 6, 7],
        ])

        assert point_tokens.shape == expected_point_tokens.shape
        assert np.allclose(point_tokens, expected_point_tokens)

        assert label_tokens.shape == expected_label_tokens.shape
        assert np.array_equal(label_tokens, expected_label_tokens)

        assert n_missing_rings == 0

    def test_training_bubbles_creator_bubble_to_rings_with_mask(self):
        # GIVEN
        max_points_per_bubble = 25_000
        points_per_ring = 3
        rings_per_bubble = 3
        n_point_features = 3
        n_classes_model = 8
        ignore_index = -100
        ring_padding = 0.0

        model_resolution = 0.08
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6]])
        classification = np.array([1, 2, 3, 4, 5, 6, 7])

        # WHEN
        training_bubbles_creator = \
            TrainingBubblesCreator(max_points_per_bubble=max_points_per_bubble,
                                   points_per_ring=points_per_ring,
                                   rings_per_bubble=rings_per_bubble,
                                   n_point_features=n_point_features,
                                   model_resolution=model_resolution,
                                   n_classes_model=n_classes_model,
                                   ignore_index=ignore_index,
                                   ring_padding=ring_padding)
        point_tokens, label_tokens, n_missing_rings = \
            training_bubbles_creator._split_bubble_to_rings(points, classification)

        # THEN
        expected_point_tokens = np.array([
            # ring 0
            [[0.0, 0.0, 0.0],
             [0.0, 0.0, -0.1],
             [0.0, 0.0, 0.1]],
            # ring 1
            [[0.0, 0.0, -0.2],
             [0.0, 0.0, 0.2],
             [0.0, 0.0, 0.3]],
            # ring 2 (mask)
            [[0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]]
        ])
        expected_label_tokens = np.array([
            # ring 0
            [4, 3, 5],
            # ring 1
            [2, 6, 7],
            # ring 2 (mask)
            [0, 0, 0]
        ])

        assert point_tokens.shape == expected_point_tokens.shape
        assert np.allclose(point_tokens, expected_point_tokens)

        assert label_tokens.shape == expected_label_tokens.shape
        assert np.array_equal(label_tokens, expected_label_tokens)

        assert n_missing_rings == 1

    def test_calc_train_bubble_centres(self):
        # GIVEN
        bounding_box_dimensions = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
        bubble_slide_step = 0.5

        # WHEN
        train_bubble_centres = calc_train_bubble_centres(bounding_box_dimensions, bubble_slide_step)
        expected_train_bubble_centres = np.array([[0.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.5],
                                                  [0.5, 0.0, 0.0],
                                                  [0.5, 0.0, 0.5],
                                                  [0.0, 0.5, 0.0],
                                                  [0.0, 0.5, 0.5],
                                                  [0.5, 0.5, 0.0],
                                                  [0.5, 0.5, 0.5]])

        # THEN
        assert np.array_equal(train_bubble_centres, expected_train_bubble_centres)

    def test_bounding_box_calculator(self):
        # GIVEN
        point_cloud = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 1.0]])

        # WHEN
        bounding_box = bounding_box_calculator(point_cloud)

        # THEN
        expected_bounding_box = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 1.0])
        assert np.array_equal(bounding_box, expected_bounding_box)

    def test_get_down_sampled_points_and_classification(self):
        # GIVEN
        points = np.array([[0.1, 0.0, 0.0],
                           [0.11, 0.0, 0.0],
                           [0.2, 0.0, 0.0],
                           [0.0, 0.0, -2.0],
                           [0.0, 0.0, -3.0]])
        classification = np.array([1, 1, 2, 3, 4])

        # WHEN
        down_sampled_points, down_sampled_classification = \
            get_down_sampled_points_and_classification(points, classification, 0.5)
        expected_down_sampled_points = np.array([[0.0, 0.0, -3.0],
                                                 [0.0, 0.0, -2.0],
                                                 [(0.1 + 0.11 + 0.2)/3.0, 0.0, 0.0]])
        expected_down_sampled_classification = np.array([4, 3, 1])

        # THEN
        assert np.array_equal(down_sampled_points, expected_down_sampled_points)
        assert np.array_equal(down_sampled_classification, expected_down_sampled_classification)

    def test_get_data_from_laz_file_points_only(self):
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            # GIVEN
            points = np.array([[0.1, 0.0, -0.5],
                               [-0.5, 0.1, 1.0],
                               [100, 900, 0.4]])
            las_file = os.path.join(tmp_local_dir, "test.laz")
            las = laspy.create(file_version="1.4", point_format=6)
            las.x = points[:, 0]
            las.y = points[:, 1]
            las.z = points[:, 2]
            las.write(las_file)

            # WHEN
            points_from_laz_file = get_data_from_laz_file(las_file, n_classes_model=None, classification=False)

            # THEN
            assert np.array_equal(points, points_from_laz_file)

    def test_get_data_from_laz_file_points_and_classification(self):
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            # GIVEN
            points = np.array([[0.1, 0.0, -0.5],
                               [-0.5, 0.1, 1.0],
                               [100, 900, 0.4],
                               [100, 900, 0.4]])

            classification = np.array([1, 20, 2, 0])

            las_file = os.path.join(tmp_local_dir, "test.laz")
            las = laspy.create(file_version="1.4", point_format=6)
            las.x = points[:, 0]
            las.y = points[:, 1]
            las.z = points[:, 2]
            las.classification = classification
            las.write(las_file)

            # WHEN
            points_out, classification_out = get_data_from_laz_file(las_file, n_classes_model=3, classification=True)
            expected_points_out = np.array([[0.1, 0.0, -0.5],
                                            [100, 900, 0.4],
                                            [100, 900, 0.4]])
            expected_classification_out = np.array([1, 2, 0])

            # THEN
            assert np.array_equal(points_out, expected_points_out)
            assert np.array_equal(classification_out, expected_classification_out)

    def test_down_sample_point_data(self):

        # GIVEN
        voxel_size = 4.0
        pc_input = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],  # voxel 0 -> index = 0
                             [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # voxel 0 -> index = 1
                             [+3.0, -1.0, -1.0, -1.0, -1.0, -1.0],  # voxel 3 -> index = 2
                             [+2.0, +2.0, +2.0, +2.0, +2.0, +1.0],  # voxel 0 -> index = 3
                             [-1.0, +5.0, -1.0, -1.0, -1.0, -1.0],  # voxel 2 -> index = 4
                             [-1.0, -1.0, +5.0, -1.0, -1.0, -1.0]]) # voxel 1 -> index = 5

        # WHEN
        pc_down_sampled, voxel_map = down_sample_point_data(voxel_size=voxel_size, pc_input=pc_input)
        expected_voxel_values = [[ 1/3,  1/3,  1/3,  1/3,  1/3,  0.0],  # voxel 0
                                 [-1.0, -1.0, +5.0, -1.0, -1.0, -1.0],  # voxel 1
                                 [-1.0, +5.0, -1.0, -1.0, -1.0, -1.0],  # voxel 2
                                 [+3.0, -1.0, -1.0, -1.0, -1.0, -1.0]]  # voxel 3
        expected_n_voxels = len(expected_voxel_values)
        pc_down_sampled_list = pc_down_sampled.tolist()

        # THEN
        assert len(pc_down_sampled) == expected_n_voxels
        for expected_voxel_value in expected_voxel_values:
            assert expected_voxel_value in pc_down_sampled_list
        assert len(voxel_map) == expected_n_voxels
        assert voxel_map[0] == [0, 1, 3]
        assert voxel_map[1] == [5]
        assert voxel_map[2] == [4]
        assert voxel_map[3] == [2]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()

