import unittest
import os
import glob
import tempfile

import numpy as np
import laspy

import torch
from torch.utils.data import DataLoader

from dataset.datautils import get_data_from_laz_file, down_sample_point_data, get_down_sampled_points_and_classification, \
    bounding_box_calculator, calc_train_bubble_centres, save_as_laz_file
from dataset.dataset import TrainingBubblesCreator, TokenizedBubbleDataset, collate_fn

import logging
LOGGER = logging.getLogger(__name__)


class Test_3d_Transformer(unittest.TestCase):

    def test_collate_fn(self):
        # GIVEN
        batch = [(1, "a"), (2, "b"), (3, "c")]

        # WHEN
        data, target = collate_fn(batch)

        # THEN
        assert data == [1, 2, 3]
        assert target == ["a", "b", "c"]

    def test_dataloader_all_rings_dont_have_equal_n_points(self):
        # GIVEN
        max_points_per_bubble = 10
        max_points_per_ring = 5
        model_resolution = 0.01
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6]])
        classification = np.array([0, 1, 2, 3, 0, 1, 2])

        # WHEN / THEN
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            save_as_laz_file(points, os.path.join(tmp_local_dir, "test.laz"), classification)
            training_bubbles_creator = TrainingBubblesCreator(max_points_per_bubble, max_points_per_ring,
                                                              model_resolution)
            training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0)

            dataset = TokenizedBubbleDataset(os.path.join(tmp_local_dir, 'bubbles'), 4)
            dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

            expected_point_tokens = [
                np.array([[0., 0., -0.4], [0., 0., -0.5], [0., 0., -0.6], [0., 0., -0.7], [0, 0., -0.8]]),
                np.array([[0., 0., -0.9], [0., 0., -1.0]])
            ]

            # get next batch
            iterator = iter(dataloader)
            point_tokens, label_tokens = next(iterator)
            assert len(point_tokens) == 1
            assert len(label_tokens) == 1

            for point_token in point_tokens:
                assert len(point_token) == 2
                for counter, ring in enumerate(point_token):
                    assert np.allclose(np.asarray(ring.numpy()), np.asarray(expected_point_tokens[counter]))
                    assert ring.dtype == torch.float32

            # expected_label_tokens = [np.array([2, 1, 0, 3, 2]), np.array([1, 0])] -> one hot encoded
            expected_label_tokens = [
                # batch 0 (each batch is a list of rings)
                [
                    torch.tensor(
                        [[0., 0., 1., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 1., 0.]]),
                    torch.tensor(
                        [[0., 1., 0., 0.], [1., 0., 0., 0.]])
                ]
                # batch 1
            ]
            for label_token in label_tokens:
                assert len(label_token) == 2
                for counter, ring in enumerate(label_token):
                    assert ring.dtype == torch.float32
                    assert torch.equal(ring, expected_label_tokens[0][counter])

    def test_dataloader_all_rings_have_equal_n_points(self):
        # GIVEN
        max_points_per_bubble = 10
        max_points_per_ring = 5
        model_resolution = 0.01
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], [0.0, 0.0, 0.7], [0.0, 0.0, 0.8], [0.0, 0.0, 0.9],
                           [0.0, 0.0, 1.0]])
        classification = np.array([0, 2, 1, 3, 0, 1, 0, 1, 3, 0, 2])

        # WHEN / THEN
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            save_as_laz_file(points, os.path.join(tmp_local_dir, "test.laz"), classification)
            training_bubbles_creator = TrainingBubblesCreator(max_points_per_bubble, max_points_per_ring,
                                                              model_resolution)
            training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0)

            dataset = TokenizedBubbleDataset(os.path.join(tmp_local_dir, 'bubbles'), 4)
            dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

            # get next batch
            iterator = iter(dataloader)
            point_tokens, label_tokens = next(iterator)
            assert len(point_tokens) == 1
            assert len(label_tokens) == 1

            expected_point_tokens = [
                # batch 0 (each batch is a list of rings)
                [
                    np.array([[0., 0., 0.], [0., 0., -0.1], [0., 0., -0.2], [0., 0., -0.3], [0, 0., -0.4]]),
                    np.array([[0., 0., -0.5], [0., 0., -0.6], [0., 0., -0.7], [0., 0., -0.8], [0., 0., -0.9]])
                ]
                # batch 1
                                    ]
            for point_token in point_tokens:
                assert len(point_token) == 2
                for counter, ring in enumerate(point_token):
                    assert np.allclose(np.asarray(ring.numpy()), np.asarray(expected_point_tokens[0][counter]))
                    assert ring.dtype == torch.float32

            # [np.array([2, 0, 3, 1, 0]), np.array([1, 0, 3, 1, 2])] -> one hot encoded
            expected_label_tokens = [
                # batch 0 (each batch is a list of rings)
                [
                    torch.tensor([[0., 0., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.], [1., 0., 0., 0.]]),
                    torch.tensor([[0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
                ]
                # batch 1
                                    ]
            for label_token in label_tokens:
                assert len(label_token) == 2
                for counter, ring in enumerate(label_token):
                    assert ring.dtype == torch.float32
                    assert torch.equal(ring, expected_label_tokens[0][counter])

            # test for batch size 2
            dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)
            iterator = iter(dataloader)
            point_tokens, label_tokens = next(iterator)
            assert len(point_tokens) == 2
            assert len(label_tokens) == 2

    def test_training_bubbles_creator_total_points_less_than_max_points_per_bubble(self):
        # GIVEN
        max_points_per_bubble = 10
        max_points_per_ring = 5
        model_resolution = 0.01
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6]])
        classification = np.array([0, 1, 2, 3, 4, 5, 6])

        # WHEN / THEN
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            save_as_laz_file(points, os.path.join(tmp_local_dir, "test.laz"), classification)
            training_bubbles_creator = TrainingBubblesCreator(max_points_per_bubble, max_points_per_ring,
                                                              model_resolution)
            training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0)
            # get all files at output directory
            all_bubbles = glob.glob(os.path.join(tmp_local_dir, 'bubbles', "*.pt"))
            test_bubble = torch.load(all_bubbles[1])
            point_tokens = test_bubble[0]
            expected_point_tokens = [
                np.array([[0., 0., 0.], [0., 0., 0.1], [0., 0., 0.2], [0., 0., 0.3], [0, 0., 0.4]]),
                np.array([[0., 0., 0.5], [0., 0., 0.6]])
            ]
            for counter, point_token in enumerate(point_tokens):
                assert np.allclose(np.asarray(point_token), np.asarray(expected_point_tokens[counter]))

            label_tokens = test_bubble[1]
            expected_label_tokens = [np.array([0, 1, 2, 3, 4]), np.array([5, 6])]
            for counter, label_token in enumerate(label_tokens):
                assert np.array_equal(np.asarray(label_token), np.asarray(expected_label_tokens[counter]))

            test_bubble = torch.load(all_bubbles[0])
            point_tokens = test_bubble[0]
            expected_point_tokens = [
                np.array([[0., 0., -0.4], [0., 0., -0.5], [0., 0., -0.6], [0., 0., -0.7], [0, 0., -0.8]]),
                np.array([[0., 0., -0.9], [0., 0., -1.0]])
            ]
            for counter, point_token in enumerate(point_tokens):
                assert np.allclose(np.asarray(point_token), np.asarray(expected_point_tokens[counter]))

            label_tokens = test_bubble[1]
            expected_label_tokens = [np.array([6, 5, 4, 3, 2]), np.array([1, 0])]
            for counter, label_token in enumerate(label_tokens):
                assert np.array_equal(np.asarray(label_token), np.asarray(expected_label_tokens[counter]))

    def test_training_bubbles_creator_total_points_more_than_max_points_per_bubble(self):
        # GIVEN
        max_points_per_bubble = 10
        max_points_per_ring = 5
        model_resolution = 0.01
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], [0.0, 0.0, 0.7], [0.0, 0.0, 0.8], [0.0, 0.0, 0.9],
                           [0.0, 0.0, 1.0]])
        classification = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # WHEN / THEN
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            save_as_laz_file(points, os.path.join(tmp_local_dir, "test.laz"), classification)
            training_bubbles_creator = TrainingBubblesCreator(max_points_per_bubble, max_points_per_ring,
                                                              model_resolution)
            training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0)
            # get all files at output directory
            all_bubbles = glob.glob(os.path.join(tmp_local_dir, 'bubbles', "*.pt"))
            test_bubble = torch.load(all_bubbles[1])
            point_tokens = test_bubble[0]
            expected_point_tokens = [
                np.array([[0., 0., 0.], [0., 0., 0.1], [0., 0., 0.2], [0., 0., 0.3], [0, 0., 0.4]]),
                np.array([[0., 0., 0.5], [0., 0., 0.6], [0., 0., 0.7], [0., 0., 0.8], [0., 0., 0.9]])
            ]
            for counter, point_token in enumerate(point_tokens):
                assert np.allclose(np.asarray(point_token), np.asarray(expected_point_tokens[counter]))

            label_tokens = test_bubble[1]
            expected_label_tokens = [np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9])]
            for counter, label_token in enumerate(label_tokens):
                assert np.array_equal(np.asarray(label_token), np.asarray(expected_label_tokens[counter]))

            test_bubble = torch.load(all_bubbles[0])
            point_tokens = test_bubble[0]
            expected_point_tokens = [
                np.array([[0., 0., 0.], [0., 0., -0.1], [0., 0., -0.2], [0., 0., -0.3], [0, 0., -0.4]]),
                np.array([[0., 0., -0.5], [0., 0., -0.6], [0., 0., -0.7], [0., 0., -0.8], [0., 0., -0.9]])
            ]
            for counter, point_token in enumerate(point_tokens):
                assert np.allclose(np.asarray(point_token), np.asarray(expected_point_tokens[counter]))

            label_tokens = test_bubble[1]
            expected_label_tokens = [np.array([10, 9, 8, 7, 6]), np.array([5, 4, 3, 2, 1])]
            for counter, label_token in enumerate(label_tokens):
                assert np.array_equal(np.asarray(label_token), np.asarray(expected_label_tokens[counter]))

    def test_training_bubbles_creator_bubble_to_rings(self):
        # GIVEN
        max_points_per_bubble = 25_000
        max_points_per_ring = 3
        model_resolution = 0.08
        train_bubble_centre = np.array([0.1, -0.1, 0.0])
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3]])
        classification = np.array([1, 2, 3, 4])

        # WHEN
        training_bubbles_creator = TrainingBubblesCreator(max_points_per_bubble, max_points_per_ring, model_resolution)
        point_tokens, label_tokens = training_bubbles_creator._split_bubble_to_rings(points, classification,
                                                                                     train_bubble_centre)

        # THEN
        expected_point_tokens = [np.array([[-0.1, 0.1, 0.0], [-0.1, 0.1, 0.1], [-0.1, 0.1, 0.2]]),
                                 np.array([[-0.1, 0.1, 0.3]])]
        expected_label_tokens = [np.array([1, 2, 3]),
                                 np.array([4])]
        assert len(point_tokens) == len(expected_point_tokens)
        for counter, point_token in enumerate(point_tokens):
            assert np.array_equal(point_token, expected_point_tokens[counter])
        assert len(label_tokens) == len(expected_label_tokens)
        for counter, label_token in enumerate(label_tokens):
            assert np.array_equal(label_token, expected_label_tokens[counter])

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
            points_from_laz_file = get_data_from_laz_file(las_file, classification=False)

            # THEN
            assert np.array_equal(points, points_from_laz_file)

    def test_get_data_from_laz_file_points_and_classification(self):
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            # GIVEN
            points = np.array([[0.1, 0.0, -0.5],
                               [-0.5, 0.1, 1.0],
                               [100, 900, 0.4]])
            classification = np.array([1, 2, 3])

            las_file = os.path.join(tmp_local_dir, "test.laz")
            las = laspy.create(file_version="1.4", point_format=6)
            las.x = points[:, 0]
            las.y = points[:, 1]
            las.z = points[:, 2]
            las.classification = classification
            las.write(las_file)

            # WHEN
            points_out, classification_out = get_data_from_laz_file(las_file, classification=True)

            # THEN
            assert np.array_equal(points_out, points)
            assert np.array_equal(classification_out, classification)

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

