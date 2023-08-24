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


class Test_3d_transformer_dataset(unittest.TestCase):

    def test_collate_fn(self):
        # GIVEN
        batch = [(1, "a"), (2, "b"), (3, "c")]

        # WHEN
        data, target = collate_fn(batch)

        # THEN
        assert data == [1, 2, 3]
        assert target == ["a", "b", "c"]

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
            training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0,
                                         min_point_per_laz_file_factor=1)

            dataset = TokenizedBubbleDataset(os.path.join(tmp_local_dir, 'bubbles'), 4)
            dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

            # get next batch
            iterator = iter(dataloader)
            point_tokens, label_tokens = next(iterator)
            assert len(point_tokens) == 1
            assert len(label_tokens) == 1

            expected_point_tokens = [
                # batch 0 (each batch is a list of rings)
                torch.tensor([
                    [[0., 0., 0.],
                     [0., 0., -0.1],
                     [0., 0., -0.2],
                     [0., 0., -0.3],
                     [0, 0., -0.4]],
                    [[0., 0., -0.5],
                     [0., 0., -0.6],
                     [0., 0., -0.7],
                     [0., 0., -0.8],
                     [0., 0., -0.9]]
                ])
                # batch 1
                                    ]
            # use torch to check if two tensors are almost equal
            assert torch.equal(point_tokens[0], expected_point_tokens[0])
            assert point_tokens[0].dtype == torch.float32

            # [np.array([2, 0, 3, 1, 0]), np.array([1, 0, 3, 1, 2])] -> one hot encoded
            expected_label_tokens = [
                # batch 0 (each batch is a list of rings)
                torch.tensor([
                    [[0., 0., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.], [1., 0., 0., 0.]],
                    [[0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.], [0., 0., 1., 0.]]
                ])
                # batch 1
                                    ]
            assert torch.equal(label_tokens[0], expected_label_tokens[0])
            assert label_tokens[0].dtype == torch.float32

            # test for batch size 2
            dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)
            iterator = iter(dataloader)
            point_tokens, label_tokens = next(iterator)
            assert len(point_tokens) == 2
            assert len(label_tokens) == 2

    def test_training_bubbles_creator_laz_file_with_not_enough_points(self):
        # GIVEN
        max_points_per_bubble = 10
        max_points_per_ring = 5
        model_resolution = 0.01
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6]])
        classification = np.array([0, 1, 2, 3, 4, 5, 6])

        file_name = 'test.laz'
        # WHEN / THEN
        with tempfile.TemporaryDirectory() as tmp_local_dir:
            save_as_laz_file(points, os.path.join(tmp_local_dir, file_name), classification)
            training_bubbles_creator = TrainingBubblesCreator(max_points_per_bubble, max_points_per_ring,
                                                              model_resolution)
            with self.assertLogs() as captured:
                training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0,
                                             min_point_per_laz_file_factor=3)
                logs = [captured.records[i].message for i in range(len(captured.records))]
                assert f"Skipping {os.path.join(tmp_local_dir, file_name)} because it has too few points" in logs

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
            training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0,
                                         min_point_per_laz_file_factor=1)
            # get all files at output directory
            all_bubbles = glob.glob(os.path.join(tmp_local_dir, 'bubbles', "*.pt"))
            test_bubble = torch.load(all_bubbles[1])
            point_tokens = test_bubble[0]
            expected_point_tokens = np.array([
                [[0., 0., 0.], [0., 0., 0.1], [0., 0., 0.2], [0., 0., 0.3], [0, 0., 0.4]]
            ])
            assert point_tokens.shape == expected_point_tokens.shape
            assert np.allclose(point_tokens, expected_point_tokens)

            label_tokens = test_bubble[1]
            expected_label_tokens = np.array([[0, 1, 2, 3, 4]])
            assert label_tokens.shape == expected_label_tokens.shape
            assert np.array_equal(label_tokens, expected_label_tokens)

            test_bubble = torch.load(all_bubbles[0])
            point_tokens = test_bubble[0]
            expected_point_tokens = np.array([
                [[0., 0., -0.4], [0., 0., -0.5], [0., 0., -0.6], [0., 0., -0.7], [0, 0., -0.8]]
            ])
            assert point_tokens.shape == expected_point_tokens.shape
            assert np.allclose(point_tokens, expected_point_tokens)

            label_tokens = test_bubble[1]
            expected_label_tokens = np.array([[6, 5, 4, 3, 2]])
            assert label_tokens.shape == expected_label_tokens.shape
            assert np.array_equal(label_tokens, expected_label_tokens)

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
            training_bubbles_creator.run(tmp_local_dir, os.path.join(tmp_local_dir, 'bubbles'), 1.0,
                                         min_point_per_laz_file_factor=1)
            # get all files at output directory
            all_bubbles = glob.glob(os.path.join(tmp_local_dir, 'bubbles', "*.pt"))
            test_bubble = torch.load(all_bubbles[1])
            point_tokens = test_bubble[0]
            print(point_tokens)
            expected_point_tokens = np.array([
                [[0., 0., 0.], [0., 0., 0.1], [0., 0., 0.2], [0., 0., 0.3], [0, 0., 0.4]],
                [[0., 0., 0.5], [0., 0., 0.6], [0., 0., 0.7], [0., 0., 0.8], [0., 0., 0.9]]
            ])
            assert point_tokens.shape == expected_point_tokens.shape
            assert np.allclose(point_tokens, expected_point_tokens)

            label_tokens = test_bubble[1]
            expected_label_tokens = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
            assert label_tokens.shape == expected_label_tokens.shape
            assert np.array_equal(label_tokens, expected_label_tokens)

            test_bubble = torch.load(all_bubbles[0])
            point_tokens = test_bubble[0]
            expected_point_tokens = np.array([
                [[0., 0., 0.], [0., 0., -0.1], [0., 0., -0.2], [0., 0., -0.3], [0, 0., -0.4]],
                [[0., 0., -0.5], [0., 0., -0.6], [0., 0., -0.7], [0., 0., -0.8], [0., 0., -0.9]]
            ])
            assert point_tokens.shape == expected_point_tokens.shape
            assert np.allclose(point_tokens, expected_point_tokens)

            label_tokens = test_bubble[1]
            expected_label_tokens = np.array([[10, 9, 8, 7, 6], [5, 4, 3, 2, 1]])
            assert label_tokens.shape == expected_label_tokens.shape
            assert np.array_equal(label_tokens, expected_label_tokens)

    def test_training_bubbles_creator_bubble_to_rings(self):
        # GIVEN
        max_points_per_bubble = 25_000
        max_points_per_ring = 3
        model_resolution = 0.08
        train_bubble_centre = np.array([0.1, -0.1, 0.0])
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4],
                           [0.0, 0.0, 0.5], [0.0, 0.0, 0.6]])
        classification = np.array([1, 2, 3, 4, 5, 6, 7])

        # WHEN
        training_bubbles_creator = TrainingBubblesCreator(max_points_per_bubble, max_points_per_ring, model_resolution)
        point_tokens, label_tokens = training_bubbles_creator._split_bubble_to_rings(points, classification,
                                                                                     train_bubble_centre)

        # THEN
        print(point_tokens)
        expected_point_tokens = np.array([
            # ring 0
            [[-0.1, 0.1, 0.0],
             [-0.1, 0.1, 0.1],
             [-0.1, 0.1, 0.2]],
            # ring 1
            [[-0.1, 0.1, 0.3],
             [-0.1, 0.1, 0.4],
             [-0.1, 0.1, 0.5]]
        ])
        expected_label_tokens = np.array([
            # ring 0
            [1, 2, 3],
            # ring 1
            [4, 5, 6]
        ])
        assert point_tokens.shape == expected_point_tokens.shape
        assert np.array_equal(point_tokens, expected_point_tokens)
        assert label_tokens.shape == expected_label_tokens.shape
        assert np.array_equal(label_tokens, expected_label_tokens)

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

