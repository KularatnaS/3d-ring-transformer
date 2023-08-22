import unittest
import os
import tempfile
import numpy as np
import laspy

from datautils import get_points_from_laz_file, down_sample_point_data

import logging
LOGGER = logging.getLogger(__name__)


class Test_3d_Transformer(unittest.TestCase):

    def test_get_points_from_laz_file(self):
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
            points_from_laz_file = get_points_from_laz_file(las_file)

            # THEN
            assert np.array_equal(points, points_from_laz_file)

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
    unittest.main()

