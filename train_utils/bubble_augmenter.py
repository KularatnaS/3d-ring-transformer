import numpy as np
import torch
from einops import rearrange

import logging
LOGGER = logging.getLogger(__name__)


class BatchAugmenter:

    def __init__(self, batch, model_resolution):
        self.model_resolution = model_resolution
        #  batch -> (batch_size, n_rings, points_per_ring, n_features)
        self.n_rings = batch.shape[1]
        self.points_per_ring = batch.shape[2]
        self.batch = rearrange(batch, 'b n p f -> b (n p) f')
        # convert from torch to numpy using from numpy, batch is already in cpu
        self.batch = self.batch.numpy()
        self.n_bubbles = batch.shape[0]

    def augment(self):
        for n in range(self.n_bubbles):

            # jitter bubble
            self.batch[n] = self.jitter_bubble(self.batch[n])

            # rotate bubble about x,y,z axis
            self.batch[n] = self.rotate_bubble_z_axis(min_alpha=-np.pi / 2, max_alpha=np.pi / 2, bubble=self.batch[n])
            self.batch[n] = self.rotate_bubble_x_axis(min_alpha=-np.pi / 36, max_alpha=np.pi / 36, bubble=self.batch[n])
            self.batch[n] = self.rotate_bubble_y_axis(min_alpha=-np.pi / 36, max_alpha=np.pi / 36, bubble=self.batch[n])

            # scale bubble
            self.batch[n] = self.scale_bubble(min_percentage=0.8, max_percentage=1.2,
                                              axis=np.random.randint(low=0, high=3), bubble=self.batch[n])

            self.batch[n] = self.recentre_bubble(self.batch[n])

        # convert from numpy to torch
        self.batch = torch.from_numpy(self.batch)
        return rearrange(self.batch, 'b (n p) f -> b n p f', n=self.n_rings, p=self.points_per_ring)

    def recentre_bubble(self, bubble):
        mid_point = (np.amax(bubble, axis=0)[0:3] + np.amin(bubble, axis=0)[0:3]) / 2.0
        bubble -= mid_point
        return bubble

    def jitter_bubble(self, bubble):
        n_points_in_bubble = bubble.shape[0]
        random_xyz = np.random.uniform(low=-self.model_resolution * 0.2, high=self.model_resolution * 0.2,
                                       size=(n_points_in_bubble, 3))
        bubble[:, 0:3] += random_xyz
        return bubble

    def scale_bubble(self, min_percentage, max_percentage, axis, bubble):
        """
        this function is used to scale down the bubble in either the x,y or z axis. Scaling up is not recommended since
        the maximum bubble volume is coupled with available GPU memory of the training instance.

        Parameters
        ----------
        min_percentage, max_percentage : float (0-1)
            a randomly chosen value from this range will be used to scale down the bubble
        axis : int [0-x axis, 1-y axis or 2-z axis]
            axis along which the bubble will be scaled
        """

        assert min_percentage > 0, 'Check acceptable range for input variable: min_percentage'
        assert max_percentage > 1, 'Check acceptable range for input variable: max_percentage'
        assert max_percentage > min_percentage, 'max_percentage_of_points > min_percentage_of_points'

        assert axis == 0 or axis == 1 or axis == 2, 'check accepted values for axis'

        percentage = np.random.uniform(low=min_percentage, high=max_percentage)
        bubble[:, axis] = bubble[:, axis] * percentage

        return bubble

    def rotate_bubble_y_axis(self, min_alpha, max_alpha, bubble):
        """
        rotates the bubble about the y axis with a randomly chosen angle between the provided range of
        [min_alpha (radians), max_alpha (radians)]
        """

        assert max_alpha > min_alpha, 'max_alpha > min_alpha'

        alpha = np.random.uniform(low=min_alpha, high=max_alpha)
        x, y, z = self._get_bubble_coordinates(bubble)
        min_x, max_x, min_y, max_y, min_z, max_z = self._bounding_box_calculator(bubble[:, 0:3])
        x = x - min_x
        z = z - min_z
        bubble[:, 0] = x * np.cos(alpha) + z * np.sin(alpha) + min_x
        bubble[:, 2] = - x * np.sin(alpha) + z * np.cos(alpha) + min_z

        return bubble

    def rotate_bubble_z_axis(self, min_alpha, max_alpha, bubble):
        """
        rotates the bubble about the z axis with a randomly chosen angle between the provided range of
        [min_alpha (radians), max_alpha (radians)]
        """

        assert max_alpha > min_alpha, 'max_alpha > min_alpha'

        alpha = np.random.uniform(low=min_alpha, high=max_alpha)
        x, y, z = self._get_bubble_coordinates(bubble)
        min_x, max_x, min_y, max_y, min_z, max_z = self._bounding_box_calculator(bubble[:, 0:3])
        x = x-min_x
        y = y-min_y
        bubble[:, 0] = x * np.cos(alpha) - y * np.sin(alpha) + min_x
        bubble[:, 1] = x * np.sin(alpha) + y * np.cos(alpha) + min_y

        return bubble

    def rotate_bubble_x_axis(self, min_alpha, max_alpha, bubble):
        """
        rotates the bubble about the x axis with a randomly chosen angle between the provided range of
        [min_alpha (radians), max_alpha (radians)]

        bubble: n x f
        """

        assert max_alpha > min_alpha, 'max_alpha > min_alpha'

        alpha = np.random.uniform(low=min_alpha, high=max_alpha)
        x, y, z = self._get_bubble_coordinates(bubble)
        min_x, max_x, min_y, max_y, min_z, max_z = self._bounding_box_calculator(bubble[:, 0:3])
        y = y-min_y
        z = z-min_z
        bubble[:, 1] = y * np.cos(alpha) - z * np.sin(alpha) + min_y
        bubble[:, 2] = y * np.sin(alpha) + z * np.cos(alpha) + min_z

        return bubble

    def _bounding_box_calculator(self, point_cloud):
        """
        Parameters
        ----------
        point_cloud : nPoints x 3 array containing [x,y,z] information

        Returns
        -------
        type: array of 6 floats
        Values corresponding to the dimensions of the bounding box in the order: [min_x, max_x, min_y, max_y, min_z, max_z]

        """

        # Calculating the bounding box of the point cloud
        max_bounds = np.amax(point_cloud, axis=0)
        min_bounds = np.amin(point_cloud, axis=0)

        max_x = max_bounds[0]
        min_x = min_bounds[0]
        max_y = max_bounds[1]
        min_y = min_bounds[1]
        max_z = max_bounds[2]
        min_z = min_bounds[2]

        return np.array([min_x, max_x, min_y, max_y, min_z, max_z])

    def _get_bubble_coordinates(self, bubble):
        x = bubble[:, 0]
        y = bubble[:, 1]
        z = bubble[:, 2]

        return x, y, z

