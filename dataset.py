import laspy
import numpy as np
from datautils import get_data_from_laz_file, down_sample_point_data, get_down_sampled_points_and_classification


class PointTokenizer:
    def __init__(self, max_points_per_bubble, rings_per_bubble, points_per_ring, model_resolution):
        self.max_points_per_bubble = max_points_per_bubble
        self.rings_per_bubble = rings_per_bubble
        self.points_per_ring = points_per_ring
        self.model_resolution = model_resolution


