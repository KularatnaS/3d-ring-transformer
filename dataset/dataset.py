import os
import glob
import math
import shutil
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch
from torch.utils.data import Dataset

from dataset.datautils import get_data_from_laz_file, get_down_sampled_points_and_classification, \
    bounding_box_calculator, calc_train_bubble_centres


import logging
LOGGER = logging.getLogger(__name__)


def collate_fn(batch):
    """
    batch is a list of tuples
    each tuple is of the form (point_tokens, label_tokens, n_missing_rings)
    Example batch -> [(), (), (), ...]
    """

    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    n_missing_rings = [item[2] for item in batch]

    return torch.stack(data), torch.stack(target), torch.stack(n_missing_rings)


class TokenizedBubbleDataset(Dataset):
    def __init__(self, data_dir, n_classes_model, rings_per_bubble):
        self.data_dir = data_dir
        self.tokenized_bubbles = glob.glob(os.path.join(data_dir, "*.pt"))
        self.n_classes_model = n_classes_model
        self.rings_per_bubble = rings_per_bubble

    def __getitem__(self, index):
        # load data
        data = torch.load(self.tokenized_bubbles[index])

        point_tokens = torch.from_numpy(data[0].astype(np.float32))
        assert point_tokens.shape[0] == self.rings_per_bubble

        n_missing_rings = torch.tensor([data[2]], dtype=torch.int32)
        encoder_mask = torch.ones(self.rings_per_bubble, dtype=torch.int32)

        if n_missing_rings > 0:
            encoder_mask[-n_missing_rings:] = 0
        encoder_mask = encoder_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, rings_per_bubble]

        label_tokens = data[1]
        label_tokens_one_hot_encoded = []

        # one hot encode labels
        for counter, label_token in enumerate(label_tokens):
            one_hot_encoded_matrix = np.zeros((len(label_token), self.n_classes_model))
            if counter < self.rings_per_bubble - n_missing_rings:
                one_hot_encoded_matrix[range(len(label_token)), label_token.astype(int)] = 1
            label_tokens_one_hot_encoded.append(one_hot_encoded_matrix.astype(np.float32))

        return point_tokens, torch.from_numpy(np.asarray(label_tokens_one_hot_encoded)), encoder_mask

    def __len__(self):
        return len(self.tokenized_bubbles)


class TrainingBubblesCreator:
    def __init__(self, max_points_per_bubble, points_per_ring, rings_per_bubble, n_point_features, model_resolution):
        self.max_points_per_bubble = max_points_per_bubble
        self.points_per_ring = points_per_ring
        self.rings_per_bubble = rings_per_bubble
        self.n_point_features = n_point_features
        self.model_resolution = model_resolution

    def run(self, input_data_dir, output_data_dir, grid_resolution, min_rings_per_laz=3):
        # check if output directory exists, and if so, delete it and recreate it
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)
        os.mkdir(output_data_dir)

        laz_files = glob.glob(os.path.join(input_data_dir, "*.laz"))
        for laz_file in laz_files:
            LOGGER.info(f"Processing {laz_file}")
            LOGGER.info(f"Reading points")
            points, classification = get_data_from_laz_file(laz_file, classification=True)
            LOGGER.info(f"Down sampling points")
            down_sampled_points, down_sampled_classification = \
                get_down_sampled_points_and_classification(points, classification, self.model_resolution)
            del points, classification

            total_n_points = down_sampled_points.shape[0]
            if total_n_points > min_rings_per_laz * self.points_per_ring:
                if self.max_points_per_bubble > total_n_points:
                    n_neighbours = total_n_points
                else:
                    n_neighbours = self.max_points_per_bubble

                LOGGER.info(f"Calculating nearest neighbours")
                neighbours = NearestNeighbors(n_neighbors=n_neighbours,
                                              algorithm='auto').fit(down_sampled_points)
                bounding_box = bounding_box_calculator(down_sampled_points)
                train_bubble_centres = calc_train_bubble_centres(bounding_box, grid_resolution)

                counter = 0
                LOGGER.info(f"Tokenizing bubbles")
                for bubble_centre in train_bubble_centres:
                    LOGGER.info(f"Remaining bubbles: {len(train_bubble_centres) - counter}")
                    _, indices = neighbours.kneighbors(bubble_centre.reshape(1, -1))
                    point_tokens, label_tokens, n_missing_rings = \
                        self._split_bubble_to_rings(down_sampled_points[indices.flatten()],
                                                    down_sampled_classification[indices.flatten()], bubble_centre)
                    # get name of laz file
                    laz_file_name = Path(laz_file).stem
                    save_name = os.path.join(output_data_dir, f"{laz_file_name}_{counter}.pt")
                    torch.save((point_tokens, label_tokens, n_missing_rings), save_name)
                    counter += 1
            else:
                LOGGER.info(f"Skipping {laz_file} because it has too few points")

    def _split_bubble_to_rings(self, points, classification, bubble_centre):
        point_tokens = np.zeros((self.rings_per_bubble, self.points_per_ring, self.n_point_features))
        label_tokens = np.zeros((self.rings_per_bubble, self.points_per_ring))

        n_valid_rings = math.floor(len(points)/self.points_per_ring)
        n_missing_rings = self.rings_per_bubble - n_valid_rings

        for i in range(n_valid_rings):
            start_index = i * self.points_per_ring
            end_index = (i + 1) * self.points_per_ring
            if end_index > len(points):
                raise ValueError("end_index is greater than the number of points in the bubble")

            point_tokens[i] = points[start_index:end_index] - bubble_centre
            label_tokens[i] = classification[start_index:end_index]

        return point_tokens, label_tokens, n_missing_rings
