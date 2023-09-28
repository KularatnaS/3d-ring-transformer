import os
import glob
import math
import shutil
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch
from torch.utils.data import Dataset

from einops import rearrange

from dataset.datautils import get_data_from_laz_file, get_down_sampled_points_and_classification, \
    bounding_box_calculator, calc_train_bubble_centres


import logging
LOGGER = logging.getLogger(__name__)

np.random.seed(0)


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
        assert point_tokens.shape[0] == self.rings_per_bubble, f'shape of point_tokens is {point_tokens.shape}'

        n_missing_rings = torch.tensor([data[2]], dtype=torch.int32)
        encoder_mask = torch.ones(self.rings_per_bubble, dtype=torch.int32)

        if n_missing_rings > 0:
            encoder_mask[-n_missing_rings:] = 0
        encoder_mask = encoder_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, rings_per_bubble]

        label_tokens = data[1]
        label_tokens = torch.from_numpy(label_tokens.astype(np.float32))

        # # one hot encode labels
        # for counter, label_token in enumerate(label_tokens):
        #     one_hot_encoded_matrix = np.zeros((len(label_token), self.n_classes_model))
        #     if counter < self.rings_per_bubble - n_missing_rings:
        #         one_hot_encoded_matrix[range(len(label_token)), label_token.astype(int)] = 1
        #     label_tokens_one_hot_encoded.append(one_hot_encoded_matrix.astype(np.float32))

        return point_tokens, label_tokens, encoder_mask

    def __len__(self):
        return len(self.tokenized_bubbles)


class TrainingBubblesCreator:
    def __init__(self, max_points_per_bubble, points_per_ring, rings_per_bubble, n_point_features, model_resolution,
                 n_classes_model, ignore_index, ring_padding):
        assert ring_padding >= 0

        self.max_points_per_bubble = max_points_per_bubble
        self.max_points_sampled_per_bubble = int(max_points_per_bubble * (1 + ring_padding))
        assert self.max_points_sampled_per_bubble >= max_points_per_bubble

        self.points_per_ring = points_per_ring
        self.dense_points_per_ring = int(points_per_ring * (1 - ring_padding))
        assert self.dense_points_per_ring <= points_per_ring
        self.sparse_points_per_ring = points_per_ring - self.dense_points_per_ring

        self.rings_per_bubble = rings_per_bubble
        self.n_point_features = n_point_features
        self.model_resolution = model_resolution
        self.n_classes_model = n_classes_model
        self.ignore_index = ignore_index
        self.ring_padding = ring_padding

    def run(self, input_data_dir, output_data_dir, grid_resolution, min_rings_per_laz=3):
        # check if output directory exists, and if so, delete it and recreate it
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)
        os.mkdir(output_data_dir)

        laz_files = glob.glob(os.path.join(input_data_dir, "*.laz"))
        for laz_file in laz_files:
            LOGGER.info(f"Processing {laz_file}")
            LOGGER.info(f"Reading points")
            points, classification = get_data_from_laz_file(laz_file, n_classes_model=self.n_classes_model,
                                                            classification=True)
            LOGGER.info(f"Down sampling points")
            down_sampled_points, down_sampled_classification = \
                get_down_sampled_points_and_classification(points, classification, self.model_resolution)
            del points, classification

            total_n_points_pc = down_sampled_points.shape[0]
            if total_n_points_pc > min_rings_per_laz * self.points_per_ring:
                if self.max_points_sampled_per_bubble > total_n_points_pc:
                    n_neighbours = total_n_points_pc
                else:
                    n_neighbours = self.max_points_sampled_per_bubble

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
                                                    down_sampled_classification[indices.flatten()])
                    # get name of laz file
                    laz_file_name = Path(laz_file).stem
                    save_name = os.path.join(output_data_dir, f"{laz_file_name}_{counter}.pt")
                    torch.save((point_tokens, label_tokens, n_missing_rings), save_name)
                    counter += 1
            else:
                LOGGER.info(f"Skipping {laz_file} because it has too few points")

    def _split_bubble_to_rings(self, points, classification):

        # normalised coordinates of points in bubble
        mid_point = (np.amax(points, axis=0)[0:3] + np.amin(points, axis=0)[0:3]) / 2.0
        neighbours = NearestNeighbors(n_neighbors=len(points),
                                      algorithm='auto').fit(points)
        _, indices = neighbours.kneighbors(mid_point.reshape(1, -1))
        points = points[indices.flatten()]
        points = points - mid_point

        classification = classification[indices.flatten()]

        point_tokens = np.zeros((self.rings_per_bubble, self.points_per_ring, self.n_point_features))
        label_tokens = np.zeros((self.rings_per_bubble, self.points_per_ring))

        n_valid_rings = math.floor(len(points)/self.points_per_ring)
        if n_valid_rings >= self.rings_per_bubble:
            n_valid_rings = self.rings_per_bubble

        if n_valid_rings > self.rings_per_bubble:
            n_missing_rings = 0
        else:
            n_missing_rings = self.rings_per_bubble - n_valid_rings
        assert n_missing_rings >= 0

        for i in range(n_valid_rings):
            # get dense points of the ring
            start_index = i * self.dense_points_per_ring
            end_index = (i + 1) * self.dense_points_per_ring
            if end_index > len(points):
                raise ValueError("end_index is greater than the number of points in the bubble")

            point_tokens[i, 0:self.dense_points_per_ring] = points[start_index:end_index]
            label_tokens[i, 0:self.dense_points_per_ring] = classification[start_index:end_index]

            if n_valid_rings > 1:
                # get sparse points of the ring - points with indices outside the range [start_index, end_index]
                surrounding_ring_indices = np.delete(np.arange(n_valid_rings), i)
                surrounding_ring_distance = np.abs(surrounding_ring_indices - i)
                surrounding_ring_distance = surrounding_ring_distance.astype(float)
                surrounding_ring_distance /= np.max(surrounding_ring_distance)
                fraction_of_points_from_surrounding_ring = \
                    np.exp(-np.power(surrounding_ring_distance, 4) / (2 * np.power(0.5, 4)))
                fraction_of_points_from_surrounding_ring /= np.sum(fraction_of_points_from_surrounding_ring)
                n_sparse_points_sampled = 0

                n_surrounding_rings = len(surrounding_ring_indices)
                for j in range(n_surrounding_rings):

                    if j == n_surrounding_rings - 1:
                        n_points_to_sample = self.sparse_points_per_ring - n_sparse_points_sampled
                    else:
                        n_points_to_sample = int(fraction_of_points_from_surrounding_ring[j] * self.sparse_points_per_ring)

                    # use np.random.choice to sample points from the surrounding rings
                    indices_of_points_to_sample = np.arange(surrounding_ring_indices[j] * self.dense_points_per_ring,
                                                            (surrounding_ring_indices[j] + 1) * self.dense_points_per_ring)
                    if len(indices_of_points_to_sample) < n_points_to_sample:
                        sparse_points_indices = np.random.choice(len(self.dense_points_per_ring), n_points_to_sample,
                                                                 replace=True)
                    else:
                        sparse_points_indices = np.random.choice(indices_of_points_to_sample, n_points_to_sample,
                                                                 replace=False)
                    point_tokens[i, (self.dense_points_per_ring + n_sparse_points_sampled):
                                    (self.dense_points_per_ring + n_sparse_points_sampled + n_points_to_sample)] = \
                        points[sparse_points_indices]
                    n_sparse_points_sampled += n_points_to_sample

            label_tokens[i, self.dense_points_per_ring:] = self.ignore_index

        return point_tokens, label_tokens, n_missing_rings
