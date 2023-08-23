import os
import glob
import math
import shutil
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch
from torch.utils.data import Dataset, DataLoader

from datautils import get_data_from_laz_file, get_down_sampled_points_and_classification, bounding_box_calculator, \
    calc_train_bubble_centres


import logging
LOGGER = logging.getLogger(__name__)


class TokenizedBubbleDataset(Dataset):
    def __init__(self, data_dir, n_classes_model):
        self.data_dir = data_dir
        self.tokenized_bubbles = glob.glob(os.path.join(data_dir, "*.pt"))
        self.n_classes_model = n_classes_model

    def __getitem__(self, index):
        # load data
        data = torch.load(self.tokenized_bubbles[index])

        point_tokens = data[0]
        point_tokens_torch = [torch.from_numpy(x.astype(np.float32)) for x in point_tokens]

        label_tokens = data[1]
        label_tokens_one_hot_encoded = []
        # one hot encode labels
        for label_token in label_tokens:
            one_hot_encoded_matrix = np.zeros((len(label_token), self.n_classes_model))
            one_hot_encoded_matrix[range(len(label_token)), label_token] = 1
            label_tokens_one_hot_encoded.append(torch.from_numpy(one_hot_encoded_matrix.astype(np.float32)))

        return point_tokens_torch, label_tokens_one_hot_encoded

    def __len__(self):
        return len(self.tokenized_bubbles)


class TrainingBubblesCreator:
    def __init__(self, max_points_per_bubble, max_points_per_ring, model_resolution):
        self.max_points_per_bubble = max_points_per_bubble
        self.max_points_per_ring = max_points_per_ring
        self.model_resolution = model_resolution

    def run(self, input_data_dir, output_data_dir, grid_resolution):
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
            if self.max_points_per_bubble > total_n_points:
                LOGGER.info(f"Total number of points in the laz file is less than max points per bubble. Skipping")
            else:
                LOGGER.info(f"Calculating nearest neighbours")
                neighbours = NearestNeighbors(n_neighbors=self.max_points_per_bubble,
                                              algorithm='auto').fit(down_sampled_points)
                bounding_box = bounding_box_calculator(down_sampled_points)
                train_bubble_centres = calc_train_bubble_centres(bounding_box, grid_resolution)

                counter = 0
                LOGGER.info(f"Tokenizing bubbles")
                for bubble_centre in train_bubble_centres:
                    LOGGER.info(f"Remaining bubbles: {len(train_bubble_centres) - counter}")
                    _, indices = neighbours.kneighbors(bubble_centre.reshape(1, -1))
                    point_tokens, label_tokens = self._split_bubble_to_rings(down_sampled_points[indices.flatten()],
                                                                             down_sampled_classification[indices.flatten()],
                                                                             bubble_centre)
                    # get name of laz file
                    laz_file_name = Path(laz_file).stem
                    save_name = os.path.join(output_data_dir, f"{laz_file_name}_{counter}.npz")
                    np.savez(save_name, point_tokens=point_tokens, label_tokens=label_tokens)
                    #torch.save((point_tokens, label_tokens), save_name)
                    counter += 1

    def _split_bubble_to_rings(self, points, classification, bubble_centre):
        point_tokens = []
        label_tokens = []

        for i in range(math.floor(len(points)/self.max_points_per_ring)):
            start_index = i * self.max_points_per_ring
            end_index = (i + 1) * self.max_points_per_ring
            if end_index > len(points):
                raise ValueError("End index is greater than the number of points in the bubble")

            point_tokens.append(points[start_index:end_index] - bubble_centre)
            label_tokens.append(classification[start_index:end_index])

        return np.asarray(point_tokens), np.asarray(label_tokens)


logging.basicConfig(level=logging.INFO)
import config
cf = config.get_config()
BT = TrainingBubblesCreator(cf["max_points_per_bubble"], cf["max_points_per_ring"], 0.08)
BT.run("train-data-mini", "train-bubbles", 15.0)

# dataset = TokenizedBubbleDataset('train-bubbles/', n_classes_model=4)
# dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=2)
#
# # get next batch
# iterator = iter(dataloader)
# point_tokens, label_tokens = next(iterator)
# print(point_tokens[0])
# print(point_tokens[0].shape)

from einops import rearrange
