from dataset.dataset import TrainingBubblesCreator
from config.config import get_config

import os

from dataset.datautils import bubble_to_laz_file, rings_to_laz_file, visualise_individual_ring

import torch
import numpy as np

import logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

np.random.seed(0)
torch.manual_seed(0)

config = get_config()

max_points_per_bubble = config["max_points_per_bubble"]
points_per_ring = config["points_per_ring"]
rings_per_bubble = config["rings_per_bubble"]
extra_rings_for_last_ring_padding = config["extra_rings_for_last_ring_padding"]
n_point_features = config["n_point_features"]
model_resolution = config["model_resolution"]
n_classes_model = config["n_classes_model"]
ignore_index = config["ignore_index"]
ring_padding = config["ring_padding"]
laz_data_dir = config["laz_data_dir"]
bubble_data_dir = config["bubble_data_dir"]

bubbles_creator = \
    TrainingBubblesCreator(max_points_per_bubble=max_points_per_bubble, points_per_ring=points_per_ring,
                           rings_per_bubble=rings_per_bubble, n_point_features=n_point_features,
                           model_resolution=model_resolution, n_classes_model=n_classes_model,
                           ignore_index=ignore_index, ring_padding=ring_padding,
                           extra_rings_for_last_ring_padding=extra_rings_for_last_ring_padding)

grid_resolution = 120.0
min_rings_per_laz = rings_per_bubble + extra_rings_for_last_ring_padding

# clean train, val and test data directories
train_bubbles_dir = os.path.join(bubble_data_dir, 'train-bubbles')
test_bubbles_dir = os.path.join(bubble_data_dir, 'test-bubbles')
val_bubbles_dir = os.path.join(bubble_data_dir, 'val-bubbles')
for directory in [train_bubbles_dir, test_bubbles_dir, val_bubbles_dir]:
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))


# train data
bubbles_creator.run(input_data_dir=os.path.join(laz_data_dir, 'train-data'),
                    output_data_dir=train_bubbles_dir,
                    grid_resolution=grid_resolution, min_rings_per_laz=min_rings_per_laz)

# test data
bubbles_creator.run(input_data_dir=os.path.join(laz_data_dir, 'test-data'),
                    output_data_dir=test_bubbles_dir,
                    grid_resolution=grid_resolution, min_rings_per_laz=min_rings_per_laz)

# val data
bubbles_creator.run(input_data_dir=os.path.join(laz_data_dir, 'val-data'),
                    output_data_dir=val_bubbles_dir,
                    grid_resolution=grid_resolution, min_rings_per_laz=min_rings_per_laz)

bubble_path = 'data/bubbles/train-bubbles/5080_54435_out_30.pt'
bubble_to_laz_file(bubble_path, 'view_clas.laz', ignore_index=ignore_index, n_classes_model=n_classes_model)
rings_to_laz_file(bubble_path, 'view_rings.laz')
visualise_individual_ring(bubble_path, 'view_ring_0.laz', ring_index=0)
visualise_individual_ring(bubble_path, 'view_ring_1.laz', ring_index=1)

# # dummy train data_generator
# point_tokens = np.random.uniform(low=-100.0, high=100.0, size=(rings_per_bubble, points_per_ring, n_point_features))
# label_tokens = np.random.randint(0, n_classes_model, size=(rings_per_bubble, points_per_ring))
# n_missing_rings = 0
# save_names = [os.path.join(train_bubbles_dir, 'dummy_train_data.pt'),
#               os.path.join(test_bubbles_dir, 'dummy_test_data.pt'),
#               os.path.join(val_bubbles_dir, 'dummy_val_data.pt')]
# for save_name in save_names:
#     torch.save((point_tokens, label_tokens, n_missing_rings), save_name)
