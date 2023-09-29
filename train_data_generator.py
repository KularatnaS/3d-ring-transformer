from dataset.dataset import TrainingBubblesCreator
from config.config import get_config

import os

from dataset.datautils import bubble_to_laz_file, rings_to_laz_file, visualise_individual_ring

import logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

config = get_config()

max_points_per_bubble = config["max_points_per_bubble"]
points_per_ring = config["points_per_ring"]
rings_per_bubble = config["rings_per_bubble"]
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
                           ignore_index=ignore_index, ring_padding=ring_padding)

grid_resolution = 60.0
min_rings_per_laz = rings_per_bubble

# train data
bubbles_creator.run(input_data_dir=os.path.join(laz_data_dir, 'train-data'),
                    output_data_dir=os.path.join(bubble_data_dir, 'train-bubbles'),
                    grid_resolution=grid_resolution, min_rings_per_laz=min_rings_per_laz)

# test data
bubbles_creator.run(input_data_dir=os.path.join(laz_data_dir, 'test-data'),
                    output_data_dir=os.path.join(bubble_data_dir, 'test-bubbles'),
                    grid_resolution=grid_resolution, min_rings_per_laz=min_rings_per_laz)

# val data
bubbles_creator.run(input_data_dir=os.path.join(laz_data_dir, 'val-data'),
                    output_data_dir=os.path.join(bubble_data_dir, 'val-bubbles'),
                    grid_resolution=grid_resolution, min_rings_per_laz=min_rings_per_laz)

# bubble_path = 'data/train-bubbles/pc_11_0_out_32.pt'
# bubble_to_laz_file(bubble_path, 'view_clas.laz')
# rings_to_laz_file(bubble_path, 'view_rings.laz')
# visualise_individual_ring(bubble_path, 'view_ring_0.laz', ring_index=0)
# visualise_individual_ring(bubble_path, 'view_ring_1.laz', ring_index=1)