from dataset.dataset import TrainingBubblesCreator
from config.config import get_config

from dataset.datautils import bubble_to_laz_file

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

training_bubbles_creator = \
    TrainingBubblesCreator(max_points_per_bubble=max_points_per_bubble, points_per_ring=points_per_ring,
                           rings_per_bubble=rings_per_bubble, n_point_features=n_point_features,
                           model_resolution=model_resolution, n_classes_model=n_classes_model)

input_data_dir = 'data/train-data/'
output_data_dir = 'data/train-bubbles/'
grid_resolution = 30.0
min_rings_per_laz = 3

# training_bubbles_creator.run(input_data_dir=input_data_dir, output_data_dir=output_data_dir,
#                                 grid_resolution=grid_resolution, min_rings_per_laz=min_rings_per_laz)

bubble_to_laz_file('data/train-bubbles/pc_146_0_out_1.pt', 'view.laz')